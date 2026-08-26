"""Benchmark Kimi-K3 NVFP4 EP kernels at global decode batch sizes.

This benchmark compares the fused FlashInfer MegaMoE candidate with the
TensorRT-LLM NVFP4 routed-MoE baseline.  The candidate can be run with its
integrated shared FC1+FC2 work or routed-only, making the extra fused work
explicit instead of comparing unlike launch boundaries silently.

Launch one process per GPU.  Kimi-K3 DEP16 on GB300 uses four 4-GPU nodes::

    torchrun --nnodes=4 --nproc-per-node=4 benchmarks/bench_kimi_k3_nvfp4_ep.py \
        --provider mega-fused

Global token counts are distributed over EP ranks using the same rectangular
padding required by data-parallel decode.  Rows beyond the requested global
count are assigned no routed experts.  CSV output records both requested and
padded token counts. CUDA events separately measure the complete provider
invocation and, for MegaMoE, the persistent kernel. Synchronized wall time is
reported separately to expose launch overhead.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import os
import statistics
import time
from typing import Any


DEFAULT_GLOBAL_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


@dataclasses.dataclass
class CaseTensors:
    """Provider-neutral subset of the MegaMoE per-iteration tensor bundle."""

    hidden_states: Any
    topk_ids: Any
    topk_weights: Any
    scales: Any = None
    fc1_alpha: Any = None
    fc2_alpha: Any = None
    fc1_norm_const: Any = None
    recv_count: Any = None
    num_tokens_per_expert: Any = None
    shared_hidden_states: Any = None
    shared_expert_weights: Any = None
    shared_expert_output: Any = None

    @property
    def num_tokens(self) -> int:
        return self.hidden_states.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=("mega-fused", "mega-routed", "trtllm"),
        required=True,
    )
    parser.add_argument(
        "--global-tokens",
        type=int,
        nargs="+",
        default=DEFAULT_GLOBAL_TOKENS,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--expected-world-size", type=int, default=16)
    parser.add_argument("--output-jsonl")
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Time CUDA graph replays instead of eager layer invocations.",
    )
    parser.add_argument(
        "--profile-cuda-range",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--skip-component-timing",
        action="store_true",
        help="Measure only the complete provider invocation.",
    )
    return parser.parse_args()


def production_knobs() -> dict:
    return {
        "cluster_shape_mnk": (2, 1, 1),
        "group_hint": 512,
        "epi_flag_batch": (2, 4),
        "load_balance_mode": "atomic_counter",
        "mma_tiler_mnk": (256, 128, 256),
        "flag_batch": 4,
        "token_back_mode": "standalone_warps",
    }


def make_weights(rank: int, local_experts: int, hidden: int, intermediate: int):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(1701 + rank)
    w13 = (
        torch.randn(
            local_experts,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 64
    )
    w2 = (
        torch.randn(
            local_experts,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 64
    )
    return w13, w2


def make_mega_layer(
    rank: int,
    world_size: int,
    max_tokens_per_rank: int,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    num_experts = 896
    hidden = 3584
    intermediate = 3072
    local_experts = num_experts // world_size
    w13, w2 = make_weights(rank, local_experts, hidden, intermediate)
    kernel = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=16,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        in_kernel_fc2_reduce=True,
        combine_dtype="bf16",
        shared_hidden_size=7168,
        shared_intermediate_size=6144,
        knobs=production_knobs(),
    )
    return MoEEpLayer(
        BootstrapConfig(world_size=world_size, rank=rank),
        FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens_per_rank,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(megakernel=kernel),
    )


def make_trtllm_layer(rank: int, world_size: int, max_tokens_per_rank: int):
    import torch

    from flashinfer.fused_moe.api import TrtllmFp4Config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        FusedMoEQuantConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
        TrtLlmNvFp4ExpertsModular,
    )

    num_experts = 896
    hidden = 3584
    intermediate = 3072
    local_experts = num_experts // world_size
    w13, w2 = make_weights(rank, local_experts, hidden, intermediate)
    weight_view = TrtllmFp4Config.prepare_weights(
        w13,
        w2,
        num_local_experts=local_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        device=w13.device,
    )
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype="nvfp4",
        w1_scale=weight_view["gemm1_weights_scale"],
        w2_scale=weight_view["gemm2_weights_scale"],
        g1_alphas=weight_view["output1_scale_gate_scalar"],
        g2_alphas=weight_view["output2_scale_scalar"],
        a2_gscale=weight_view["output1_scale_scalar"],
        is_scale_swizzled=False,
    )
    parallel_config = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=world_size,
        ep_size=world_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=rank,
        ep_rank=rank,
        sp_size=1,
        use_ep=True,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=16,
        hidden_dim=hidden,
        intermediate_size=intermediate,
        num_local_experts=local_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SITU,
        device="cuda",
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        routing_method=RoutingMethodType.TopK,
        max_num_tokens=max_tokens_per_rank * world_size,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    experts = TrtLlmNvFp4ExpertsModular(
        moe_config=moe_config, quant_config=quant_config
    )
    return TrtllmAgRsLayer(
        rank=rank,
        world_size=world_size,
        local_experts=local_experts,
        hidden=hidden,
        experts=experts,
        weight_view=weight_view,
        activation_global_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
    )


class TrtllmAgRsLayer:
    """TRT-LLM routed MoE behind Kimi's rectangular AG/RS EP protocol."""

    supports_output_view = False

    def __init__(
        self,
        *,
        rank,
        world_size,
        local_experts,
        hidden,
        experts,
        weight_view,
        activation_global_scale,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.local_experts = local_experts
        self.hidden = hidden
        self.experts = experts
        self.weight_view = weight_view
        self.activation_global_scale = activation_global_scale
        self._buffers = {}

    def _buffers_for(self, tokens_per_rank: int, q, scale):
        import torch

        buffers = self._buffers.get(tokens_per_rank)
        if buffers is None:
            global_tokens = tokens_per_rank * self.world_size
            buffers = {
                "q": torch.empty(
                    global_tokens,
                    *q.shape[1:],
                    dtype=q.dtype,
                    device="cuda",
                ),
                "scale": torch.empty(
                    global_tokens,
                    *scale.shape[1:],
                    dtype=scale.dtype,
                    device="cuda",
                ),
                "ids": torch.empty(global_tokens, 16, dtype=torch.int32, device="cuda"),
                "weights": torch.empty(
                    global_tokens, 16, dtype=torch.float32, device="cuda"
                ),
                "output": torch.empty(
                    tokens_per_rank,
                    self.hidden,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                "partial": torch.empty(
                    global_tokens,
                    self.hidden,
                    dtype=torch.bfloat16,
                    device="cuda",
                ),
                "workspace13": torch.empty(0, dtype=torch.uint8, device="cuda"),
                "workspace2": torch.empty(0, dtype=torch.uint8, device="cuda"),
            }
            self._buffers[tokens_per_rank] = buffers
        return buffers

    def forward(self, tensors):
        import torch
        import torch.distributed as dist

        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        tokens_per_rank = tensors.hidden_states.shape[0]
        q, scale = ops.scaled_fp4_quant(
            tensors.hidden_states,
            self.activation_global_scale,
            is_sf_swizzled_layout=False,
        )
        buffers = self._buffers_for(tokens_per_rank, q, scale)
        dist.all_gather_single(buffers["q"], q)
        dist.all_gather_single(buffers["scale"], scale)
        dist.all_gather_single(buffers["ids"], tensors.topk_ids.to(torch.int32))
        dist.all_gather_single(buffers["weights"], tensors.topk_weights)

        offset = self.rank * self.local_experts
        ids = buffers["ids"]
        is_local = (ids >= offset) & (ids < offset + self.local_experts)
        local_ids = torch.where(is_local, ids, torch.full_like(ids, offset))
        local_weights = torch.where(
            is_local, buffers["weights"], torch.zeros_like(buffers["weights"])
        )
        self.experts.apply(
            output=buffers["partial"],
            hidden_states=buffers["q"],
            w1=self.weight_view["gemm1_weights"],
            w2=self.weight_view["gemm2_weights"],
            topk_weights=local_weights,
            topk_ids=local_ids,
            activation=MoEActivation.SITU,
            global_num_experts=896,
            expert_map=None,
            a1q_scale=buffers["scale"],
            a2_scale=None,
            workspace13=buffers["workspace13"],
            workspace2=buffers["workspace2"],
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        dist.reduce_scatter_single(buffers["output"], buffers["partial"])
        return buffers["output"]

    def destroy(self) -> None:
        self._buffers.clear()


def make_shared_weights():
    import torch

    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_nvfp4_cutedsl_mega_weights,
    )

    hidden = 7168
    intermediate = 6144
    generator = torch.Generator(device="cuda").manual_seed(2701)
    w13 = (
        torch.randn(
            1,
            2 * intermediate,
            hidden,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 64
    )
    w2 = (
        torch.randn(
            1,
            hidden,
            intermediate,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        / 64
    )
    return preprocess_nvfp4_cutedsl_mega_weights(
        MoEWeightPack(w13=w13, w2=w2),
        intermediate_size=intermediate,
        hidden_size=hidden,
    )


def make_case(rank: int, world_size: int, global_tokens: int, shared_weights):
    import torch

    tokens_per_rank = math.ceil(global_tokens / world_size)
    rank_start = rank * tokens_per_rank
    active_tokens = max(0, min(tokens_per_rank, global_tokens - rank_start))
    generator = torch.Generator(device="cuda").manual_seed(3701 + global_tokens + rank)
    hidden = torch.randn(
        tokens_per_rank,
        3584,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    scores = torch.randn(
        tokens_per_rank,
        896,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    topk_weights, topk_ids = torch.topk(scores, 16, dim=-1, sorted=False)
    topk_weights = torch.softmax(topk_weights, dim=-1)
    if active_tokens < tokens_per_rank:
        topk_ids[active_tokens:].fill_(-1)
        topk_weights[active_tokens:].zero_()
    local_experts = 896 // world_size
    fc1_alpha = torch.ones(local_experts, dtype=torch.float32, device="cuda")
    fc2_alpha = torch.ones(local_experts, dtype=torch.float32, device="cuda")
    fc1_norm_const = torch.ones(local_experts, dtype=torch.float32, device="cuda")
    shared_hidden_states = None
    shared_expert_output = None
    if shared_weights is not None:
        shared_hidden_states = torch.randn(
            tokens_per_rank,
            7168,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        shared_expert_output = torch.empty_like(shared_hidden_states)
    tensors = CaseTensors(
        hidden_states=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        shared_hidden_states=shared_hidden_states,
        shared_expert_weights=shared_weights,
        shared_expert_output=shared_expert_output,
    )
    return tensors, active_tokens, tokens_per_rank


def max_rank_time_ms(local_ms: float) -> float:
    import torch
    import torch.distributed as dist

    value = torch.tensor(local_ms, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def all_rank_values(local_value: float) -> list[float]:
    import torch
    import torch.distributed as dist

    value = torch.tensor(local_value, dtype=torch.float64, device="cuda")
    values = torch.empty(dist.get_world_size(), dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(values, value)
    return values.cpu().tolist()


def sum_rank_value(local_value: float) -> float:
    import torch
    import torch.distributed as dist

    value = torch.tensor(local_value, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value.item()


def make_mega_kernel_thunks(layer, num_tokens: int):
    """Build separate reset and persistent-launch thunks for a warmed layer."""
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import MegaMoENvfp4Inputs

    workspace = layer._workspace
    transformed = layer._transformed
    if workspace is None or transformed is None:
        raise RuntimeError("MegaMoE layer must be warmed before kernel timing")
    active_epilogue = getattr(layer._kernel, "_active_epilogue", None)
    if active_epilogue is None:
        active_epilogue = (
            workspace.fc1_alpha,
            workspace.fc2_alpha,
            workspace.fc1_norm_const,
        )
    fc1_alpha, fc2_alpha, fc1_norm_const = active_epilogue
    inputs = MegaMoENvfp4Inputs(
        activation=workspace.x,
        activation_sf=workspace.x_sf,
        topk_idx=workspace.topk_idx,
        topk_weights=workspace.topk_weights,
        fc1_weight=transformed[0][0],
        fc1_weight_sf=transformed[0][1],
        fc2_weight=transformed[1][0],
        fc2_weight_sf=transformed[1][1],
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
        output_activation=workspace.output_activation,
        shared=workspace._shared_inputs,
    )
    clear_tokens = min(
        ((max(num_tokens, 1) + 63) // 64) * 64,
        workspace.x.shape[0],
    )
    return workspace._frontend.make_launch_thunks(
        inputs,
        zero_num_tokens=clear_tokens,
    )


def measure_gpu(
    invoke,
    *,
    prepare,
    warmup: int,
    repeat: int,
    cuda_graph: bool,
    capture_invoke_factory=None,
) -> tuple[list[float], list[float], Any]:
    """Measure local and slowest-EP-rank CUDA latency for one callable."""
    import torch
    import torch.distributed as dist

    output = None
    for _ in range(warmup):
        prepare()
        output = invoke()
    torch.cuda.synchronize()
    dist.barrier()

    graph = None
    graph_start = None
    graph_end = None
    if cuda_graph:
        prepare()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        graph_start = torch.cuda.Event(enable_timing=True, external=True)
        graph_end = torch.cuda.Event(enable_timing=True, external=True)
        with torch.cuda.graph(graph):
            graph_invoke = (
                capture_invoke_factory()
                if capture_invoke_factory is not None
                else invoke
            )
            graph_start.record()
            output = graph_invoke()
            graph_end.record()
        torch.cuda.synchronize()
        dist.barrier()

    eager_start = torch.cuda.Event(enable_timing=True)
    eager_end = torch.cuda.Event(enable_timing=True)
    local_samples = []
    critical_samples = []
    for _ in range(repeat):
        prepare()
        torch.cuda.synchronize()
        dist.barrier()
        if graph is None:
            eager_start.record()
            output = invoke()
            eager_end.record()
            eager_end.synchronize()
            local_ms = eager_start.elapsed_time(eager_end)
        else:
            graph.replay()
            assert graph_start is not None and graph_end is not None
            graph_end.synchronize()
            local_ms = graph_start.elapsed_time(graph_end)
        local_samples.append(local_ms)
        critical_samples.append(max_rank_time_ms(local_ms))
    return local_samples, critical_samples, output


def run_case(
    layer,
    tensors,
    warmup: int,
    repeat: int,
    profile_cuda_range: bool,
    cuda_graph: bool,
    skip_component_timing: bool,
) -> dict:
    import torch
    import torch.distributed as dist

    def invoke():
        if layer.supports_output_view:
            return layer.forward(tensors, return_workspace_view=True)
        return layer.forward(tensors)

    invocation_local, invocation_critical, output = measure_gpu(
        invoke,
        prepare=lambda: None,
        warmup=warmup,
        repeat=repeat,
        cuda_graph=cuda_graph,
    )

    kernel_local = None
    kernel_critical = None
    preparation_local = None
    preparation_critical = None
    if layer.supports_output_view and not skip_component_timing:
        quantize_input = layer._resolve_quantize_input(tensors)

        def prepare_only():
            layer._kernel.stage_inputs(
                tensors,
                layer._workspace,
                quantize_input=quantize_input,
            )
            prepare_launch, _ = make_mega_kernel_thunks(layer, tensors.num_tokens)
            prepare_launch()

        preparation_local, preparation_critical, _ = measure_gpu(
            prepare_only,
            prepare=lambda: None,
            warmup=warmup,
            repeat=repeat,
            cuda_graph=cuda_graph,
        )
        prepare_kernel, launch_kernel = make_mega_kernel_thunks(
            layer, tensors.num_tokens
        )
        kernel_local, kernel_critical, _ = measure_gpu(
            launch_kernel,
            prepare=prepare_kernel,
            warmup=warmup,
            repeat=repeat,
            cuda_graph=cuda_graph,
            capture_invoke_factory=lambda: make_mega_kernel_thunks(
                layer, tensors.num_tokens
            )[1],
        )

    wall_samples = []
    if profile_cuda_range:
        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(repeat):
        dist.barrier()
        torch.cuda.synchronize()
        wall_started = time.perf_counter()
        output = invoke()
        torch.cuda.synchronize()
        wall_samples.append(
            max_rank_time_ms((time.perf_counter() - wall_started) * 1e3)
        )
    if profile_cuda_range:
        torch.cuda.cudart().cudaProfilerStop()
    assert output is not None
    assert torch.isfinite(output).all()
    if tensors.shared_expert_output is not None:
        assert torch.isfinite(tensors.shared_expert_output).all()
    output_float = output.float()
    output_sum = sum_rank_value(output_float.sum().item())
    output_l2_sq = sum_rank_value(output_float.square().sum().item())
    shared_output_sum = None
    shared_output_l2_sq = None
    if tensors.shared_expert_output is not None:
        shared_float = tensors.shared_expert_output.float()
        shared_output_sum = sum_rank_value(shared_float.sum().item())
        shared_output_l2_sq = sum_rank_value(shared_float.square().sum().item())
    invocation_local.sort()
    invocation_critical.sort()
    if kernel_local is not None:
        kernel_local.sort()
        assert kernel_critical is not None
        kernel_critical.sort()
        assert preparation_local is not None
        assert preparation_critical is not None
        preparation_local.sort()
        preparation_critical.sort()
    wall_samples.sort()

    def percentile(samples, fraction):
        return samples[max(0, math.ceil(fraction * len(samples)) - 1)]

    invocation_local_p50 = statistics.median(invocation_local)
    rank_invocation_p50 = sorted(all_rank_values(invocation_local_p50))
    result = {
        "invocation_gpu_latency_ms_p50": statistics.median(invocation_critical),
        "invocation_gpu_latency_ms_p10": percentile(invocation_critical, 0.10),
        "invocation_gpu_latency_ms_p90": percentile(invocation_critical, 0.90),
        "local_invocation_gpu_latency_ms_p50": invocation_local_p50,
        "rank_invocation_gpu_latency_ms_p50_min": rank_invocation_p50[0],
        "rank_invocation_gpu_latency_ms_p50_median": statistics.median(
            rank_invocation_p50
        ),
        "rank_invocation_gpu_latency_ms_p50_max": rank_invocation_p50[-1],
        "wall_latency_ms_p50": statistics.median(wall_samples),
        "wall_latency_ms_p10": percentile(wall_samples, 0.10),
        "wall_latency_ms_p90": percentile(wall_samples, 0.90),
        "output_sum": output_sum,
        "output_l2_sq": output_l2_sq,
        "shared_output_sum": shared_output_sum,
        "shared_output_l2_sq": shared_output_l2_sq,
        "cuda_graph": cuda_graph,
        "samples": len(invocation_critical),
    }
    if kernel_critical is not None:
        preparation_local_p50 = statistics.median(preparation_local)
        kernel_local_p50 = statistics.median(kernel_local)
        rank_preparation_p50 = sorted(all_rank_values(preparation_local_p50))
        rank_kernel_p50 = sorted(all_rank_values(kernel_local_p50))
        result.update(
            preparation_gpu_latency_ms_p50=statistics.median(preparation_critical),
            preparation_gpu_latency_ms_p10=percentile(preparation_critical, 0.10),
            preparation_gpu_latency_ms_p90=percentile(preparation_critical, 0.90),
            local_preparation_gpu_latency_ms_p50=preparation_local_p50,
            rank_preparation_gpu_latency_ms_p50_min=rank_preparation_p50[0],
            rank_preparation_gpu_latency_ms_p50_median=statistics.median(
                rank_preparation_p50
            ),
            rank_preparation_gpu_latency_ms_p50_max=rank_preparation_p50[-1],
            persistent_kernel_gpu_latency_ms_p50=statistics.median(kernel_critical),
            persistent_kernel_gpu_latency_ms_p10=percentile(kernel_critical, 0.10),
            persistent_kernel_gpu_latency_ms_p90=percentile(kernel_critical, 0.90),
            local_persistent_kernel_gpu_latency_ms_p50=kernel_local_p50,
            rank_persistent_kernel_gpu_latency_ms_p50_min=rank_kernel_p50[0],
            rank_persistent_kernel_gpu_latency_ms_p50_median=statistics.median(
                rank_kernel_p50
            ),
            rank_persistent_kernel_gpu_latency_ms_p50_max=rank_kernel_p50[-1],
        )
    return result


def main() -> int:
    args = parse_args()
    if any(tokens <= 0 for tokens in args.global_tokens):
        raise ValueError("global token counts must be positive")

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
        timeout=datetime.timedelta(minutes=30),
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.expected_world_size:
        raise ValueError(f"expected EP{args.expected_world_size}, got EP{world_size}")

    max_tokens_per_rank = max(math.ceil(t / world_size) for t in args.global_tokens)
    shared_weights = None
    if args.provider.startswith("mega"):
        layer = make_mega_layer(
            rank,
            world_size,
            max_tokens_per_rank,
        )
        if args.provider == "mega-fused":
            shared_weights = make_shared_weights()
    else:
        layer = make_trtllm_layer(rank, world_size, max_tokens_per_rank)

    header = (
        "provider,global_tokens,padded_global_tokens,tokens_per_rank,"
        "invocation_gpu_latency_ms_p10,invocation_gpu_latency_ms_p50,"
        "invocation_gpu_latency_ms_p90,preparation_gpu_latency_ms_p10,"
        "preparation_gpu_latency_ms_p50,preparation_gpu_latency_ms_p90,"
        "persistent_kernel_gpu_latency_ms_p10,"
        "persistent_kernel_gpu_latency_ms_p50,"
        "persistent_kernel_gpu_latency_ms_p90,"
        "wall_latency_ms_p50,samples"
    )
    import flashinfer

    metadata = {
        "world_size": world_size,
        "device_name": torch.cuda.get_device_name(local_rank),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
    }
    if rank == 0:
        print(f"# metadata={json.dumps(metadata, sort_keys=True)}", flush=True)
        print(header, flush=True)
    for global_tokens in args.global_tokens:
        tensors, active_tokens, tokens_per_rank = make_case(
            rank, world_size, global_tokens, shared_weights
        )
        result = run_case(
            layer,
            tensors,
            args.warmup,
            args.repeat,
            args.profile_cuda_range,
            args.cuda_graph,
            args.skip_component_timing,
        )
        result.update(
            **metadata,
            provider=args.provider,
            global_tokens=global_tokens,
            padded_global_tokens=tokens_per_rank * world_size,
            tokens_per_rank=tokens_per_rank,
            active_tokens_on_rank=active_tokens,
        )
        if rank == 0:
            preparation_values = (
                result.get("preparation_gpu_latency_ms_p10"),
                result.get("preparation_gpu_latency_ms_p50"),
                result.get("preparation_gpu_latency_ms_p90"),
            )
            preparation_csv = ",".join(
                "" if value is None else f"{value:.6f}" for value in preparation_values
            )
            kernel_values = (
                result.get("persistent_kernel_gpu_latency_ms_p10"),
                result.get("persistent_kernel_gpu_latency_ms_p50"),
                result.get("persistent_kernel_gpu_latency_ms_p90"),
            )
            kernel_csv = ",".join(
                "" if value is None else f"{value:.6f}" for value in kernel_values
            )
            print(
                f"{args.provider},{global_tokens},{tokens_per_rank * world_size},"
                f"{tokens_per_rank},"
                f"{result['invocation_gpu_latency_ms_p10']:.6f},"
                f"{result['invocation_gpu_latency_ms_p50']:.6f},"
                f"{result['invocation_gpu_latency_ms_p90']:.6f},"
                f"{preparation_csv},"
                f"{kernel_csv},"
                f"{result['wall_latency_ms_p50']:.6f},{result['samples']}",
                flush=True,
            )
            if args.output_jsonl:
                with open(args.output_jsonl, "a", encoding="utf-8") as output_file:
                    output_file.write(json.dumps(result, sort_keys=True) + "\n")

    layer.destroy()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
