"""Multi-rank smoke + correctness tests for MoEEpMegaLayer (sm100_nvfp4_nvfp4_bf16_cutedsl).

Launched via torchrun:
    torchrun --nproc_per_node=4 -m pytest tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py -v -m "gpu_4 and arch_blackwell"

Requires Blackwell (sm_100+), >=4 GPUs, and CuTeDSL runtime deps
(``nvidia-cutlass-dsl[cu13]``, ``nvshmem4py-cu13``).  Kernels ship in-tree under
``flashinfer.moe_ep.kernel_src.cutedsl_megamoe``.

Runtime bootstrap (``torch.distributed`` + NVSHMEM) is handled by
:class:`flashinfer.moe_ep.MoEEpMegaLayer` via :func:`bootstrap_moe_ep_runtime`.

Weights: the CuTeDSL kernel consumes NVFP4 expert weights in kernel-ready
(swizzled scale-factor) layout. These tests pass canonical bf16
:class:`~flashinfer.moe_ep.MoEWeightPack`; the layer quantizes them at init via
``preprocess_weights=True`` (see ``preprocess_mega_weights``). To supply
pre-quantized NVFP4 weights instead, pass ``w13``/``w2`` plus ``w13_scale``/``w2_scale``.

Torch-oracle anchor: parity alone cannot catch a kernel that is wrong but
self-consistent at ``world_size > 1`` (peer-pull addressing, expert→rank
ownership, cross-rank combine), because both sides run the same CUDA kernel.
``test_moe_ep_nvfp4_cutedsl_mega_multirank_torch_oracle`` closes that gap with
the sm90_fp8_fp8_bf16_pull_cutedsl twin's methodology: each rank all-gathers the actual plain
quantized weight legs and checks its real-EP kernel output slice against the
single-GPU pure-torch oracle run on its staged tokens + the global expert set.
"""

from __future__ import annotations

import os

import pytest

# This test verifies the mega path only through the cutedsl_megamoe shim public
# API (``flashinfer.moe_ep.kernel_src.cutedsl_megamoe``); it never imports the
# src/ kernel packages directly, so a new src/ drop can't silently break it.
pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def _require_cuda():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")


def _launcher_ranks() -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    return rank, world_size


def _make_inputs(
    rank: int,
    *,
    num_tokens: int,
    hidden: int,
    num_experts: int,
    topk: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    hidden_states = torch.randn(
        num_tokens, hidden, dtype=torch.bfloat16, device="cuda", generator=g
    )
    scores = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device="cuda", generator=g
    )
    topk_weights, topk_ids = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    return (
        hidden_states,
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int64),
    )


def _make_epilogue_params(rank: int, num_local_experts: int):
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        make_dummy_epilogue_params,
    )

    g = torch.Generator(device="cuda").manual_seed(19 + rank)
    return make_dummy_epilogue_params(num_local_experts, generator=g)


def _make_bf16_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    w13 = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    w2 = torch.randn(
        num_local_experts,
        hidden,
        intermediate,
        dtype=torch.bfloat16,
        device="cuda",
        generator=g,
    )
    return w13, w2


def _make_packed_weights(
    rank: int,
    *,
    num_local_experts: int,
    hidden: int,
    intermediate: int,
):
    import torch

    g = torch.Generator(device="cuda").manual_seed(31 + rank)
    w13 = torch.randint(
        0,
        256,
        (num_local_experts, 2 * intermediate, hidden // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w2 = torch.randint(
        0,
        256,
        (num_local_experts, hidden, intermediate // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    w13_scale = torch.randn(
        num_local_experts,
        2 * intermediate,
        hidden // 16,
        dtype=torch.float32,
        device="cuda",
        generator=g,
    ).to(torch.float8_e4m3fn)
    w2_scale = torch.randn(
        num_local_experts,
        hidden,
        intermediate // 16,
        dtype=torch.float32,
        device="cuda",
        generator=g,
    ).to(torch.float8_e4m3fn)
    return w13, w2, w13_scale, w2_scale


def _mega_problem(
    rank: int, world_size: int, *, num_tokens: int = 64, max_tokens: int = 64
):
    hidden = int(os.environ.get("MEGA_TEST_HIDDEN", "2048"))
    intermediate = int(os.environ.get("MEGA_TEST_INTERMEDIATE", "1024"))
    num_experts = int(os.environ.get("MEGA_TEST_NUM_EXPERTS", "8"))
    topk = int(os.environ.get("MEGA_TEST_TOPK", "4"))
    fast_math = True
    activation = os.environ.get("MEGA_TEST_ACTIVATION", "swiglu")
    gate_up_clamp = None if activation == "situ" else 10.0
    situ_beta = 7.0 if activation == "situ" else None
    situ_linear_beta = 1.0 if activation == "situ" else None

    assert hidden % 128 == 0
    assert intermediate % 128 == 0
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    hidden_states, topk_weights, topk_ids = _make_inputs(
        rank,
        num_tokens=num_tokens,
        hidden=hidden,
        num_experts=num_experts,
        topk=topk,
    )
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    fc1_alpha, fc2_alpha, fc1_norm_const = _make_epilogue_params(
        rank, num_local_experts
    )
    return dict(
        hidden=hidden,
        intermediate=intermediate,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        fast_math=fast_math,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=w13,
        w2=w2,
        fc1_alpha=fc1_alpha,
        fc2_alpha=fc2_alpha,
        fc1_norm_const=fc1_norm_const,
    )


def _reference_nvfp4_mega_moe_staged(
    problem: dict, *, destroy_buffer: bool = True, combine_dtype: str = "bf16"
):
    """Reference with bf16 activations staged inside the symm buffer."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        activation=problem["activation"],
        situ_beta=problem["situ_beta"],
        situ_linear_beta=problem["situ_linear_beta"],
        combine_dtype=combine_dtype,
        fc1_alpha=problem["fc1_alpha"],
        fc2_alpha=problem["fc2_alpha"],
        fc1_norm_const=problem["fc1_norm_const"],
    )
    num_tokens = problem["num_tokens"]
    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        symm_buffer.x[:num_tokens],
        symm_buffer.x_sf[:num_tokens],
        symm_buffer.topk_idx[:num_tokens],
        symm_buffer.topk_weights[:num_tokens],
    )

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    nvfp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _reference_nvfp4_mega_moe_prestaged(
    problem: dict, x_nvfp4, x_sf, *, destroy_buffer: bool = True
):
    """Reference with caller-supplied NVFP4 activations + fp8 block scales."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    symm_buffer = get_symm_buffer_for_mega_moe(
        problem["num_experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        2 * problem["intermediate"],
        rank,
        world_size,
        gate_up_clamp=problem["gate_up_clamp"],
        fc1_alpha=problem["fc1_alpha"],
        fc2_alpha=problem["fc2_alpha"],
        fc1_norm_const=problem["fc1_norm_const"],
    )
    num_tokens = problem["num_tokens"]
    symm_buffer.x[:num_tokens].copy_(x_nvfp4)
    symm_buffer.x_sf[:num_tokens].copy_(x_sf)
    symm_buffer.topk_idx[:num_tokens].copy_(problem["topk_ids"])
    symm_buffer.topk_weights[:num_tokens].copy_(problem["topk_weights"])

    pack = MoEWeightPack(w13=problem["w13"], w2=problem["w2"])
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        pack,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    y = torch.empty(num_tokens, problem["hidden"], dtype=torch.bfloat16, device="cuda")
    nvfp4_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        symm_buffer,
        num_tokens=num_tokens,
        gate_up_clamp=problem["gate_up_clamp"],
        fast_math=problem["fast_math"],
    )
    torch.cuda.synchronize()
    if destroy_buffer:
        symm_buffer.destroy()
    return y


def _assert_ikr_close(y, y_ref, *, topk):
    """Scale-aware compare for the in-flight (REDG) top-k reduce.

    The ikr path accumulates the K per-topk bf16 terms in nondeterministic
    order; the explicit-reduce reference accumulates the same terms in fp32.
    Where large terms nearly cancel, the achievable agreement is bounded by
    the bf16 round-off of the largest TERM, not of the final value, so a flat
    atol/rtol band misfires (this is why the kernel repo validates ikr with a
    K!-ordering bitwise check).  Bound the error per row by the row magnitude
    scale instead: K terms x bf16 eps (2^-8) x safety 8.  Measured need at
    this geometry: <= 0.67x the unscaled band; a missing per-launch output
    zero (the 2x-accumulation bug) overshoots by ~64x, so the guard stays
    sharp.
    """
    import torch

    a = y.float()
    b = y_ref.float()
    diff = (a - b).abs()
    row_scale = torch.maximum(a.abs(), b.abs()).amax(dim=1, keepdim=True)
    tol = 5e-2 + (topk * 2.0**-8 * 8.0) * row_scale
    worst = (diff - tol).max().item()
    assert worst <= 0.0, (
        f"ikr output outside the bf16 K-term accumulation band "
        f"(worst overshoot {worst:.4f}, max diff {diff.max().item():.4f})"
    )


def _megakernel_config(problem: dict, *, epilogue_via_config: bool, **config_extra):
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

    kwargs = dict(
        intermediate_size=problem["intermediate"],
        top_k=problem["topk"],
        gate_up_clamp=problem["gate_up_clamp"],
        activation=problem["activation"],
        situ_beta=problem["situ_beta"],
        situ_linear_beta=problem["situ_linear_beta"],
        fast_math=problem["fast_math"],
    )
    if epilogue_via_config:
        kwargs.update(
            fc1_alpha=problem["fc1_alpha"],
            fc2_alpha=problem["fc2_alpha"],
            fc1_norm_const=problem["fc1_norm_const"],
        )
    kwargs.update(config_extra)
    return Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(**kwargs)


def _run_mega_layer(
    rank,
    world_size,
    *,
    quantize_input: bool,
    num_tokens: int = 64,
    max_tokens: int = 64,
    in_kernel_fc2_reduce: bool = False,
    combine_dtype: str = "bf16",
    check_output_view: bool = False,
    shared_expert: bool = False,
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    problem = _mega_problem(
        rank, world_size, num_tokens=num_tokens, max_tokens=max_tokens
    )
    if shared_expert and os.environ.get("SHARED_TEST_DISABLE_ROUTED") == "1":
        problem["topk_ids"].fill_(-1)
    routed_tokens = int(
        os.environ.get("SHARED_TEST_ROUTED_TOKENS", problem["num_tokens"])
    )
    if routed_tokens < problem["num_tokens"]:
        problem["topk_ids"][routed_tokens:].fill_(-1)
    shared_intermediate = int(
        os.environ.get("SHARED_TEST_INTERMEDIATE", problem["intermediate"])
    )
    shared_hidden = int(
        os.environ.get("SHARED_TEST_HIDDEN", problem["hidden"])
    )
    shared_capacity = int(
        os.environ.get("SHARED_TEST_CAPACITY", problem["num_tokens"])
    )
    config_extra = dict(
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_dtype=combine_dtype,
    )
    if os.environ.get("SHARED_TEST_PRODUCTION_KNOBS") == "1":
        cluster_m = int(os.environ.get("SHARED_TEST_CLUSTER_M", "2"))
        config_extra["knobs"] = {
            "cluster_shape_mnk": (cluster_m, 1, 1),
            "group_hint": 512,
            "max_active_clusters": 120 // cluster_m,
            "epi_flag_batch": (2, 4),
            "load_balance_mode": "atomic_counter",
            "mma_tiler_mnk": (128 * cluster_m, 128, 256),
            "flag_batch": 4,
            "token_back_mode": "standalone_warps",
        }
    if shared_expert:
        config_extra.update(
            shared_hidden_size=shared_hidden,
            shared_intermediate_size=shared_intermediate,
        )
    kernel = create_mega_kernel(
        _megakernel_config(problem, epilogue_via_config=quantize_input, **config_extra)
    )
    print(f"rank {rank}: bootstrapping MegaMoE runtime", flush=True)
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )

    print(f"rank {rank}: MegaMoE runtime ready", flush=True)
    try:
        shared_inputs = None
        shared_reference = None
        shared_buffer = None
        if shared_expert:
            from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                MegaMoESharedNvfp4Inputs,
                fused_quant_stage,
                get_symm_buffer_for_mega_moe,
            )
            from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
                preprocess_mega_weights,
            )

            shared_g = torch.Generator(device="cuda").manual_seed(101)
            shared_hidden_states = torch.randn(
                problem["num_tokens"],
                shared_hidden,
                dtype=torch.bfloat16,
                device="cuda",
                generator=shared_g,
            )
            shared_w13 = torch.randn(
                1,
                2 * shared_intermediate,
                shared_hidden,
                dtype=torch.bfloat16,
                device="cuda",
                generator=shared_g,
            )
            shared_w2 = torch.randn(
                1,
                shared_hidden,
                shared_intermediate,
                dtype=torch.bfloat16,
                device="cuda",
                generator=shared_g,
            )
            shared_fc1, shared_fc2 = preprocess_mega_weights(
                MoEWeightPack(w13=shared_w13, w2=shared_w2),
                intermediate_size=shared_intermediate,
                hidden_size=shared_hidden,
            )
            shared_buffer = get_symm_buffer_for_mega_moe(
                1,
                shared_capacity,
                1,
                shared_hidden,
                2 * shared_intermediate,
                0,
                1,
                activation=problem["activation"],
                situ_beta=problem["situ_beta"],
                situ_linear_beta=problem["situ_linear_beta"],
                local_only=True,
            )
            shared_topk_ids = torch.zeros(
                problem["num_tokens"], 1, dtype=torch.int64, device="cuda"
            )
            shared_topk_weights = torch.ones(
                problem["num_tokens"], 1, dtype=torch.float32, device="cuda"
            )
            sf_rows = ((shared_capacity + 127) // 128) * 128
            activation_sf = torch.zeros(
                sf_rows,
                ((shared_hidden // 16 + 3) // 4) * 4,
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            fused_quant_stage(
                shared_hidden_states,
                shared_topk_ids,
                shared_topk_weights,
                shared_buffer.x,
                activation_sf,
                shared_buffer.topk_idx,
                shared_buffer.topk_weights,
                quant_type="nvfp4",
                norm_const=1.0,
                sf_layout="blocked_128x4",
            )
            sf_cols = ((shared_intermediate // 16 + 3) // 4) * 4
            shared_inputs = MegaMoESharedNvfp4Inputs(
                activation=shared_buffer.x,
                activation_sf=activation_sf,
                fc1_weight=shared_fc1[0],
                fc1_weight_sf=shared_fc1[1],
                fc1_output=torch.empty(
                    shared_capacity,
                    shared_intermediate // 2,
                    dtype=torch.float4_e2m1fn_x2,
                    device="cuda",
                ),
                fc1_output_sf=torch.zeros(
                    sf_rows,
                    sf_cols,
                    dtype=torch.float8_e4m3fn,
                    device="cuda",
                ),
                fc2_weight=shared_fc2[0],
                fc2_weight_sf=shared_fc2[1],
                fc1_alpha=shared_buffer.fc1_alpha,
                fc2_alpha=shared_buffer.fc2_alpha,
                fc1_norm_const=shared_buffer.fc1_norm_const,
                fc1_done_counter=torch.zeros(
                    max(shared_capacity, 1),
                    dtype=torch.int32,
                    device="cuda",
                ),
                output_activation=shared_buffer.output_activation,
            )
            shared_inputs.output_activation.zero_()

        if quantize_input:
            t_hidden = problem["hidden_states"]
            t_scales = None
        else:
            from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                get_symm_buffer_for_mega_moe,
            )

            staging_buffer = get_symm_buffer_for_mega_moe(
                problem["num_experts"],
                problem["max_tokens"],
                problem["topk"],
                problem["hidden"],
                2 * problem["intermediate"],
                rank,
                world_size,
                gate_up_clamp=problem["gate_up_clamp"],
            )
            num_tokens = problem["num_tokens"]
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                staging_buffer.x[:num_tokens],
                staging_buffer.x_sf[:num_tokens],
                staging_buffer.topk_idx[:num_tokens],
                staging_buffer.topk_weights[:num_tokens],
            )
            t_hidden = staging_buffer.x[:num_tokens].clone()
            t_scales = staging_buffer.x_sf[:num_tokens].clone()
            staging_buffer.destroy()

        mega = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
            ),
            fleet_params=FleetParams(
                num_experts=problem["num_experts"],
                max_tokens_per_rank=problem["max_tokens"],
                token_hidden_size=problem["hidden"],
            ),
            weights=MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
            backend=MegaConfig(
                megakernel=_megakernel_config(
                    problem, epilogue_via_config=quantize_input, **config_extra
                ),
                quantize_input=quantize_input,
                preprocess_weights=True,
            ),
        )
        assert isinstance(mega, MoEEpMegaLayer)

        tensor_kwargs = {}
        if not quantize_input:
            tensor_kwargs = dict(
                fc1_alpha=problem["fc1_alpha"],
                fc2_alpha=problem["fc2_alpha"],
                fc1_norm_const=problem["fc1_norm_const"],
            )
        layer_shared_inputs = shared_inputs
        if os.environ.get("SHARED_TEST_ROUTED_ONLY_FRONTEND") == "1":
            layer_shared_inputs = None
        if os.environ.get("SHARED_TEST_ROUTE_THEN_FUSED") == "1":
            routed_t = MoEEpTensors(
                hidden_states=t_hidden,
                topk_ids=problem["topk_ids"],
                topk_weights=problem["topk_weights"],
                scales=t_scales,
                mega_shared_inputs=None,
                **tensor_kwargs,
            )
            mega.forward(routed_t)
        t = MoEEpTensors(
            hidden_states=t_hidden,
            topk_ids=problem["topk_ids"],
            topk_weights=problem["topk_weights"],
            scales=t_scales,
            mega_shared_inputs=layer_shared_inputs,
            **tensor_kwargs,
        )
        print(
            f"rank {rank}: launching MegaMoE forward "
            f"(shared_expert={shared_expert})",
            flush=True,
        )
        y_layer = mega.forward(t).clone()
        if os.environ.get("SHARED_TEST_ROUTE_THEN_FUSED") == "1":
            workspace = mega._workspace
            routed_mega = workspace._routed_frontend._mega
            fused_mega = workspace._frontend._mega
            assert routed_mega is not None
            assert fused_mega is not None
            assert (
                routed_mega.local_workspace.data_ptr()
                == fused_mega.local_workspace.data_ptr()
            )
            assert (
                routed_mega.shared_workspace.data_ptr()
                == fused_mega.shared_workspace.data_ptr()
            )
        print(f"rank {rank}: MegaMoE forward complete", flush=True)
        shared_actual = None
        if layer_shared_inputs is not None:
            shared_actual = shared_inputs.output_activation[
                : problem["num_tokens"]
            ].clone()
            print(f"rank {rank}: launching shared reference", flush=True)
            from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                get_symm_buffer_for_mega_moe,
                nvfp4_mega_launch_thunk,
            )

            reference_buffer = get_symm_buffer_for_mega_moe(
                1,
                shared_capacity,
                1,
                shared_hidden,
                2 * shared_intermediate,
                0,
                1,
                activation=problem["activation"],
                situ_beta=problem["situ_beta"],
                situ_linear_beta=problem["situ_linear_beta"],
                local_only=True,
            )
            stage_mega_moe_inputs(
                shared_hidden_states,
                shared_topk_weights,
                shared_topk_ids,
                reference_buffer.x[: problem["num_tokens"]],
                reference_buffer.x_sf[: problem["num_tokens"]],
                reference_buffer.topk_idx[: problem["num_tokens"]],
                reference_buffer.topk_weights[: problem["num_tokens"]],
            )
            reference_thunk = nvfp4_mega_launch_thunk(
                shared_fc1, shared_fc2, reference_buffer
            )
            reference_thunk()
            torch.cuda.synchronize()
            print(f"rank {rank}: shared reference complete", flush=True)
            shared_reference = reference_buffer.output_activation[
                : problem["num_tokens"]
            ].clone()
            reference_buffer.destroy()
        clear_tokens = min(
            ((max(problem["num_tokens"], 1) + 63) // 64) * 64,
            problem["max_tokens"],
        )
        untouched_tail = (
            in_kernel_fc2_reduce and clear_tokens < problem["max_tokens"]
        )
        if untouched_tail:
            mega._workspace.output_activation[clear_tokens:].fill_(7.0)
        # Repeated forward on the same session: with no per-launch host reset
        # (run() default reset_counters=False) the second launch relies on the
        # kernel's tail cleanup of its workspace counters/flags -- this is the
        # regression guard for that contract.
        y_layer2 = mega.forward(t)

        if check_output_view:
            assert mega.supports_output_view
            y_view = mega.forward(t, return_workspace_view=True)
            torch.cuda.synchronize()
            assert y_view.shape == (problem["num_tokens"], problem["hidden"])
            assert y_view.data_ptr() == mega._workspace.output_activation.data_ptr()
            y_view_copy = y_view.clone()
            y_view_repeat = mega.forward(t, return_workspace_view=True)
            torch.cuda.synchronize()
            assert torch.equal(y_view_copy, y_layer)
            assert torch.equal(y_view_repeat, y_layer)

        torch.cuda.synchronize()
        if shared_reference is not None:
            torch.testing.assert_close(
                shared_actual,
                shared_reference,
            )
        if untouched_tail:
            assert torch.all(
                mega._workspace.output_activation[clear_tokens:] == 7.0
            )
        dist.barrier()

        if quantize_input:
            y_ref = _reference_nvfp4_mega_moe_staged(
                problem, destroy_buffer=True, combine_dtype=combine_dtype
            )
        else:
            y_ref = _reference_nvfp4_mega_moe_prestaged(
                problem, t_hidden, t_scales, destroy_buffer=True
            )
        dist.barrier()

        assert y_layer.shape == (problem["num_tokens"], problem["hidden"])
        assert y_layer.dtype == torch.bfloat16
        assert torch.isfinite(y_layer).all()
        if in_kernel_fc2_reduce:
            # Tolerance verdict vs the explicit-reduce (plain-sum) reference;
            # see _assert_ikr_close.  The repeated forward doubles as the
            # regression guard for the per-launch output_activation.zero_()
            # (accumulate-from-zero contract): without it y_layer2 would be
            # ~2x the reference and fail loudly.
            _assert_ikr_close(y_layer, y_ref, topk=problem["topk"])
            _assert_ikr_close(y_layer2, y_ref, topk=problem["topk"])
        else:
            torch.testing.assert_close(y_layer, y_ref, atol=0.0, rtol=0.0)
            torch.testing.assert_close(y_layer2, y_ref, atol=0.0, rtol=0.0)

        if layer_shared_inputs is not None:
            routed_only_t = MoEEpTensors(
                hidden_states=t_hidden,
                topk_ids=problem["topk_ids"],
                topk_weights=problem["topk_weights"],
                scales=t_scales,
                **tensor_kwargs,
            )
            y_routed_only = mega.forward(routed_only_t)
            torch.cuda.synchronize()
            if in_kernel_fc2_reduce:
                _assert_ikr_close(
                    y_routed_only, y_ref, topk=problem["topk"]
                )
            else:
                torch.testing.assert_close(
                    y_routed_only, y_ref, atol=0.0, rtol=0.0
                )

        if combine_dtype != "bf16":
            # Numerics sanity vs the exact bf16 combine wire: the quantized
            # wire lossily encodes the per-topk fc2 outputs, so bound the
            # whole-tensor relative L2 error rather than per-element compare.
            # The strong plumbing check is the bit-exact compare above
            # (layer vs direct-shim reference on the same wire format).
            y_ref_bf16 = _reference_nvfp4_mega_moe_staged(
                problem, destroy_buffer=True, combine_dtype="bf16"
            )
            dist.barrier()
            rel_l2 = (
                (y_layer.float() - y_ref_bf16.float()).norm()
                / y_ref_bf16.float().norm().clamp_min(1e-6)
            ).item()
            band = 0.25 if combine_dtype == "nvfp4" else 0.10
            assert rel_l2 < band, (
                f"quantized combine ({combine_dtype}) rel-L2 {rel_l2:.4f} "
                f"vs bf16 combine exceeds {band}"
            )
        mega.destroy()
        if shared_buffer is not None:
            shared_buffer.destroy()
        return rank
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_matches_reference():
    """MoEEpMegaLayer (sm100_nvfp4_nvfp4_bf16_cutedsl) with on-the-fly bf16→NVFP4 staging.

    Per-expert ``fc1_alpha`` / ``fc2_alpha`` / ``fc1_norm_const`` are supplied
    via :class:`Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig` (workspace allocation).
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=True)
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (staged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_with_shared_expert():
    """Distributed shared-expert staging completes and matches its reference."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    num_tokens = int(os.environ.get("SHARED_TEST_TOKENS", "64"))
    max_tokens = int(os.environ.get("MEGA_TEST_MAX_TOKENS", str(num_tokens)))
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        num_tokens=num_tokens,
        max_tokens=max_tokens,
        shared_expert=True,
    )
    print(f"rank {rank}: shared expert matches the standalone NVFP4 reference")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_prestaged_inputs_matches_reference():
    """MoEEpMegaLayer (sm100_nvfp4_nvfp4_bf16_cutedsl) with pre-staged NVFP4 activations.

    Per-expert epilogue scalars are supplied via :class:`MoEEpTensors` and copied
    into the symm workspace when ``quantize_input=False``.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(rank, world_size, quantize_input=False)
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (prestaged inputs) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_large_tokens_matches_reference():
    """Large-token (>=2048) path: exercises the tuner's LARGE profile.

    With num_max_tokens >= 2048 the token-count heuristic selects the
    throughput tile (mma_tiler (256,256,256), token_back reuse_dispatch_warps).
    This confirms that profile compiles + runs and that the layer path stays
    bit-exact with the direct-kernel reference (both use the same profile).
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=True, num_tokens=2048, max_tokens=2048
    )
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (large tokens) matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("num_tokens", [64, 2048])
def test_moe_ep_nvfp4_cutedsl_mega_layer_output_view(num_tokens):
    """Public output-view API is bit-exact and reusable on every EP rank."""
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        num_tokens=num_tokens,
        max_tokens=num_tokens,
        check_output_view=True,
    )
    print(f"rank {rank}: output view matches copied output for {num_tokens} tokens")


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("num_tokens", [512, 1024])
def test_moe_ep_nvfp4_cutedsl_mega_layer_mid_tokens_matches_reference(num_tokens):
    """Mid-token paths: exercise the tuner's MID / MID-LARGE profiles.

    512 selects the mid tactic (mma_tiler (256,128,256), flag_batch 4,
    token_back reuse_dispatch_warps); 1024 selects the mid-large tactic
    (mma_tiler (256,256,256), flag_batch 4, token_back standalone_warps) --
    the 2026-07-14 autotune winners at those sizes.  Confirms each profile
    compiles + runs and the layer path stays bit-exact with the direct-kernel
    reference.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        num_tokens=num_tokens,
        max_tokens=num_tokens,
    )
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (mid tokens={num_tokens}) "
        "matches reference"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_in_kernel_fc2_reduce():
    """In-flight top-k combine (``in_kernel_fc2_reduce=True``).

    The REDG atomic-add collapses the combine as peer data arrives, so the
    output matches the plain-sum reference only up to accumulation order
    (tolerance verdict, not bit-exact).  The second forward inside
    ``_run_mega_layer`` guards the accumulate-from-zero contract: the frontend
    must zero ``output_activation`` before every launch or the repeat would
    come back ~2x.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank,
        world_size,
        quantize_input=True,
        num_tokens=20,
        max_tokens=256,
        in_kernel_fc2_reduce=True,
    )
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (in_kernel_fc2_reduce) "
        "matches reference within tolerance"
    )


def _run_mega_layer_zero_token_ikr_regression(
    rank,
    world_size,
    *,
    num_iters: int = 60,
):
    """Interleave num_tokens=0 and real forward() calls, in_kernel_fc2_reduce=True,
    no barrier between iterations, at an independent per-rank schedule.

    NVFP4 mirror of the MXFP8 regression guard in
    test_moe_ep_mxfp8_cutedsl_mega_multirank.py -- same bug, same fix, same
    kernel architecture, different dtype.  ``nvfp4_mega_moe()`` has the
    identical num_tokens==0 shortcut (spelled via the ``fc2_reduces_topk``
    property, which is just ``in_kernel_fc2_reduce`` under a different name):

        if n == 0 and symm_buffer._frontend.config.fc2_reduces_topk:
            return symm_buffer.output_activation[:0] if y is None else None

    that used to return WITHOUT ever calling frontend.run() (i.e. without
    launching the kernel at all). Sm100MegaMoEKernel (NVFP4) shares the same
    persistent-megakernel scheduler infra as MXFP8
    (MoEFusedFc12SchedulerParams.get_grid_shape, sized from hardware
    occupancy, never from num_tokens), so the same fix -- fall through to the
    same full-buffer frontend.run() call every nonzero num_tokens already
    takes -- applies unchanged. See
    kernel_src/cutedsl_megamoe/shim/nvfp4.py::nvfp4_mega_moe.

    Shapes/scale intentionally match the real repro (hidden=2048,
    intermediate=768, num_experts=128, top_k=8, max_tokens_per_rank=16384),
    not this file's usual small test defaults, with genuinely independent
    (per-rank-seeded ``random.Random``, not a fixed formula) per-rank timing --
    see the MXFP8 twin's docstring for why both matter (the same test at
    smaller scale plus a deterministic schedule passes vacuously even without
    the fix).

    CAVEAT: see the MXFP8 twin's docstring for the full story on why this is
    hard to make reliably fail pre-fix under pytest specifically (it does
    reliably fail pre-fix as a plain torchrun-launched script -- see
    tests/moe_ep/../repro_ikr_zero_token_idle.py, the authoritative
    regression artifact for this bug); this test is kept as a documented,
    passing correctness check of the exact scenario under the project's
    normal test harness, with a deliberate (if unproven-sufficient) timing
    nudge to improve its odds of catching a real regression.
    """
    import random
    import time

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEEpTensors,
        MoEWeightPack,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)

    hidden = 2048
    intermediate = 768
    num_experts = 128
    topk = 8
    max_tokens = 16384
    real_tokens = 4
    assert num_experts % world_size == 0
    num_local_experts = num_experts // world_size

    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=hidden,
        intermediate=intermediate,
    )
    warmup_hidden_states, warmup_topk_weights, warmup_topk_ids = _make_inputs(
        rank, num_tokens=real_tokens, hidden=hidden, num_experts=num_experts, topk=topk
    )
    fc1_alpha, fc2_alpha, fc1_norm_const = _make_epilogue_params(
        rank, num_local_experts
    )
    megakernel_config = _megakernel_config(
        dict(
            intermediate=intermediate,
            topk=topk,
            gate_up_clamp=10.0,
            activation="swiglu",
            situ_beta=None,
            situ_linear_beta=None,
            fast_math=True,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
        ),
        epilogue_via_config=True,
        in_kernel_fc2_reduce=True,
    )

    mega = MoEEpMegaLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=hidden,
        ),
        weights=MoEWeightPack(w13=w13, w2=w2),
        backend=MegaConfig(megakernel=megakernel_config, preprocess_weights=True),
    )
    try:
        # Matched-count collective warmup -- every rank calls forward() with
        # real tokens once, together, before the independent-cadence loop.
        mega.forward(
            MoEEpTensors(
                hidden_states=warmup_hidden_states,
                topk_ids=warmup_topk_ids,
                topk_weights=warmup_topk_weights,
            )
        )
        torch.cuda.synchronize()
        dist.barrier()

        # Independently-seeded per-rank RNG, no barrier between iterations:
        # rank 0 always real (mirrors an always-busy rank); other ranks
        # independently coin-flip zero/real every iteration, so each rank's
        # actual wall-clock cadence diverges from its peers' in a way a fixed
        # formula doesn't produce. Seeded for CI reproducibility.
        rnd = random.Random(4242 + rank)
        for it in range(num_iters):
            n = real_tokens if rank == 0 else (0 if rnd.random() < 0.5 else real_tokens)
            g = torch.Generator(device="cuda").manual_seed(1000 * it + rank)
            hidden_states = torch.randn(
                n, hidden, dtype=torch.bfloat16, device="cuda", generator=g
            )
            scores = torch.randn(
                n, num_experts, dtype=torch.float32, device="cuda", generator=g
            )
            topk_weights, topk_ids = torch.topk(
                scores, topk, dim=-1, largest=True, sorted=False
            )
            t = MoEEpTensors(
                hidden_states=hidden_states,
                topk_ids=topk_ids.to(torch.int64),
                topk_weights=topk_weights.to(torch.float32),
            )
            if n > 0:
                # Nudge real per-rank wall-clock divergence: pre-fix, a
                # zero-token round skips the kernel launch entirely and is
                # near-instant, while a real round pays actual GPU cost --
                # under plain torchrun that gap alone is enough to desync
                # ranks within tens of rounds, but empirically not reliably
                # under pytest (unconfirmed why; see the CAVEAT above). This
                # doesn't guarantee detection here, just improves the odds.
                time.sleep(0.003)
            y = mega.forward(t)
            torch.cuda.synchronize()
            assert y.shape == (n, hidden)
            assert y.dtype == torch.bfloat16
            assert torch.isfinite(y).all()

        dist.barrier()
        return rank
    finally:
        mega.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_layer_in_kernel_fc2_reduce_zero_token_regression():
    """Zero-token / in_kernel_fc2_reduce livelock regression guard (NVFP4).

    See ``_run_mega_layer_zero_token_ikr_regression`` for the full bug
    writeup. Before the fix, this reliably livelocks within tens of
    iterations; after the fix, all ``num_iters`` complete cleanly regardless
    of each rank's independent zero/nonzero token schedule.
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer_zero_token_ikr_regression(rank, world_size)
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer survives "
        "interleaved zero-token/real in_kernel_fc2_reduce forward calls"
    )


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize("combine_dtype", ["nvfp4", "mxfp8"])
def test_moe_ep_nvfp4_cutedsl_mega_layer_quantized_combine(combine_dtype):
    """Quantized cross-rank combine wire (``combine_dtype`` != bf16).

    ``nvfp4`` (16e2m1xbf16) shrinks the NVLink combine traffic 4x, ``mxfp8``
    (32e4m3xe8m0) 2x.  The layer output must be bit-exact with the direct-shim
    reference on the same wire format (deterministic explicit-reduce path) and
    within a loose relative-L2 band of the exact bf16-combine reference
    (wire quantization is a numerics tradeoff).
    """
    _require_cuda()
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_layer(
        rank, world_size, quantize_input=True, combine_dtype=combine_dtype
    )
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega layer (combine_dtype={combine_dtype}) "
        "matches reference"
    )


def _all_gather_stack(t):
    """all_gather a per-rank tensor and stack it on a new leading rank dim.

    FP4/FP8 payloads and scale planes travel as uint8 bytes (NCCL supports
    none of the sub-byte/8-bit float dtypes) and are reinterpreted after the
    stack.
    """
    import torch
    import torch.distributed as dist

    world_size = dist.get_world_size()
    tc = t.contiguous()
    byte_wire = tc.element_size() == 1 and tc.dtype != torch.uint8
    wire = tc.view(torch.uint8) if byte_wire else tc
    gathered = [torch.empty_like(wire) for _ in range(world_size)]
    dist.all_gather(gathered, wire)
    stacked = torch.stack(gathered)
    return stacked.view(tc.dtype) if byte_wire else stacked


def _identity_epilogue_params(num_local_experts: int):
    """Identity per-expert epilogue scalars (fc1_alpha/fc2_alpha/norm_const=1).

    The pure-torch oracle models the alpha_swiglu_clamp epilogue only at its
    identity point (like ``bench_moe_ep_mega``'s accuracy column), so the
    oracle test pins the scalars to 1.0 instead of ``make_dummy_epilogue_params``.
    """
    import torch

    ones = torch.ones(num_local_experts, dtype=torch.float32, device="cuda")
    return ones, ones.clone(), ones.clone()


def _run_mega_torch_oracle(
    rank,
    world_size,
    *,
    in_kernel_fc2_reduce: bool = False,
    combine_dtype: str = "bf16",
    activation: str = "swiglu",
):
    """Real-EP kernel launch vs a pure-torch oracle on the GLOBAL expert set.

    Parity tests alone cannot catch a kernel that is wrong but self-consistent
    at ``world_size > 1`` (peer-pull addressing, expert→rank ownership,
    peer-token dequant, cross-rank combine), because both sides run the same
    CUDA kernel; this closes that gap (sm90_fp8_fp8_bf16_pull_cutedsl twin methodology).

    Every rank stages its own bf16 shard, runs the fused kernel with real
    cross-rank NVSHMEM traffic, then all-gathers the ACTUAL plain (pre-swizzle)
    fp4 weight legs of all ranks (no reliance on cross-rank RNG determinism)
    and feeds its own staged activations + routing plus the gathered global
    expert set to the single-GPU pure-torch oracle
    (``_torch_nvfp4_mega_reference``): y[t] = Σ_k w_k · expert_{id_k}(x_t) is
    per-token math, so the local token shard against global weights is the
    full ground truth for this rank's output slice.

    Variants: ``in_kernel_fc2_reduce`` runs the REDG in-flight combine (torch
    reference unchanged; the compare widens by the bf16 K-term accumulation
    band, see ``_assert_ikr_close``).  A quantized ``combine_dtype`` runs the
    low-precision combine wire; the oracle models the wire exactly via
    ``combine_roundtrip_to_fp32`` on each per-topk fc2 term (mirroring the
    device combine encoder + topk_reduce), so the tolerance band stays the
    single-GPU one.  The two knobs are mutually exclusive (shim contract).
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEWeightPack,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        nvfp4_mega_moe,
    )

    from .test_nvfp4_cutedsl_kernel_vs_reference import (
        _plain_nvfp4_from_bf16,
        _torch_nvfp4_mega_reference,
    )

    bootstrap = BootstrapConfig(world_size=world_size, rank=rank)
    ensure_moe_ep_cuda_device(bootstrap)
    problem = _mega_problem(rank, world_size)
    if activation == "situ":
        problem.update(
            gate_up_clamp=None,
            activation="situ",
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
    num_local = problem["num_experts"] // world_size
    # Identity epilogue scalars: the torch oracle has no alpha/norm-const legs.
    (
        problem["fc1_alpha"],
        problem["fc2_alpha"],
        problem["fc1_norm_const"],
    ) = _identity_epilogue_params(num_local)
    # Guarantee cross-rank traffic by construction: token 0 routes one expert
    # per EP rank (contiguous block ownership: rank r owns [r*L, (r+1)*L)).
    forced = (
        torch.arange(min(problem["topk"], world_size), device="cuda", dtype=torch.int64)
        * num_local
    )
    problem["topk_ids"][0, : forced.numel()] = forced

    kernel = create_mega_kernel(
        _megakernel_config(
            problem,
            epilogue_via_config=True,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            combine_dtype=combine_dtype,
        )
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        kernel.runtime_requirements(bootstrap),
    )
    try:
        n = problem["num_tokens"]

        symm_buffer = get_symm_buffer_for_mega_moe(
            problem["num_experts"],
            problem["max_tokens"],
            problem["topk"],
            problem["hidden"],
            2 * problem["intermediate"],
            rank,
            world_size,
            gate_up_clamp=problem["gate_up_clamp"],
            activation=problem["activation"],
            situ_beta=problem["situ_beta"],
            situ_linear_beta=problem["situ_linear_beta"],
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            combine_dtype=combine_dtype,
            fc1_alpha=problem["fc1_alpha"],
            fc2_alpha=problem["fc2_alpha"],
            fc1_norm_const=problem["fc1_norm_const"],
        )
        try:
            stage_mega_moe_inputs(
                problem["hidden_states"],
                problem["topk_weights"],
                problem["topk_ids"],
                symm_buffer.x[:n],
                symm_buffer.x_sf[:n],
                symm_buffer.topk_idx[:n],
                symm_buffer.topk_weights[:n],
            )
            # Snapshot exactly what the kernel consumes (this rank's shard).
            x_local = symm_buffer.x[:n].clone()
            x_sf_local = symm_buffer.x_sf[:n].clone()
            idx_local = symm_buffer.topk_idx[:n].clone()
            w_local = symm_buffer.topk_weights[:n].clone()

            transformed_l1, transformed_l2 = preprocess_mega_weights(
                MoEWeightPack(w13=problem["w13"], w2=problem["w2"]),
                intermediate_size=problem["intermediate"],
                hidden_size=problem["hidden"],
                gate_up_clamp=problem["gate_up_clamp"],
            )

            y_kernel = torch.empty(
                n, problem["hidden"], dtype=torch.bfloat16, device="cuda"
            )
            nvfp4_mega_moe(
                y_kernel,
                transformed_l1,
                transformed_l2,
                symm_buffer,
                num_tokens=n,
                gate_up_clamp=problem["gate_up_clamp"],
                fast_math=problem["fast_math"],
            )
            torch.cuda.synchronize()
            dist.barrier()

            # Reassemble the global expert set from each rank's ACTUAL plain
            # (pre-swizzle) quantized weight legs; (R, E_local, ...) → (E, ...)
            # rank-major matches the global expert id convention.
            fc1_plain, fc1_sf, fc2_plain, fc2_sf = _plain_nvfp4_from_bf16(problem)
            fc1_w_g = _all_gather_stack(fc1_plain).flatten(0, 1)
            fc1_sf_g = _all_gather_stack(fc1_sf).flatten(0, 1)
            fc2_w_g = _all_gather_stack(fc2_plain).flatten(0, 1)
            fc2_sf_g = _all_gather_stack(fc2_sf).flatten(0, 1)

            # Model the quantized combine wire in the reference: each per-topk fc2
            # term goes bf16 → wire quantize → dequantize exactly as the device
            # combine encoder + topk_reduce do.
            term_transform = None
            if combine_dtype != "bf16":
                from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
                    CombineFormat,
                    combine_roundtrip_to_fp32,
                )

                wire = CombineFormat.parse(
                    {"nvfp4": "16e2m1xbf16", "mxfp8": "32e4m3xe8m0"}[combine_dtype]
                )

                def term_transform(t):
                    return combine_roundtrip_to_fp32(t.to(torch.bfloat16).float(), wire)

            y_ref = _torch_nvfp4_mega_reference(
                act_packed=x_local,
                act_sf=x_sf_local,
                topk_idx=idx_local,
                topk_weights=w_local,
                fc1_weight=fc1_w_g,
                fc1_sf=fc1_sf_g,
                fc2_weight=fc2_w_g,
                fc2_sf=fc2_sf_g,
                hidden=problem["hidden"],
                intermediate=problem["intermediate"],
                gate_up_clamp=problem["gate_up_clamp"],
                activation=problem["activation"],
                situ_beta=problem["situ_beta"],
                situ_linear_beta=problem["situ_linear_beta"],
                term_transform=term_transform,
            )

            assert torch.isfinite(y_kernel).all()
            yk = y_kernel.to(torch.float32)
            yr = y_ref.to(torch.float32)
            rel_l2 = (yk - yr).norm() / yr.norm().clamp_min(1e-6)
            print(
                f"[nvfp4 multirank oracle rank {rank} ikr={in_kernel_fc2_reduce} "
                f"combine={combine_dtype}] rel_l2={rel_l2.item():.4g} "
                f"max|d|={(yk - yr).abs().max().item():.4g} "
                f"amax(ref)={yr.abs().max().item():.4g}"
            )
            # Single-GPU oracle tolerances: the residual is NVFP4 RTNE flips at
            # fc1-out + accumulation-order noise; atol scales with the output
            # range (random unscaled weights put |y|~1e4 here).
            atol = 2e-3 * yr.abs().max().item()
            if in_kernel_fc2_reduce:
                # The REDG reduce accumulates the K per-topk bf16 terms in
                # nondeterministic order vs the reference's fp32 sum; widen the
                # quant band by the bf16 K-term accumulation band per row (same
                # bound as _assert_ikr_close).
                diff = (yk - yr).abs()
                row_scale = torch.maximum(yk.abs(), yr.abs()).amax(dim=1, keepdim=True)
                tol = (
                    atol
                    + 0.05 * yr.abs()
                    + (problem["topk"] * 2.0**-8 * 8.0) * row_scale
                )
                worst = (diff - tol).max().item()
                assert worst <= 0.0, (
                    f"ikr oracle output outside the widened band "
                    f"(worst overshoot {worst:.4f}, max diff {diff.max().item():.4f})"
                )
                assert rel_l2.item() < 0.03
            else:
                torch.testing.assert_close(yk, yr, atol=atol, rtol=0.05)
                assert rel_l2.item() < 0.02
            return rank
        finally:
            # A failing rank must still free its symmetric-heap slice;
            # leaking it turns a clean failure into a multi-rank hang.
            symm_buffer.destroy()
    finally:
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "in_kernel_fc2_reduce,combine_dtype,activation",
    [
        (False, "bf16", "swiglu"),
        (True, "bf16", "swiglu"),
        (True, "bf16", "situ"),
        (False, "nvfp4", "swiglu"),
        (False, "mxfp8", "swiglu"),
    ],
)
def test_moe_ep_nvfp4_cutedsl_mega_multirank_torch_oracle(
    in_kernel_fc2_reduce, combine_dtype, activation
):
    """Real cross-rank EP kernel vs pure-torch global math (see helper doc)."""
    _require_cuda()
    pytest.importorskip("triton")
    rank, world_size = _launcher_ranks()
    if world_size < 4:
        pytest.skip("needs >=4 ranks")
    rank = _run_mega_torch_oracle(
        rank,
        world_size,
        in_kernel_fc2_reduce=in_kernel_fc2_reduce,
        combine_dtype=combine_dtype,
        activation=activation,
    )
    print(
        f"rank {rank}: sm100_nvfp4_nvfp4_bf16_cutedsl mega kernel (ikr={in_kernel_fc2_reduce}, "
        f"combine={combine_dtype}, activation={activation}) matches the "
        "multi-rank torch oracle"
    )


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
def test_moe_ep_nvfp4_cutedsl_mega_situ_multirank_torch_oracle():
    """SiTU with real cross-rank dispatch/combine vs the global torch oracle."""
    _require_cuda()
    pytest.importorskip("triton")
    rank, world_size = _launcher_ranks()
    if world_size < 2:
        pytest.skip("needs >=2 ranks")
    rank = _run_mega_torch_oracle(rank, world_size, activation="situ")
    print(f"rank {rank}: NVFP4 MegaMoE SiTU matches the multi-rank torch oracle")


@pytest.mark.arch_blackwell
def test_nvfp4_cutedsl_preprocess_accepts_sglang_packed_weights():
    _require_cuda()

    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_local_experts = problem["num_experts"] // world_size
    w13, w2, w13_scale, w2_scale = _make_packed_weights(
        rank,
        num_local_experts=num_local_experts,
        hidden=problem["hidden"],
        intermediate=problem["intermediate"],
    )

    transformed_l1, transformed_l2 = preprocess_mega_weights(
        MoEWeightPack(w13=w13, w2=w2, w13_scale=w13_scale, w2_scale=w2_scale),
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
        gate_up_clamp=problem["gate_up_clamp"],
    )

    fc1_weight, fc1_sf = transformed_l1
    fc2_weight, fc2_sf = transformed_l2
    assert fc1_weight.shape == (
        num_local_experts,
        problem["hidden"] // 2,
        2 * problem["intermediate"],
    )
    assert fc2_weight.shape == (
        num_local_experts,
        problem["intermediate"] // 2,
        problem["hidden"],
    )
    assert fc1_weight.dtype == torch.float4_e2m1fn_x2
    assert fc2_weight.dtype == torch.float4_e2m1fn_x2
    assert fc1_sf.shape[0] == num_local_experts
    assert fc2_sf.shape[0] == num_local_experts


@pytest.mark.arch_blackwell
def test_nvfp4_cutedsl_staging_uses_input_norm_const():
    _require_cuda()

    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    rank, world_size = _launcher_ranks()
    problem = _mega_problem(rank, world_size)
    num_tokens = problem["num_tokens"]
    hidden = problem["hidden"]
    topk = problem["topk"]
    sf_cols = hidden // 16

    x_nvfp4_a = torch.empty(
        num_tokens, hidden // 2, dtype=torch.float4_e2m1fn_x2, device="cuda"
    )
    x_nvfp4_b = torch.empty_like(x_nvfp4_a)
    x_sf_a = torch.empty(num_tokens, sf_cols, dtype=torch.float8_e4m3fn, device="cuda")
    x_sf_b = torch.empty_like(x_sf_a)
    topk_idx_a = torch.empty(num_tokens, topk, dtype=torch.int64, device="cuda")
    topk_idx_b = torch.empty_like(topk_idx_a)
    topk_weights_a = torch.empty(num_tokens, topk, dtype=torch.float32, device="cuda")
    topk_weights_b = torch.empty_like(topk_weights_a)

    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        x_nvfp4_a,
        x_sf_a,
        topk_idx_a,
        topk_weights_a,
        norm_const=1.0,
    )
    stage_mega_moe_inputs(
        problem["hidden_states"],
        problem["topk_weights"],
        problem["topk_ids"],
        x_nvfp4_b,
        x_sf_b,
        topk_idx_b,
        topk_weights_b,
        norm_const=2.0,
    )

    assert not torch.equal(x_sf_a.view(torch.uint8), x_sf_b.view(torch.uint8))
    torch.testing.assert_close(topk_idx_a, topk_idx_b, atol=0, rtol=0)
    torch.testing.assert_close(topk_weights_a, topk_weights_b, atol=0, rtol=0)


def test_nvfp4_cutedsl_mega_kernel_is_registered():
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    kernel = create_mega_kernel(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=128, top_k=2)
    )
    assert kernel.kernel_name() == "sm100_nvfp4_nvfp4_bf16_cutedsl"


def test_nvfp4_cutedsl_config_exposes_ikr_and_combine_dtype():
    """The TRT-LLM-import knobs are plumbed through the FI backend config."""
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    cfg = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        in_kernel_fc2_reduce=True,
    )
    assert cfg.combine_dtype == "bf16"
    assert create_mega_kernel(cfg).kernel_name() == "sm100_nvfp4_nvfp4_bf16_cutedsl"

    cfg_q = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        combine_dtype="nvfp4",
    )
    assert create_mega_kernel(cfg_q).kernel_name() == "sm100_nvfp4_nvfp4_bf16_cutedsl"


def test_nvfp4_cutedsl_config_validates_situ():
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

    cfg = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=3072,
        top_k=16,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    assert cfg.activation == "situ"
    with pytest.raises(ValueError, match="requires situ_beta"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=3072,
            top_k=16,
            activation="situ",
        )
    with pytest.raises(ValueError, match="not supported with SiTU"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=3072,
            top_k=16,
            activation="situ",
            situ_beta=4.0,
            gate_up_clamp=10.0,
        )


def test_nvfp4_cutedsl_config_validates_shared_expert_shape():
    from flashinfer.moe_ep import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig

    cfg = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=3072,
        top_k=16,
        shared_hidden_size=7168,
        shared_intermediate_size=6144,
    )
    assert cfg.shared_hidden_size == 7168
    with pytest.raises(ValueError, match="must be set together"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=3072,
            top_k=16,
            shared_hidden_size=7168,
        )


def test_nvfp4_shim_config_rejects_invalid_ikr_combos():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        MegaMoENvfp4Config,
    )

    base = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=8,
        hidden=256,
        intermediate=256,
    )
    # ikr requires a bf16 combine wire.
    with pytest.raises(ValueError, match="in_kernel_fc2_reduce"):
        MegaMoENvfp4Config(
            **base,
            in_kernel_fc2_reduce=True,
            combine_dtype="nvfp4",
            token_back_mode="reuse_dispatch_warps",
        )
    # ikr requires the topk score folded before fc2.
    with pytest.raises(ValueError, match="apply_topk_in_fc1"):
        MegaMoENvfp4Config(**base, in_kernel_fc2_reduce=True, apply_topk_in_fc1=False)
    # quantized combine wires are only wired for dispatch-warp token-back.
    with pytest.raises(ValueError, match="reuse_dispatch_warps"):
        MegaMoENvfp4Config(**base, combine_dtype="mxfp8")
    with pytest.raises(ValueError, match="max_active_clusters"):
        MegaMoENvfp4Config(**base, max_active_clusters=0)
    with pytest.raises(ValueError, match="local_only requires world_size=1"):
        MegaMoENvfp4Config(**(base | {"world_size": 2}), local_only=True)


def test_tuner_is_valid_quantized_combine_rules():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import tuner

    # quantized combine excludes the in-kernel REDG reduce ...
    assert not tuner.is_valid(
        {
            "in_kernel_fc2_reduce": True,
            "token_back_mode": "reuse_dispatch_warps",
        },
        combine_format="16e2m1xbf16",
    )
    # ... and any non-dispatch-warp token-back.
    assert not tuner.is_valid(
        {"token_back_mode": "epi_warps"}, combine_format="16e2m1xbf16"
    )
    assert tuner.is_valid(
        {"token_back_mode": "reuse_dispatch_warps"}, combine_format="32e4m3xe8m0"
    )
    # bf16 combine composes ikr with every token-back mode.
    for mode in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
        assert tuner.is_valid({"in_kernel_fc2_reduce": True, "token_back_mode": mode})


def test_autotune_nvfp4_candidates_cover_ikr():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        nvfp4_candidates,
    )

    cands = nvfp4_candidates()
    assert any(k["in_kernel_fc2_reduce"] for k in cands)
    assert any(not k["in_kernel_fc2_reduce"] for k in cands)

    # A quantized wire prunes to the valid subset: no ikr, dispatch-warp
    # token-back only.
    qcands = nvfp4_candidates(combine_format="16e2m1xbf16")
    assert qcands
    assert all(
        not k["in_kernel_fc2_reduce"] and k["token_back_mode"] == "reuse_dispatch_warps"
        for k in qcands
    )

    pinned = nvfp4_candidates(allow_in_kernel_fc2_reduce=False)
    assert pinned and all(not k["in_kernel_fc2_reduce"] for k in pinned)
