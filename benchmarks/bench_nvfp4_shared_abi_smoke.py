"""Compile the NVFP4 MegaMoE launch ABI with shared-expert tensors."""

from __future__ import annotations

import dataclasses
import os

import torch
from flashinfer import mm_fp4

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
    MegaMoESharedNvfp4Inputs,
    create_dummy_inputs,
    nvfp4_mega_launch_thunk,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.src.moe_nvfp4_swapab.runner_common import (
    from_blocked,
    to_blocked,
    unpack_fp4_to_f32,
)


def main() -> None:
    os.environ["MEGA_NO_DIST"] = "1"
    tokens = int(os.environ.get("TOKENS", 16))
    routed_capacity = int(os.environ.get("ROUTED_CAPACITY", tokens))
    intermediate = 3072
    match_outer_dims = os.environ.get("MATCH_OUTER_DIMS") == "1"
    routed_hidden = 7168 if match_outer_dims else 3584
    routed_gateup = 4 * intermediate if match_outer_dims else 2 * intermediate
    routed_experts = 1 if match_outer_dims else 56
    routed_topk = 1 if match_outer_dims else 16
    _, routed_fc1, routed_fc2, routed = create_dummy_inputs(
        0,
        1,
        routed_experts,
        routed_capacity,
        tokens,
        routed_topk,
        routed_hidden,
        routed_gateup,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        knobs={
            "cluster_shape_mnk": (2, 1, 1),
            "group_hint": 512,
            "max_active_clusters": 60,
            "epi_flag_batch": (2, 4),
            "load_balance_mode": "atomic_counter",
            "mma_tiler_mnk": (256, 128, 256),
            "flag_batch": 4,
            "token_back_mode": "standalone_warps",
        },
    )
    shared_hidden = 7168
    shared_gateup = 4 * intermediate
    shared_down = shared_gateup // 2
    _, shared_fc1, shared_fc2, shared_buffer = create_dummy_inputs(
        0,
        1,
        1,
        tokens,
        tokens,
        1,
        shared_hidden,
        shared_gateup,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        knobs={
            "cluster_shape_mnk": (2, 1, 1),
            "group_hint": 512,
            "max_active_clusters": 60,
            "epi_flag_batch": (2, 4),
            "load_balance_mode": "atomic_counter",
            "mma_tiler_mnk": (256, 128, 256),
            "flag_batch": 4,
            "token_back_mode": "standalone_warps",
        },
    )
    shared_buffer.topk_weights.fill_(1)
    fp4 = torch.float4_e2m1fn_x2
    sf_cols = ((shared_down + 15) // 16 + 3) // 4 * 4
    sf_rows = ((tokens + 127) // 128) * 128
    shared_activation_sf = to_blocked(
        shared_buffer.x_sf[:tokens, : shared_hidden // 16]
    ).reshape(sf_rows, shared_hidden // 16)
    shared = MegaMoESharedNvfp4Inputs(
        activation=shared_buffer.x,
        activation_sf=shared_activation_sf,
        fc1_weight=shared_fc1[0],
        fc1_weight_sf=shared_fc1[1],
        fc1_output=torch.empty(
            tokens, shared_down // 2, dtype=fp4, device="cuda"
        ),
        fc1_output_sf=torch.zeros(
            sf_rows, sf_cols, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        fc2_weight=shared_fc2[0],
        fc2_weight_sf=shared_fc2[1],
        fc1_alpha=shared_buffer.fc1_alpha,
        fc2_alpha=shared_buffer.fc2_alpha,
        fc1_norm_const=shared_buffer.fc1_norm_const,
        fc1_done_counter=torch.zeros(tokens, dtype=torch.int32, device="cuda"),
        output_activation=shared_buffer.output_activation,
    )
    routed._frontend = MegaMoENvfp4Frontend(
        dataclasses.replace(
            routed._frontend.config,
            shared_hidden=shared_hidden,
            shared_intermediate=shared_gateup,
        )
    )
    inputs = MegaMoENvfp4Inputs(
        activation=routed.x,
        activation_sf=routed.x_sf,
        topk_idx=routed.topk_idx,
        topk_weights=routed.topk_weights,
        fc1_weight=routed_fc1[0],
        fc1_weight_sf=routed_fc1[1],
        fc2_weight=routed_fc2[0],
        fc2_weight_sf=routed_fc2[1],
        fc1_alpha=routed.fc1_alpha,
        fc2_alpha=routed.fc2_alpha,
        fc1_norm_const=routed.fc1_norm_const,
        output_activation=routed.output_activation,
        shared=shared,
    )
    print(routed._frontend.config, flush=True)
    print(shared.fc1_weight.shape, shared.fc1_output.shape, flush=True)
    print(
        "shared scales",
        shared.fc1_alpha.item(),
        shared.fc2_alpha.item(),
        shared.fc1_norm_const.item(),
        flush=True,
    )
    reference_thunk = nvfp4_mega_launch_thunk(
        shared_fc1, shared_fc2, shared_buffer
    )
    shared_activation_before = shared.activation.view(torch.uint8).clone()
    shared_activation_sf_before = shared.activation_sf.view(torch.uint8).clone()
    reference_thunk()
    torch.cuda.synchronize()
    print("standalone reference passed", flush=True)
    print(
        "standalone input byte changes=",
        int(
            (
                shared.activation.view(torch.uint8) != shared_activation_before
            ).sum()
        ),
        int(
            (
                shared.activation_sf.view(torch.uint8)
                != shared_activation_sf_before
            ).sum()
        ),
        flush=True,
    )
    reference = shared_buffer.output_activation[:tokens].clone()
    reference_mega = shared_buffer._frontend._mega
    reference_kernel = reference_mega.kernel
    reference_workspace = reference_mega.local_workspace
    reference_fc1_spec = reference_kernel._local_region_by_name["fc1_output"]
    reference_fc1_offset = reference_kernel._local_offsets["fc1_output"]
    reference_fc1_rows = reference_fc1_spec.shape[0]
    reference_fc1_bytes = reference_fc1_rows * shared_down // 2
    reference_fc1 = (
        reference_workspace[
            reference_fc1_offset : reference_fc1_offset + reference_fc1_bytes
        ]
        .view(fp4)
        .reshape(reference_fc1_rows, shared_down // 2)[:tokens]
        .clone()
    )
    reference_metadata_spec = reference_kernel._local_region_by_name[
        "token_src_metadata"
    ]
    reference_metadata_offset = reference_kernel._local_offsets[
        "token_src_metadata"
    ]
    reference_metadata_bytes = (
        reference_metadata_spec.shape[0] * reference_metadata_spec.shape[1]
    )
    reference_metadata = reference_workspace[
        reference_metadata_offset : reference_metadata_offset
        + reference_metadata_bytes
    ].reshape(reference_metadata_spec.shape)
    reference_token_ids = (
        reference_metadata[:tokens].contiguous().view(torch.int32)[:, 0].long()
    )
    reference_fc1_original_bytes = torch.empty_like(reference_fc1.view(torch.uint8))
    reference_fc1_original_bytes[reference_token_ids] = reference_fc1.view(torch.uint8)
    reference_fc1_original = reference_fc1_original_bytes.view(fp4)
    reference_sf_spec = reference_kernel._local_region_by_name["fc1_output_sf"]
    reference_sf_offset = reference_kernel._local_offsets["fc1_output_sf"]
    reference_sf_bytes = reference_sf_spec.shape[0] * reference_sf_spec.shape[1]
    reference_fc1_sf = (
        reference_workspace[
            reference_sf_offset : reference_sf_offset + reference_sf_bytes
        ]
        .view(torch.float8_e4m3fn)
        .reshape(reference_sf_spec.shape)
        .clone()
    )
    reference_fc1_sf_raw = from_blocked(
        reference_fc1_sf[:sf_rows].flatten(), tokens, shared_down // 16
    )
    reference_fc1_sf_original_bytes = torch.empty_like(
        reference_fc1_sf_raw.view(torch.uint8)
    )
    reference_fc1_sf_original_bytes[reference_token_ids] = reference_fc1_sf_raw.view(
        torch.uint8
    )
    reference_fc1_sf_original_raw = reference_fc1_sf_original_bytes.view(
        torch.float8_e4m3fn
    )
    reference_fc1_sf_original = to_blocked(reference_fc1_sf_original_raw).reshape(
        sf_rows, sf_cols
    )
    reference_mm_pool = mm_fp4(
        reference_fc1.view(torch.uint8),
        shared.fc2_weight[0].view(torch.uint8),
        reference_fc1_sf[:sf_rows],
        shared.fc2_weight_sf[0],
        alpha=shared.fc2_alpha,
        backend="cute-dsl",
    )
    reference_mm = torch.empty_like(reference_mm_pool)
    reference_mm[reference_token_ids] = reference_mm_pool
    print(
        "standalone fc1 + dense fc2/reference max diff=",
        (reference_mm.float() - reference.float()).abs().max().item(),
        flush=True,
    )
    shared.output_activation.zero_()
    if os.environ.get("PRESEED_SHARED_COUNTER") == "1":
        shared.fc1_done_counter.fill_(1_000_000)
    thunk = routed._frontend.make_launch_thunk(inputs)
    print("integrated thunk compiled", flush=True)
    thunk()
    if tokens > 256:
        mm_fp4(
            shared.fc1_output.view(torch.uint8),
            shared.fc2_weight[0].view(torch.uint8),
            shared.fc1_output_sf,
            shared.fc2_weight_sf[0],
            alpha=shared.fc2_alpha,
            out=shared.output_activation,
            backend="cute-dsl",
        )
    torch.cuda.synchronize()
    print(
        "integrated fc2/reference max diff=",
        (shared.output_activation.float() - reference.float()).abs().max().item(),
        flush=True,
    )
    torch.testing.assert_close(shared.output_activation, reference)
    raw_fc1_sf = from_blocked(
        shared.fc1_output_sf.flatten(), tokens, shared_down // 16
    ).float()
    print(
        "integrated fc1 sf",
        "finite=",
        int(torch.isfinite(raw_fc1_sf).sum()),
        "/",
        raw_fc1_sf.numel(),
        "min=",
        raw_fc1_sf.nan_to_num().min().item(),
        "max=",
        raw_fc1_sf.nan_to_num().max().item(),
        flush=True,
    )
    print(
        "integrated fc1",
        "packed byte mismatch=",
        int(
            (
                shared.fc1_output.view(torch.uint8)
                != reference_fc1_original.view(torch.uint8)
            )
            .sum()
            .item()
        ),
        "sf mismatch=",
        int(
            (
                shared.fc1_output_sf.view(torch.uint8)
                != reference_fc1_sf_original.view(torch.uint8)
            )
            .sum()
            .item()
        ),
        flush=True,
    )
    reference_fc1_deq = unpack_fp4_to_f32(reference_fc1_original)
    reference_fc1_deq *= reference_fc1_sf_original_raw.float().repeat_interleave(
        16, dim=1
    )
    integrated_fc1_deq = unpack_fp4_to_f32(shared.fc1_output)
    integrated_fc1_deq *= raw_fc1_sf.repeat_interleave(16, dim=1)
    fc1_noise = (integrated_fc1_deq - reference_fc1_deq).pow(2).mean()
    fc1_signal = reference_fc1_deq.pow(2).mean()
    print(
        "integrated fc1 SNR dB=",
        (10 * torch.log10(fc1_signal / fc1_noise)).item(),
        flush=True,
    )
    mm_fp4(
        shared.fc1_output.view(torch.uint8),
        shared.fc2_weight[0].view(torch.uint8),
        shared.fc1_output_sf,
        shared.fc2_weight_sf[0],
        alpha=shared.fc2_alpha,
        out=shared.output_activation,
        backend="cute-dsl",
    )
    torch.cuda.synchronize()
    print(
        "split/reference max diff=",
        (shared.output_activation.float() - reference.float()).abs().max().item(),
        "NRMSE=",
        (
            (shared.output_activation.float() - reference.float())
            .pow(2)
            .mean()
            .sqrt()
            / reference.float().pow(2).mean().sqrt()
        ).item(),
        flush=True,
    )
    torch.testing.assert_close(shared.output_activation, reference)
    print("shared ABI compile, launch, and correctness passed")

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.backend import (
        Nvfp4CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    backend = Nvfp4CutedslMegaKernelBackend(
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate,
            top_k=routed_topk,
            activation="situ",
            situ_beta=4.0,
            situ_linear_beta=25.0,
            shared_hidden_size=shared_hidden,
            shared_intermediate_size=shared_down,
        )
    )
    routed._mega_shared_inputs = shared
    backend_output = torch.empty(
        tokens, routed_hidden, dtype=torch.bfloat16, device="cuda"
    )
    shared.output_activation.zero_()
    backend.compute(
        routed,
        (routed_fc1, routed_fc2),
        output=backend_output,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(shared.output_activation, reference)
    print("backend fused launch and correctness passed")

    routed_only_frontend = MegaMoENvfp4Frontend(
        dataclasses.replace(
            routed._frontend.config,
            shared_hidden=None,
            shared_intermediate=None,
        )
    )
    routed_only_thunk = routed_only_frontend.make_launch_thunk(
        dataclasses.replace(inputs, shared=None)
    )

    def dense_fc2() -> None:
        mm_fp4(
            shared.fc1_output.view(torch.uint8),
            shared.fc2_weight[0].view(torch.uint8),
            shared.fc1_output_sf,
            shared.fc2_weight_sf[0],
            alpha=shared.fc2_alpha,
            out=shared.output_activation,
            backend="cute-dsl",
        )

    def sequential_baseline() -> None:
        routed_only_thunk()
        reference_thunk()

    def bench_ms(fn, warmup: int = 5, iterations: int = 30) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / iterations

    timings = {
        "routed_only": bench_ms(routed_only_thunk),
        "shared_standalone": bench_ms(reference_thunk),
        "sequential_baseline": bench_ms(sequential_baseline),
        "routed_plus_shared_fc12": bench_ms(thunk),
        "dense_shared_fc2": bench_ms(dense_fc2),
    }
    print("timings_ms", timings, flush=True)


if __name__ == "__main__":
    main()
