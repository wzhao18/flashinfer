# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark an NVFP4 SiTU shared expert reusing routed activations."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
    create_dummy_inputs,
    nvfp4_mega_launch_thunk,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.staging import (
    stage_mega_moe_inputs,
)


def event_times(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=6144)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-active-clusters", type=int)
    parser.add_argument("--correctness", action="store_true")
    args = parser.parse_args()

    if not 0 < args.tokens <= args.capacity:
        raise ValueError("tokens must be in [1, capacity].")
    os.environ["MEGA_NO_DIST"] = "1"
    _, fc1, fc2, symm = create_dummy_inputs(
        0,
        1,
        1,
        args.capacity,
        args.tokens,
        1,
        args.hidden,
        2 * args.intermediate,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        fc1_alpha=torch.ones(1, device="cuda", dtype=torch.float32),
        fc2_alpha=torch.ones(1, device="cuda", dtype=torch.float32),
        fc1_norm_const=torch.ones(1, device="cuda", dtype=torch.float32),
        knobs=(
            {"max_active_clusters": args.max_active_clusters}
            if args.max_active_clusters is not None
            else None
        ),
    )
    reference_weights = None
    if args.correctness:
        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
            preprocess_mega_weights,
        )

        gate = (
            torch.randn(
                args.intermediate,
                args.hidden,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 64
        )
        up = torch.randn_like(gate) / 64
        down = (
            torch.randn(
                args.hidden,
                args.intermediate,
                device="cuda",
                dtype=torch.bfloat16,
            )
            / 64
        )
        fc1, fc2 = preprocess_mega_weights(
            MoEWeightPack(
                w13=torch.cat((gate, up), dim=0).unsqueeze(0),
                w2=down.unsqueeze(0),
            ),
            intermediate_size=args.intermediate,
            hidden_size=args.hidden,
        )
        reference_weights = gate, up, down

    symm.topk_idx.fill_(-1)
    symm.topk_idx[: args.tokens].zero_()
    symm.topk_weights.zero_()
    symm.topk_weights[: args.tokens].fill_(1.0)
    launch = nvfp4_mega_launch_thunk(fc1, fc2, symm)
    hidden = torch.randn(args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16)
    topk_ids = torch.zeros(args.tokens, 1, device="cuda", dtype=torch.int64)
    topk_weights = torch.ones(args.tokens, 1, device="cuda", dtype=torch.float32)

    def stage() -> None:
        stage_mega_moe_inputs(
            hidden,
            topk_weights,
            topk_ids,
            symm.x,
            symm.x_sf,
            symm.topk_idx,
            symm.topk_weights,
        )

    def end_to_end() -> None:
        stage()
        launch()

    if reference_weights is not None:
        stage()
        launch()
        actual = symm.output_activation[: args.tokens].clone()
        gate, up, down = reference_weights
        gate_output = hidden @ gate.t()
        up_output = hidden @ up.t()
        gate_output = 4.0 * torch.tanh(gate_output / 4.0) * torch.sigmoid(gate_output)
        up_output = 25.0 * torch.tanh(up_output / 25.0)
        expected = (gate_output * up_output).to(torch.bfloat16) @ down.t()
        relative_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        print(f"correctness: relative_l2={relative_l2:.6f} cosine={cosine:.8f}")

    kernel_times = event_times(launch, args.warmup, args.iterations)
    stage_times = event_times(stage, args.warmup, args.iterations)
    total_times = event_times(end_to_end, args.warmup, args.iterations)
    print(
        f"reuse_nvfp4_activation clusters="
        f"{args.max_active_clusters or 'all'}: "
        f"kernel_p50_us={statistics.median(kernel_times) * 1000:.3f} "
        f"stage_p50_us={statistics.median(stage_times) * 1000:.3f} "
        f"total_p50_us={statistics.median(total_times) * 1000:.3f}"
    )
    symm.destroy()


if __name__ == "__main__":
    main()
