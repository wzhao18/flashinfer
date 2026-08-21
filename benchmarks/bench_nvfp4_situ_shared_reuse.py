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
    symm.topk_idx.fill_(-1)
    symm.topk_idx[: args.tokens].zero_()
    symm.topk_weights.zero_()
    symm.topk_weights[: args.tokens].fill_(1.0)
    launch = nvfp4_mega_launch_thunk(fc1, fc2, symm)
    hidden = torch.randn(
        args.tokens, args.hidden, device="cuda", dtype=torch.bfloat16
    )
    topk_ids = torch.zeros(
        args.tokens, 1, device="cuda", dtype=torch.int64
    )
    topk_weights = torch.ones(
        args.tokens, 1, device="cuda", dtype=torch.float32
    )

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
