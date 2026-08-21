# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark the local fused BF16 SiTU shared expert."""

from __future__ import annotations

import argparse
import statistics

import torch

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import fused_bf16_situ_mlp


def _event_times(fn, warmup: int, iterations: int) -> list[float]:
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
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=6144)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--single-cta", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device=device)
    gate = torch.randn(
        args.intermediate, args.hidden, dtype=torch.bfloat16, device=device
    )
    up = torch.randn(
        args.intermediate, args.hidden, dtype=torch.bfloat16, device=device
    )
    down = torch.randn(
        args.hidden, args.intermediate, dtype=torch.bfloat16, device=device
    )
    blocks = args.intermediate // 32
    packed = torch.stack(
        (gate.view(blocks, 32, args.hidden), up.view(blocks, 32, args.hidden)),
        dim=1,
    )
    fc1_weight = packed.view(2 * args.intermediate, args.hidden).t().unsqueeze(0)
    fc2_weight = down.t().unsqueeze(0)

    def fused():
        return fused_bf16_situ_mlp(
            x,
            fc1_weight,
            fc2_weight,
            capacity=args.capacity,
            situ_beta=4.0,
            situ_linear_beta=25.0,
            single_cta=args.single_cta,
        )

    actual = fused().clone()
    gate_out = torch.mm(x, gate.t(), out_dtype=torch.float32)
    up_out = torch.mm(x, up.t(), out_dtype=torch.float32)
    gate_out = 4.0 * torch.tanh(gate_out / 4.0) * torch.sigmoid(gate_out)
    up_out = 25.0 * torch.tanh(up_out / 25.0)
    expected = (gate_out * up_out).to(torch.bfloat16) @ down.t()
    error = (actual.float() - expected.float()).norm() / expected.float().norm()
    max_error = (actual.float() - expected.float()).abs().max()
    print(f"relative_l2={error.item():.6f} max_abs={max_error.item():.3f}")
    if error.item() >= 1e-3:
        raise AssertionError(f"relative L2 error is too large: {error.item():.6f}")

    times = _event_times(fused, 10, args.iterations)
    print(
        f"tokens={args.tokens} single_cta={args.single_cta} "
        f"fused_p50_ms={statistics.median(times):.4f} "
        f"min_ms={min(times):.4f} max_ms={max(times):.4f}"
    )


if __name__ == "__main__":
    main()
