# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Benchmark two always-selected shared experts with BF16 MegaMoE."""

from __future__ import annotations

import argparse
import os
import statistics

import torch

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import (
    MegaMoEBf16Config,
    MegaMoEBf16Frontend,
    MegaMoEBf16Inputs,
)


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


def _pack_fc1(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    experts, intermediate, hidden = gate.shape
    blocks = intermediate // 32
    return (
        torch.stack(
            (
                gate.view(experts, blocks, 32, hidden),
                up.view(experts, blocks, 32, hidden),
            ),
            dim=2,
        )
        .view(experts, 2 * intermediate, hidden)
        .transpose(1, 2)
        .contiguous()
    )


def _reference(
    x: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    result = torch.zeros_like(x)
    for expert in range(gate.shape[0]):
        gate_out = torch.mm(x, gate[expert].t(), out_dtype=torch.float32)
        up_out = torch.mm(x, up[expert].t(), out_dtype=torch.float32)
        gate_out = 4.0 * torch.tanh(gate_out / 4.0) * torch.sigmoid(gate_out)
        up_out = 25.0 * torch.tanh(up_out / 25.0)
        result.add_((gate_out * up_out).to(torch.bfloat16) @ down[expert].t())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[16, 20, 32, 64, 160, 256]
    )
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=6144)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--in-kernel-reduce", action="store_true")
    parser.add_argument("--max-active-clusters", type=int)
    args = parser.parse_args()

    os.environ["MEGA_NO_DIST"] = "1"
    torch.manual_seed(0)
    device = torch.device("cuda")
    gate = torch.randn(
        2, args.intermediate, args.hidden, dtype=torch.bfloat16, device=device
    )
    up = torch.randn_like(gate)
    down = torch.randn(
        2, args.hidden, args.intermediate, dtype=torch.bfloat16, device=device
    )
    fc1_weight = _pack_fc1(gate, up)
    fc2_weight = down.transpose(1, 2).contiguous()
    activation = torch.zeros(
        args.capacity, args.hidden, dtype=torch.bfloat16, device=device
    )
    topk_idx = torch.full(
        (args.capacity, 2), -1, dtype=torch.int64, device=device
    )
    topk_weights = torch.zeros(
        args.capacity, 2, dtype=torch.float32, device=device
    )
    topk_dim = 1 if args.in_kernel_reduce else 2
    combine_output = torch.zeros(
        args.capacity,
        topk_dim,
        args.hidden,
        dtype=torch.bfloat16,
        device=device,
    )
    config = MegaMoEBf16Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=args.capacity,
        num_topk=2,
        num_total_experts=2,
        hidden=args.hidden,
        intermediate=args.intermediate,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        in_kernel_fc2_reduce=args.in_kernel_reduce,
        token_back_mode=(
            "reuse_dispatch_warps" if args.in_kernel_reduce else "epi_warps"
        ),
        apply_topk_in_fc1=args.in_kernel_reduce,
        max_active_clusters=args.max_active_clusters,
    )
    frontend = MegaMoEBf16Frontend(config)
    inputs = MegaMoEBf16Inputs(
        activation,
        topk_idx,
        topk_weights,
        fc1_weight,
        fc2_weight,
        combine_output,
    )

    for tokens in args.tokens:
        if tokens > args.capacity:
            raise ValueError(f"tokens={tokens} exceeds capacity={args.capacity}.")
        activation.zero_()
        activation[:tokens].normal_()
        topk_idx.fill_(-1)
        topk_idx[:tokens, 0] = 0
        topk_idx[:tokens, 1] = 1
        topk_weights.zero_()
        topk_weights[:tokens] = 1.0

        def fused():
            output = frontend.run(inputs, num_tokens=tokens)
            return output[:, 0] if args.in_kernel_reduce else output.sum(dim=1)

        actual = fused()[:tokens].clone()
        expected = _reference(activation[:tokens], gate, up, down)
        relative_l2 = (
            (actual.float() - expected.float()).norm() / expected.float().norm()
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        times = _event_times(fused, args.warmup, args.iterations)
        print(
            f"clusters={args.max_active_clusters or 'all'} tokens={tokens} "
            f"p50_ms={statistics.median(times):.6f} "
            f"min_ms={min(times):.6f} relative_l2={relative_l2:.6f} "
            f"cosine={cosine:.8f}"
        )
    frontend.release()


if __name__ == "__main__":
    main()
