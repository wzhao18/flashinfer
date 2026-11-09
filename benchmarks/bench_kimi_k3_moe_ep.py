"""Time public Kimi or current layers with sparse layout-test weights.

Run from the repository root:
torchrun --standalone --nproc_per_node=8 -m benchmarks.bench_kimi_k3_moe_ep
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

from flashinfer import moe_ep
if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7):
    from tests.moe_ep.test_sm107_kimi_k3 import (  # noqa: E402
        EXPERTS,
        HIDDEN,
        _alpha,
        _check,
        _inputs,
        _reference,
        _runtime,
        _weights,
    )
else:
    from tests.moe_ep.test_sm100_kimi_k3 import (  # noqa: E402
        EXPERTS,
        HIDDEN,
        _alpha,
        _check,
        _inputs,
        _reference,
        _runtime,
        _weights,
    )


def _routing(mode, rank, tokens, seed):
    """Expert ids and weights for --routing; random / zipf / balanced match DeepGEMM bench_megamoe.make_routing."""
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    topk = 16
    if mode == "balanced":
        gid = rank * tokens + torch.arange(tokens, device="cuda")
        ids = (gid[:, None] * topk + torch.arange(topk, device="cuda")[None, :]) % EXPERTS
    elif mode == "random":
        scores = torch.randn((tokens, EXPERTS), device="cuda", generator=g)
        ids = scores.topk(topk, dim=-1).indices
    elif mode == "zipf":
        p = 1.0 / torch.arange(1, EXPERTS + 1, device="cuda", dtype=torch.float) ** 1.25
        perm = torch.randperm(
            EXPERTS, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)
        )
        ids = torch.multinomial(p[perm].expand(tokens, -1), topk, replacement=False, generator=g)
    else:
        raise ValueError(mode)
    weights = torch.rand((tokens, topk), device="cuda", generator=g) + 0.1
    return ids.int(), (weights / weights.sum(-1, keepdim=True)).float()


def _bench_weights(rank, world, kind, provider, rubin):
    """Test weights; the SM107 stock kernels require typed E4M3 / E8M0 block scales (same bytes)."""
    if not (provider == "current" and rubin):
        return _weights(rank, world, kind)
    count = EXPERTS // world
    from tests.moe_ep.test_sm107_kimi_k3 import INTERMEDIATE, _sparse_weight

    w13, s13 = _sparse_weight(2 * INTERMEDIATE, HIDDEN, count, rank * count, kind)
    w2, s2 = _sparse_weight(HIDDEN, INTERMEDIATE, count, rank * count, kind)
    scale_dtype = torch.float8_e4m3fn if kind == "nvfp4" else torch.float8_e8m0fnu
    return moe_ep.MoEWeightPack(w13, w2, s13.view(scale_dtype), s2.view(scale_dtype))


def _measure(
    layer,
    tensors,
    workspace,
    *,
    warmups,
    samples,
    replays,
    world,
    profile=False,
    calls_per_graph=1,
):
    layer.warmup(tensors, workspace=workspace)
    output = torch.empty_like(tensors.hidden_states)
    caller = replace(tensors, output=output)
    for _ in range(warmups):
        layer(caller, workspace=workspace)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_graph):
            layer(caller, workspace=workspace)
    try:
        graph.replay()
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(samples):
            if world > 1:
                dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            elapsed.append(
                start.elapsed_time(end) * 1000 / (replays * calls_per_graph)
            )
        values = torch.tensor(elapsed, dtype=torch.float64, device="cuda")
        rank_samples = [elapsed]
        if world > 1:
            rank_samples = [None] * world
            dist.all_gather_object(rank_samples, elapsed)
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        elapsed = values.cpu().tolist()
        quantiles = statistics.quantiles(elapsed, n=20, method="inclusive")
        kernels = (
            _profile_kernels(graph, replays * calls_per_graph, world)
            if profile
            else None
        )
        return output, {
            **({"kernels_us": kernels} if kernels is not None else {}),
            "median_us": statistics.median(elapsed),
            "p05_us": quantiles[0],
            "p95_us": quantiles[18],
            "min_us": min(elapsed),
            "max_us": max(elapsed),
            "samples_us": elapsed,
            "rank_samples_us": rank_samples,
        }
    finally:
        graph = None
        torch.cuda.synchronize()


def _profile_kernels(graph, replays, world):
    """Average device time per kernel name over ``replays`` graph replays (torch profiler, CUDA activity)."""
    if world > 1:
        dist.barrier()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    totals = {}
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            name = event.name[:120]
            totals[name] = totals.get(name, 0.0) + event.device_time
    return {name: round(total / replays, 2) for name, total in sorted(totals.items(), key=lambda item: -item[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["kimi", "current"], default="kimi")
    parser.add_argument(
        "--current-tile", type=int, nargs=3, metavar=("M", "N", "K"), default=None
    )
    parser.add_argument("--tokens", type=int, nargs="+", default=[256, 2048, 8192])
    parser.add_argument(
        "--routing",
        choices=["pattern", "balanced", "random", "zipf"],
        default="pattern",
        help="pattern = the test inputs (one route per rank, local expert = row %% experts per rank); "
        "balanced / random / zipf = DeepGEMM bench_megamoe routing",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calls-per-graph",
        type=int,
        default=1,
        help="layer calls captured back to back in one CUDA graph (DeepGEMM bench_megamoe uses 10)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="fleet max_tokens_per_rank (tactic bucket / router geometry); default max(--tokens)",
    )
    parser.add_argument(
        "--current-variant",
        choices=["inference", "genphase", "auto"],
        default=None,
        help="SM107 stock kernel_variant (provider=current on SM107)",
    )
    parser.add_argument(
        "--kimi-fast-situ",
        action="store_true",
        help="Kimi fast_situ=True (production SiTU core; can change output bits)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also report the average device time per kernel of one graph replay (torch profiler)",
    )
    parser.add_argument(
        "--kimi-knobs",
        type=lambda value: json.loads(
            Path(value[1:]).read_text() if value.startswith("@") else value
        ),
        default=None,
        help="JSON dict (or @file.json) of fixed Kimi knobs (provider=kimi), e.g. a tile override",
    )
    parser.add_argument(
        "--quant-kind", choices=["all", "nvfp4", "mxfp4_mxfp8"], default="all"
    )
    parser.add_argument("--wire", choices=["all", "bf16", "mxfp8"], default="all")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.current_tile is not None and (
        args.provider != "current" or min(args.current_tile) < 1
    ):
        parser.error("--current-tile requires provider=current and positive M N K")
    if args.kimi_knobs is not None and (
        args.provider != "kimi" or not isinstance(args.kimi_knobs, dict)
    ):
        parser.error("--kimi-knobs requires provider=kimi and a JSON object")
    if args.provider == "current":
        rubin_device = torch.cuda.get_device_capability() == (10, 7)
        kinds_ok = ("nvfp4", "mxfp4_mxfp8") if rubin_device else ("nvfp4",)
        if args.quant_kind not in ("all", *kinds_ok) or args.wire not in ("all", "bf16"):
            parser.error(f"the current-provider comparison supports {kinds_ok} with BF16 combine")
        if args.quant_kind == "all":
            args.quant_kind = "nvfp4"
        args.wire = "bf16"
    if args.calls_per_graph < 1 or (
        args.max_tokens is not None and args.max_tokens < max(args.tokens)
    ):
        parser.error("--calls-per-graph >= 1 and --max-tokens >= every --tokens")
    if min(args.tokens) < 1 or args.warmups < 1 or args.samples < 2 or args.replays < 1:
        parser.error(
            "tokens/warmups/replays must be positive; samples must be at least two"
        )
    rank, world = _runtime()
    if args.provider == "current" and world == 1:
        os.environ["MEGA_NO_DIST"] = "1"
    kinds = ["nvfp4", "mxfp4_mxfp8"] if args.quant_kind == "all" else [args.quant_kind]
    wires = ["bf16", "mxfp8"] if args.wire == "all" else [args.wire]
    report = {
        "api": "flashinfer.moe_ep.MoEEpMegaLayer",
        "provider": args.provider,
        "current_tile": args.current_tile,
        "kimi_knobs": args.kimi_knobs,
        "measurement": "warm graph forward, CUDA events, slowest EP rank per sample",
        "weights": "deterministic packed sparse layout-test weights",
        "routing": args.routing,
        "seed": args.seed,
        "input": "BF16; includes input quantization, routing, compute and combine into caller output",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ep_size": world,
        "experts": EXPERTS,
        "hidden": HIDDEN,
        "intermediate": 3072,
        "topk": 16,
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
        "fast_situ": args.kimi_fast_situ,
        "current_variant": args.current_variant,
        "fast_math": False,
        "warmups": args.warmups,
        "samples": args.samples,
        "replays_per_sample": args.replays,
        "calls_per_graph": args.calls_per_graph,
        "max_tokens_per_rank": args.max_tokens,
        "results": [],
    }
    if rank == 0:
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "results"}
            ),
            flush=True,
        )
    try:
        for kind in kinds:
            for wire in wires:
                calibration = {}
                if kind == "nvfp4":
                    expert = torch.arange(EXPERTS // world, device="cuda") + rank * (
                        EXPERTS // world
                    )
                    alpha1, alpha2, norm = _alpha(expert)
                    calibration = dict(
                        fc1_alpha=alpha1.float(),
                        fc2_alpha=alpha2.float(),
                        fc1_norm_const=norm.float(),
                    )
                options = dict(
                    intermediate_size=3072,
                    top_k=16,
                    activation="situ",
                    combine_dtype=wire,
                    situ_beta=4.0,
                    situ_linear_beta=25.0,
                    apply_routing_weights_before_fc2=False,
                    finalize_mode=moe_ep.FinalizeMode.DEFAULT,
                    **calibration,
                )
                if args.current_tile is not None:
                    options["knobs"] = {"mma_tiler_mnk": tuple(args.current_tile)}
                rubin = torch.cuda.get_device_capability() == (10, 7)
                if args.provider == "kimi":
                    kimi_config = (
                        moe_ep.Sm107_KimiK3_MegaMoeConfig
                        if rubin
                        else moe_ep.Sm100_KimiK3_MegaMoeConfig
                    )
                    if args.kimi_knobs:
                        options["knobs"] = {
                            key: tuple(value) if isinstance(value, list) else value
                            for key, value in args.kimi_knobs.items()
                        }
                    config = kimi_config(
                        quant_kind=kind, fast_situ=args.kimi_fast_situ, **options
                    )
                elif rubin:
                    # The SM107 kernel has no finalize mode; use its tuned knob cache by default.
                    rubin_options = {
                        k: v for k, v in options.items() if k != "finalize_mode"
                    }
                    rubin_options.setdefault("knobs", "cache")
                    if args.current_variant is not None:
                        rubin_options["kernel_variant"] = args.current_variant
                    stock_config = (
                        moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
                        if kind == "nvfp4"
                        else moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
                    )
                    config = stock_config(**rubin_options)
                else:
                    config = moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl_next",
                        fast_math=False,
                        **options,
                    )
                layer = moe_ep.MoEEpMegaLayer(
                    bootstrap=moe_ep.BootstrapConfig(
                        world_size=world,
                        rank=rank,
                        auto_bootstrap=args.provider == "current",
                        process_group=dist.group.WORLD if world > 1 else None,
                        device=torch.cuda.current_device(),
                    ),
                    fleet_params=moe_ep.FleetParams(
                        num_experts=EXPERTS,
                        max_tokens_per_rank=args.max_tokens or max(args.tokens),
                        token_hidden_size=HIDDEN,
                    ),
                    weights=_bench_weights(rank, world, kind, args.provider, rubin),
                    backend=moe_ep.MegaConfig(megakernel=config, quantize_input=True),
                )
                try:
                    for tokens in args.tokens:
                        workspace = layer.create_workspace(tokens)
                        try:
                            tensors = _inputs(tokens, rank, world, 0, mask_routes=False)
                            if args.routing != "pattern":
                                ids, weights = _routing(
                                    args.routing, rank, tokens, args.seed
                                )
                                tensors = replace(
                                    tensors, topk_ids=ids, topk_weights=weights
                                )
                            output, timing = _measure(
                                layer,
                                tensors,
                                workspace,
                                warmups=args.warmups,
                                samples=args.samples,
                                replays=args.replays,
                                world=world,
                                profile=args.profile,
                                calls_per_graph=args.calls_per_graph,
                            )
                            sample = replace(
                                tensors,
                                hidden_states=tensors.hidden_states[:8],
                                topk_ids=tensors.topk_ids[:8],
                                topk_weights=tensors.topk_weights[:8],
                            )
                            _check(output[:8], _reference(sample, kind, wire))
                            row = {
                                "provider": args.provider,
                                "kernel_selector": config.kernel_name,
                                "quant_kind": kind,
                                "combine_dtype": wire,
                                "tokens_per_rank": tokens,
                                **timing,
                            }
                            report["results"].append(row)
                            if rank == 0:
                                print(json.dumps(row), flush=True)
                                if args.output:
                                    args.output.parent.mkdir(
                                        parents=True, exist_ok=True
                                    )
                                    args.output.write_text(
                                        json.dumps(report, indent=2) + "\n"
                                    )
                        finally:
                            torch.cuda.synchronize()
                            workspace.destroy()
                finally:
                    layer.destroy()
                del layer
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
