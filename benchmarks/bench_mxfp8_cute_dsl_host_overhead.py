# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host time per call of the CuTe-DSL MXFP8 linear: mxfp8_quantize + mm_mxfp8.

Measures, per (M, N, K), the host time to issue one call of

- ``mxfp8_quantize(x, backend="cute-dsl")``                   (activation quantize)
- ``mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")``          (GEMM)

as the average enqueue time of back-to-back calls on an idle GPU (no
synchronization inside the timed loop; the loop is far shorter than the launch
queue). In eager execution this host time, not the kernel time, bounds a layer
whenever it is the larger of the two.

Modes:

- ``cached``   repeated calls with the same signature (the default behavior);
- ``uncached`` the per-signature memos are dropped before every call, so each
  call takes the full path (checks, runner selection, tactic resolution);
- ``launch``   the memoized GEMM launch alone (compiled-kernel call plus the
  output allocation), the floor for ``cached``.

The modes other than ``cached`` need a FlashInfer with the launch memos; on an
older checkout only ``cached`` (which is then the full path) is reported, so
the same command compares two checkouts. Cached and uncached outputs are
checked to be bitwise identical. ``--autotune`` tunes every shape first.

Default shapes are the dense projections of a Kimi-K3-sized model at TP8
(K=7168 hidden): MLA fused q/kv-a (2112), linear-attention q/k/v (4608),
attention o_proj (K=1536), shared-expert gate_up (1536) and down (K=768).

    python benchmarks/bench_mxfp8_cute_dsl_host_overhead.py
    python benchmarks/bench_mxfp8_cute_dsl_host_overhead.py --m 2160 --nk 4608x7168 --autotune
"""

import argparse
import json
import statistics
import time

import torch

import flashinfer
from flashinfer import autotune, mm_mxfp8, mxfp8_quantize
from flashinfer.gemm import gemm_base

try:
    from flashinfer.quantization.kernels.mxfp8_quantize import (
        _mxfp8_quantize_launch_plan,
    )
except ImportError:
    _mxfp8_quantize_launch_plan = None

_HAS_MEMOS = (
    hasattr(gemm_base, "_MM_MXFP8_LAUNCH_CACHE")
    and _mxfp8_quantize_launch_plan is not None
)

DEFAULT_M = (500, 2160, 4320, 8640)
DEFAULT_NK = ((2112, 7168), (4608, 7168), (7168, 1536), (1536, 7168), (7168, 768))


def _drop_memos():
    gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()
    _mxfp8_quantize_launch_plan.cache_clear()


def _host_us(fn, iters, reps, before_each=None):
    """Median over ``reps`` of the mean enqueue time of ``iters`` calls."""
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        if before_each is None:
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                before_each()
                fn()
            t1 = time.perf_counter()
        samples.append((t1 - t0) / iters * 1e6)
    torch.cuda.synchronize()
    return statistics.median(samples)


def _bench_shape(m, n, k, iters, reps):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    w_q, w_sf = mxfp8_quantize(w, is_sf_swizzled_layout=True, backend="cute-dsl")
    b = w_q.T  # (k, n), column-major

    def quant():
        return mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cute-dsl")

    a, a_sf = quant()

    def gemm():
        return mm_mxfp8(a, b, a_sf, w_sf, out_dtype=torch.bfloat16, backend="cute-dsl")

    # Warm up: compile, select, memoize.
    for _ in range(3):
        quant()
        gemm()
    row = dict(m=m, n=n, k=k)
    row["quant_cached_us"] = _host_us(quant, iters, reps)
    row["gemm_cached_us"] = _host_us(gemm, iters, reps)
    if _HAS_MEMOS:
        row["quant_uncached_us"] = _host_us(quant, iters, reps, _drop_memos)
        row["gemm_uncached_us"] = _host_us(gemm, iters, reps, _drop_memos)
        gemm()  # re-memoize
        (launch,) = gemm_base._MM_MXFP8_LAUNCH_CACHE.values()

        def launch_only():
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            launch(a, b, a_sf, w_sf, out)

        row["gemm_launch_us"] = _host_us(launch_only, iters, reps)

        # Bitwise: memoized path vs full path, same inputs.
        q_hit, sf_hit = quant()
        out_hit = gemm()
        _drop_memos()
        q_full, sf_full = quant()
        _drop_memos()
        out_full = gemm()
        torch.cuda.synchronize()
        row["bitwise_equal"] = bool(
            torch.equal(q_hit.view(torch.uint8), q_full.view(torch.uint8))
            and torch.equal(sf_hit, sf_full)
            and torch.equal(out_hit.view(torch.int16), out_full.view(torch.int16))
        )
    if _HAS_MEMOS:
        gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--m", type=int, nargs="+", default=list(DEFAULT_M))
    p.add_argument(
        "--nk",
        nargs="+",
        default=[f"{n}x{k}" for n, k in DEFAULT_NK],
        help="NxK weight shapes",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--autotune", action="store_true", help="tune every shape first")
    p.add_argument("--output", help="write the rows as JSON")
    args = p.parse_args()

    shapes = [(m, *map(int, nk.split("x"))) for nk in args.nk for m in sorted(args.m)]
    if args.autotune:
        for m, n, k in shapes:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            a, a_sf = mxfp8_quantize(x, backend="cute-dsl")
            w_q, w_sf = mxfp8_quantize(w, backend="cute-dsl")
            with autotune(True):
                mm_mxfp8(a, w_q.T, a_sf, w_sf, backend="cute-dsl")

    print(
        f"flashinfer {flashinfer.__version__} from {flashinfer.__file__}, "
        f"{torch.cuda.get_device_name()}, autotune={args.autotune}"
    )
    cols = ["quant_cached_us", "gemm_cached_us"]
    if _HAS_MEMOS:
        cols += [
            "quant_uncached_us",
            "gemm_uncached_us",
            "gemm_launch_us",
            "bitwise_equal",
        ]
    print(f"{'M':>6} {'N':>6} {'K':>6} " + " ".join(f"{c:>18}" for c in cols))
    rows = []
    for m, n, k in shapes:
        row = _bench_shape(m, n, k, args.iters, args.reps)
        rows.append(row)
        print(
            f"{m:>6} {n:>6} {k:>6} "
            + " ".join(
                f"{row[c]:>18.1f}"
                if isinstance(row[c], float)
                else f"{str(row[c]):>18}"
                for c in cols
            )
        )
    for c in cols[:-1] if _HAS_MEMOS else cols:
        print(f"median {c}: {statistics.median(r[c] for r in rows):.1f} us")
    if _HAS_MEMOS:
        print(f"all bitwise equal: {all(r['bitwise_equal'] for r in rows)}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(autotune=args.autotune, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
