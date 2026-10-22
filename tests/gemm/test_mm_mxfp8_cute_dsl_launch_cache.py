"""Memoized launches of mm_mxfp8(backend="cute-dsl") and mxfp8_quantize(backend="cute-dsl").

A repeated call with the same signature must launch exactly what the full path
launches (bitwise-identical outputs), must skip the runner selection, and must
take the full path again after any change of the autotuner state.
"""

import pytest
import torch

import flashinfer.api_logging as api_logging
from flashinfer import autotune, mm_mxfp8, mxfp8_quantize
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm import gemm_base
from flashinfer.quantization.kernels.mxfp8_quantize import (
    _mxfp8_quantize_launch_plan,
)
from flashinfer.utils import get_compute_capability


def _skip_if_unsupported():
    major, minor = get_compute_capability(torch.device("cuda"))
    if not mm_mxfp8.is_backend_supported("cute-dsl", major * 10 + minor):
        pytest.skip("mm_mxfp8 cute-dsl backend is not supported on this GPU")


@pytest.fixture(autouse=True)
def _fresh_state():
    _skip_if_unsupported()
    tuner = AutoTuner.get()
    tuner.clear_cache()
    gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()
    yield
    tuner.clear_cache()
    gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()


def _operands(m, n, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g)
    a, a_sf = mxfp8_quantize(x, backend="cute-dsl")
    w_q, w_sf = mxfp8_quantize(w, backend="cute-dsl")
    return a, w_q.T, a_sf, w_sf


def _full_path(a, b, a_sf, b_sf, **kwargs):
    """mm_mxfp8 through the full path; leaves the launch memo empty."""
    gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()
    out = mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl", **kwargs)
    gemm_base._MM_MXFP8_LAUNCH_CACHE.clear()
    return out


def _count_selections(monkeypatch):
    calls = []
    choose_one = AutoTuner.choose_one

    def counting_choose_one(self, *args, **kwargs):
        calls.append(kwargs.get("custom_op", args[0] if args else None))
        return choose_one(self, *args, **kwargs)

    monkeypatch.setattr(AutoTuner, "choose_one", counting_choose_one)
    return calls


# Swap-AB and non-swap default tactics, split-K low M, out dtypes.
@pytest.mark.parametrize(
    "m,n,k", [(1, 256, 512), (16, 4096, 7168), (500, 2112, 7168), (2160, 7168, 1536)]
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("provide_out", [False, True])
@pytest.mark.parametrize("auto_tuning", [False, True])
def test_launch_cache_hit_is_bitwise_identical(
    m, n, k, out_dtype, provide_out, auto_tuning, monkeypatch
):
    a, b, a_sf, b_sf = _operands(m, n, k)
    if auto_tuning:
        with autotune(True):
            mm_mxfp8(a, b, a_sf, b_sf, out_dtype=out_dtype, backend="cute-dsl")

    def call():
        out = torch.empty(m, n, device="cuda", dtype=out_dtype) if provide_out else None
        return mm_mxfp8(
            a, b, a_sf, b_sf, out=out, out_dtype=out_dtype, backend="cute-dsl"
        )

    expected = _full_path(a, b, a_sf, b_sf, out_dtype=out_dtype)
    selections = _count_selections(monkeypatch)
    first = call()  # miss: full path, memoized
    assert len(selections) == 1
    assert len(gemm_base._MM_MXFP8_LAUNCH_CACHE) == 1
    hits = [call() for _ in range(3)]  # hits: no selection
    assert len(selections) == 1
    for out in [first, *hits]:
        assert out.dtype == out_dtype and out.shape == (m, n)
        assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_launch_cache_keys_on_signature(monkeypatch):
    a, b, a_sf, b_sf = _operands(256, 512, 1024)
    a2, b2, a2_sf, b2_sf = _operands(320, 512, 1024)
    expected = _full_path(a2, b2, a2_sf, b2_sf)
    selections = _count_selections(monkeypatch)
    out = torch.empty(256, 512, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):  # each distinct signature takes the full path once
        mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
        mm_mxfp8(a, b, a_sf, b_sf, out_dtype=torch.float16, backend="cute-dsl")
        mm_mxfp8(a, b, a_sf, b_sf, out=out, backend="cute-dsl")
        got = mm_mxfp8(a2, b2, a2_sf, b2_sf, backend="cute-dsl")  # another M
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    assert len(selections) == 4
    assert len(gemm_base._MM_MXFP8_LAUNCH_CACHE) == 4


def test_launch_cache_dropped_on_autotuner_state_change(monkeypatch):
    m, n, k = 64, 256, 2048
    a, b, a_sf, b_sf = _operands(m, n, k)
    selections = _count_selections(monkeypatch)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 1

    # Any write to the tuner state invalidates the memo.
    tuner = AutoTuner.get()
    tuner._file_configs["unrelated"] = ("Runner", 0)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 2
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 2

    tuner.clear_cache()
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 3


def test_launch_cache_follows_a_new_tuned_tactic(monkeypatch):
    """A winner recorded after a launch was memoized must be used."""
    m, n, k = 64, 256, 2048
    a, b, a_sf, b_sf = _operands(m, n, k)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")  # memoize the default tactic

    selected = []
    choose_one = AutoTuner.choose_one

    def recording_choose_one(self, *args, **kwargs):
        result = choose_one(self, *args, **kwargs)
        if not self.is_tuning_mode:
            selected.append(result[1])
        return result

    monkeypatch.setattr(AutoTuner, "choose_one", recording_choose_one)
    # Deterministic profiling: rank the tactics by a fixed key, not by time.
    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, tuning_config, **kw: float(
            len(repr(tactic))
        ),
    )
    with autotune(True):
        mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert AutoTuner.get().profiling_cache

    out = mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selected) == 1 and selected[0] != -1  # the tuned winner
    hit = mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selected) == 1  # memoized with the tuned winner
    expected = _full_path(a, b, a_sf, b_sf)
    assert len(selected) == 2 and selected[1] == selected[0]
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(hit.view(torch.int16), expected.view(torch.int16))


def test_launch_cache_not_used_while_tuning(monkeypatch):
    m, n, k = 128, 256, 1024
    a, b, a_sf, b_sf = _operands(m, n, k)
    with autotune(True):
        mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
        assert not gemm_base._MM_MXFP8_LAUNCH_CACHE
        mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
        assert not gemm_base._MM_MXFP8_LAUNCH_CACHE
    # A memo taken before tuning is not served while tuning.
    AutoTuner.get().clear_cache()
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert gemm_base._MM_MXFP8_LAUNCH_CACHE
    selections = _count_selections(monkeypatch)
    with autotune(True):
        mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 1


def test_launch_cache_not_used_with_api_logging(monkeypatch):
    a, b, a_sf, b_sf = _operands(128, 256, 1024)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert gemm_base._MM_MXFP8_LAUNCH_CACHE
    selections = _count_selections(monkeypatch)
    monkeypatch.setattr(api_logging, "_API_LOG_LEVEL", 1)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    assert len(selections) == 1
    monkeypatch.setattr(api_logging, "_API_LOG_LEVEL", 0)
    monkeypatch.setenv("FLASHINFER_TRACE_DUMP", "1")
    assert api_logging.api_call_hooks_enabled()
    assert (
        gemm_base._mm_mxfp8_cached_launch(a, b, a_sf, b_sf, backend="cute-dsl") is None
    )


def test_launch_cache_validation_still_runs_on_misses():
    a, b, a_sf, b_sf = _operands(128, 256, 1024)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    with pytest.raises(ValueError):
        mm_mxfp8(a, b, a_sf[:-1], b_sf, backend="cute-dsl")  # wrong scale length
    with pytest.raises(ValueError):
        mm_mxfp8(a, b, a_sf, b_sf, out_dtype=torch.float32, backend="cute-dsl")


def test_launch_cache_under_cuda_graph():
    m, n, k = 256, 512, 1024
    a, b, a_sf, b_sf = _operands(m, n, k)
    expected = _full_path(a, b, a_sf, b_sf)
    mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")  # memoize eagerly
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = mm_mxfp8(a, b, a_sf, b_sf, backend="cute-dsl")
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("m,k", [(1, 64), (500, 7168), (2160, 1536), (8640, 768)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("swizzled", [True, False])
def test_quantize_plan_memo_is_bitwise_identical(m, k, dtype, swizzled):
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    _mxfp8_quantize_launch_plan.cache_clear()
    q_full, sf_full = mxfp8_quantize(
        x, is_sf_swizzled_layout=swizzled, backend="cute-dsl"
    )
    for _ in range(2):
        info = _mxfp8_quantize_launch_plan.cache_info()
        q, sf = mxfp8_quantize(x, is_sf_swizzled_layout=swizzled, backend="cute-dsl")
        assert _mxfp8_quantize_launch_plan.cache_info().hits == info.hits + 1
        assert torch.equal(q.view(torch.uint8), q_full.view(torch.uint8))
        assert torch.equal(sf, sf_full)


def test_quantize_plan_memo_with_padded_and_misaligned_inputs():
    m, k = 300, 7168
    pool = torch.randn(m * k + 8, device="cuda", dtype=torch.bfloat16)
    aligned = pool[: m * k].view(m, k)
    misaligned = pool[4 : 4 + m * k].view(m, k)  # 8-byte offset
    q_ref, sf_ref = mxfp8_quantize(misaligned.clone(), backend="cute-dsl")
    mxfp8_quantize(aligned, backend="cute-dsl")  # memoize the plan
    q, sf = mxfp8_quantize(misaligned, backend="cute-dsl")
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(sf, sf_ref)

    # alignment > K pads the columns; the memo keys on alignment.
    x = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        q, sf = mxfp8_quantize(x, alignment=128, backend="cute-dsl")
        assert q.shape == (64, 128)
        assert torch.equal(
            q[:, 96:].view(torch.uint8), torch.zeros_like(q[:, 96:].view(torch.uint8))
        )
