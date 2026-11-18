"""SM107 MegaMoE expert-parallel sizes: host-side planning at EP 28 / 32 and the EP rule (CPU only).

Kimi K3 / K3.1 routes 896 experts: EP 1, 4, 8, 16, 28 and 32 (2 for the generic workspaces) are supported;
EP 12 / 20 / 24 (896 does not divide) and EP 14 (not whole 4-GPU nodes) are rejected with the rule.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import pathlib
import sys
import types

import pytest

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
from flashinfer.moe_ep.kernel_src.sm107.ep_sizes import (
    SM107_MAX_EP_WORLD_SIZE,
    sm107_ep_world_size_error,
    sm107_supported_ep_world_sizes,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    Sm107BlockScaledMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    block_scaled,
    knob_cache,
)

_E, _TOPK, _HIDDEN, _INTER = 896, 16, 3584, 3072
_KIMI_EP = (1, 4, 8, 16, 28, 32)
_REJECTED_EP = (3, 6, 12, 14, 20, 24, 36, 56, 64, 112, 128, 896)
_SOURCES = (
    pathlib.Path(block_scaled.__file__).resolve().parents[1] / "src" / "sources"
)


def _kimi_config(**kw):
    return Sm107BlockScaledMoeConfig(
        **(
            dict(
                num_total_experts=_E,
                max_tokens_per_rank=8192,
                num_topk=_TOPK,
                hidden=_HIDDEN,
                intermediate=_INTER,
                rank=0,
                world_size=32,
                quant_kind="nvfp4",
                mma_tiler_mnk=(256, 256, 256),
                cluster_shape_mn=(4, 1),
                fallback_cluster_shape_mn=(2, 1),
                schedule_policy=("phase_interleave", None),
                work_id_mode="atomic_counter",
            )
            | kw
        )
    )


def _class_constants(path: pathlib.Path, class_name: str) -> dict:
    """Integer ClassVar constants of ``class_name`` read from kernel source (no CuTe DSL import)."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            constants = {}
            for item in node.body:
                if not (isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.value):
                    continue
                # Integer literals and literal arithmetic (e.g. ``227 * 1024``) only.
                if all(
                    isinstance(n, (ast.Constant, ast.BinOp, ast.operator)) for n in ast.walk(item.value)
                ):
                    value = eval(compile(ast.Expression(item.value), str(path), "eval"), {"__builtins__": {}})
                    if type(value) is int:
                        constants[item.target.id] = value
            return constants
    raise AssertionError(f"{class_name} not found in {path}")


def _token_comm_constants() -> dict:
    path = _SOURCES / "communication" / "nvlink_domain" / "token_comm.py"
    return {
        "_MetadataPushRouter": _class_constants(path, "_MetadataPushRouter"),
        "TokenCommNonDeterministic": _class_constants(path, "TokenCommNonDeterministic"),
    }


@pytest.fixture
def router_gates(monkeypatch):
    """The two router classes ``variant_policy_key`` reads, from source when the CuTe DSL is absent."""
    if importlib.util.find_spec("cutlass") is not None:
        return
    constants = _token_comm_constants()
    module = types.ModuleType("sources.communication.nvlink_domain.token_comm")
    for name, values in constants.items():
        setattr(module, name, type(name, (), dict(values)))
    for parent in ("sources", "sources.communication", "sources.communication.nvlink_domain"):
        package = types.ModuleType(parent)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, parent, package)
    monkeypatch.setitem(sys.modules, module.__name__, module)


# --------------------------------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------------------------------


def test_supported_ep_sizes_for_896_experts():
    assert sm107_supported_ep_world_sizes(_E, single_node_sizes=(1,)) == _KIMI_EP
    assert sm107_supported_ep_world_sizes(_E) == (1, 2) + _KIMI_EP[1:]
    for ws in _KIMI_EP:
        assert sm107_ep_world_size_error(_E, ws, single_node_sizes=(1,)) is None


@pytest.mark.parametrize("ws", _REJECTED_EP + (0, -4, 2.0, True))
def test_rejected_ep_sizes_name_the_rule(ws):
    error = sm107_ep_world_size_error(_E, ws, single_node_sizes=(1,))
    assert error is not None
    if type(ws) is int and ws > 0:
        assert f"world_size={ws}" in error
        assert "supported for 896 experts: 1, 4, 8, 16, 28, 32" in error
        assert "divide num_experts" in error and "at most 32" in error and "multiple of 4" in error


def test_ep14_divides_896_but_is_not_whole_nodes():
    assert _E % 14 == 0
    assert sm107_ep_world_size_error(_E, 14) is not None


def test_max_ep_matches_the_decode_router_lane_limit():
    """The bound is the decode-landing router's one-lane-per-source-rank limit in the kernel source."""
    router = _token_comm_constants()["_MetadataPushRouter"]
    assert router["decode_landing_max_world_size"] == SM107_MAX_EP_WORLD_SIZE == 32
    # The push router's helper CTA holds a world_size x padded-experts int32 size matrix plus two vectors; at
    # the bound it fits the router's shared-memory budget with 896 experts.
    padded = -(-_E // 4) * 4
    helper_bytes = SM107_MAX_EP_WORLD_SIZE * padded * 4 + 2 * padded * 4 + 16 * 4
    assert helper_bytes <= router["router_smem_limit_bytes"]


# --------------------------------------------------------------------------------------------------------------
# Workspace configuration
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ws,per_rank", [(4, 224), (8, 112), (16, 56), (28, 32), (32, 28)])
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
def test_expert_split_and_weight_shapes(ws, per_rank, kind):
    tile_k = 256 if kind == "nvfp4" else 128
    for rank in (0, ws - 1):
        cfg = _kimi_config(world_size=ws, rank=rank, quant_kind=kind, mma_tiler_mnk=(256, 256, tile_k))
        assert cfg.experts_per_rank == per_rank
        fc1, fc2, fc1_sf, fc2_sf = block_scaled._expected_weight_shapes(cfg)
        assert fc1 == (per_rank, _HIDDEN // 2, 2 * _INTER)
        assert fc2 == (per_rank, _INTER // 2, _HIDDEN)
        assert fc1_sf > 0 and fc2_sf > 0
    with pytest.raises(ValueError, match="invalid rank/world_size"):
        _kimi_config(world_size=ws, rank=ws)


@pytest.mark.parametrize("ws", [12, 14, 20, 24, 64])
def test_workspace_config_rejects_unsupported_ep(ws):
    with pytest.raises(ValueError, match=f"EP world_size={ws} with 896 experts"):
        _kimi_config(world_size=ws)


def test_generic_workspaces_keep_two_rank_groups():
    cfg = Sm107BlockScaledMoeConfig(
        num_total_experts=8, max_tokens_per_rank=3, num_topk=2, hidden=128, intermediate=64, rank=1, world_size=2
    )
    assert cfg.experts_per_rank == 4
    with pytest.raises(ValueError, match="multiple of 4"):
        dataclasses.replace(cfg, num_total_experts=12, world_size=6, rank=0)


@pytest.mark.parametrize("kind,tile_k,min_rows", [("nvfp4", 256, 100), ("mxfp4_mxfp8", 128, 120)])
@pytest.mark.parametrize("ws", [8, 16, 28, 32])
def test_cache_miss_heuristic_n_tile_follows_rows_per_expert(ws, kind, tile_k, min_rows):
    """A knob-cache miss at a new EP resolves the built-in heuristic (no crash, no EP table)."""
    for tokens in (1, 4, 64, 128, 176, 256, 512, 1024, 2048):
        knobs = knob_cache.default_knobs(
            tokens, quant_kind=kind, world_size=ws, topk=_TOPK, num_experts=_E
        )
        rows = tokens * ws * _TOPK / _E
        expected = 256 if (rows > min_rows or tokens >= 2048) else 128
        assert knobs["mma_tiler_mnk"] == (256, expected, tile_k), (ws, tokens)


@pytest.mark.parametrize("ws", [28, 32])
def test_knob_cache_lookup_misses_cleanly_at_new_ep(ws, tmp_path, monkeypatch):
    """Entries recorded at EP 8 / 16 never serve EP 28 / 32 (world_size is a key field)."""
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    common = dict(dtype="nvfp4", hidden=_HIDDEN, intermediate=_INTER, num_experts=_E, topk=_TOPK, max_tokens=64)
    entries = [
        knob_cache.make_record({"mma_tiler_mnk": [256, 128, 256]}, world_size=other, **common)
        for other in (8, 16)
    ]
    assert knob_cache.lookup_knobs(entries=entries, world_size=ws, **common) is None
    assert knob_cache.lookup_knobs(entries=entries, world_size=16, **common) is not None
    knobs, source = knob_cache.resolve_knobs(world_size=ws, **common)
    assert source == "heuristic" and knobs["mma_tiler_mnk"][1] in (128, 256)


@pytest.mark.parametrize("ws", [8, 16, 28, 32])
def test_die_split_policy_scales_with_ep(ws, monkeypatch):
    """die_split up to 1536 routed rows per expert: T <= 1536 * 896 / (16 * ws) tokens per rank."""
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import die_local

    monkeypatch.setattr(die_local, "locality_domain_count", lambda device: 2)
    monkeypatch.setattr(block_scaled.torch.cuda, "current_device", lambda: 0)
    limit = 1536 * _E // (_TOPK * ws)
    assert block_scaled._die_split_policy(_kimi_config(world_size=ws, max_tokens_per_rank=limit))
    assert not block_scaled._die_split_policy(_kimi_config(world_size=ws, max_tokens_per_rank=limit + 1))


# --------------------------------------------------------------------------------------------------------------
# Batch-size variant classes and the EP plan agreement
# --------------------------------------------------------------------------------------------------------------


def _policy_capacity(cfg):
    def config_for_capacity(tokens):
        tile = knob_cache.default_knobs(
            tokens,
            quant_kind="nvfp4" if cfg.quant_kind == "nvfp4" else "mxfp4_mxfp8",
            world_size=cfg.world_size,
            topk=cfg.num_topk,
            num_experts=cfg.num_total_experts,
        )["mma_tiler_mnk"]
        return dataclasses.replace(cfg, max_tokens_per_rank=tokens, mma_tiler_mnk=tile)

    return config_for_capacity


@pytest.mark.parametrize("ws", [8, 16, 28, 32])
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp4_mxfp8"])
def test_variant_classes_at_each_ep(ws, kind, router_gates, monkeypatch):
    monkeypatch.setattr(block_scaled, "_die_split_policy", lambda c: block_scaled._DIE_SPLIT_MAX_ROWS_PER_EXPERT
                        >= c.max_tokens_per_rank * c.world_size * c.num_topk / c.num_total_experts)
    tile_k = 256 if kind == "nvfp4" else 128
    cfg = _kimi_config(world_size=ws, quant_kind=kind, mma_tiler_mnk=(256, 256, tile_k))
    capacity = _policy_capacity(cfg)
    classes = block_scaled.variant_classes(cfg, capacity)
    assert classes[0].lo == 1 and classes[-1].hi == 1024
    assert all(a.hi + 1 == b.lo for a, b in zip(classes, classes[1:]))
    assert [c.slot for c in classes] == list(range(1, len(classes) + 1))
    for c in classes:
        assert c.config.world_size == ws and c.config.experts_per_rank == _E // ws
        key = block_scaled.variant_policy_key(c.config)
        _, _, data_ctas, landing, _, die_split = key
        # Decode landing (one router lane per source rank) covers every variant up to 32 ranks.
        assert landing and data_ctas == 1
        assert die_split
        for tokens in {c.lo, (c.lo + c.hi) // 2, c.hi}:
            assert block_scaled.variant_policy_key(capacity(tokens)) == key
    assert len(classes) <= 16


@pytest.mark.parametrize("ws", [28, 32])
def test_workspace_fingerprint_names_the_ep_size(ws):
    cfg = _kimi_config(world_size=ws, max_tokens_per_rank=256)
    fingerprint = block_scaled._workspace_fingerprint(cfg, None)
    assert fingerprint == block_scaled._workspace_fingerprint(dataclasses.replace(cfg, rank=ws - 1), None)
    assert fingerprint != block_scaled._workspace_fingerprint(_kimi_config(world_size=16, max_tokens_per_rank=256), None)


# --------------------------------------------------------------------------------------------------------------
# Isolated Kimi K3 backend
# --------------------------------------------------------------------------------------------------------------


def _kimi_backend(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.kimi_k3.backend import KimiK3MegaKernelBackend
    from flashinfer.moe_ep.backends.mega.kernel.sm107.kimi_k3.config import Sm107_KimiK3_MegaMoeConfig

    monkeypatch.setattr(KimiK3MegaKernelBackend, "_ensure_ep_bootstrap", lambda self, bootstrap: None)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    return KimiK3MegaKernelBackend(Sm107_KimiK3_MegaMoeConfig(situ_beta=4.0, situ_linear_beta=25.0))


@pytest.mark.parametrize("ws", _KIMI_EP)
def test_kimi_backend_accepts_supported_ep(ws, monkeypatch):
    backend = _kimi_backend(monkeypatch)
    fleet = FleetParams(num_experts=_E, max_tokens_per_rank=256, token_hidden_size=_HIDDEN)
    backend.validate_init(BootstrapConfig(world_size=ws, rank=0, auto_bootstrap=False), fleet)


@pytest.mark.parametrize("ws", [2, 12, 14, 20, 24, 64])
def test_kimi_backend_rejects_unsupported_ep(ws, monkeypatch):
    backend = _kimi_backend(monkeypatch)
    fleet = FleetParams(num_experts=_E, max_tokens_per_rank=256, token_hidden_size=_HIDDEN)
    with pytest.raises(MoEEpConfigError, match=r"supported for 896 experts: 1, 4, 8, 16, 28, 32"):
        backend.validate_init(BootstrapConfig(world_size=ws, rank=0, auto_bootstrap=False), fleet)


# --------------------------------------------------------------------------------------------------------------
# Router geometry from the kernel classes (needs the CuTe DSL; runs in the GPU container, skipped on CPU)
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ws", [8, 16, 28, 32])
def test_router_geometry_and_pool_bound_at_each_ep(ws):
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import bootstrap_paths

    bootstrap_paths()
    from sources.api import ImplDesc, ProblemDesc
    from sources.communication.nvlink_domain.token_comm import _MetadataPushRouter

    for tokens in (1, 64, 1024, 8192):
        problem = ProblemDesc(
            dict(world_size=ws, expert_count=_E, topk=_TOPK, max_tokens_per_rank=tokens, apply_topk_at_fc1=False)
        )
        impl = ImplDesc(dict(token_padding_block=128, promised_launchable_sm_count=128))
        router = _MetadataPushRouter(problem, impl)
        assert router.experts_per_rank == _E // ws
        assert router.decode_landing == (tokens * _TOPK <= 16384)
        # Pool bound: room for every route the world can send this rank, plus at most one block per local expert.
        bound = router.worst_case_padded_tokens(128)
        routes = ws * tokens * min(_TOPK, _E // ws)
        assert bound % 128 == 0 and routes <= bound <= routes + 128 * (_E // ws)
