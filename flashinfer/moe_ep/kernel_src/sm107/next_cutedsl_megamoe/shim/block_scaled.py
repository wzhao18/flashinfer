# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM107 (Rubin) block-scaled swap-AB inference mega-MoE frontend.

Wraps the vendored generic inference and GenPhase kernels behind the standard
two-entry-point mega contract:

- :func:`get_symm_buffer_for_sm107_block_scaled_mega_moe` — workspace allocator
- :func:`sm107_block_scaled_mega_moe` — fused dispatch + FC1 + activation +
  FC2 + combine compute entry, with SwiGLU or SiTU

The kernel is generic over the drop's ``QuantKind``; this shim wires up the
``nvfp4``, ``mxfp4``, ``mxfp8_e4m3``, ``mxfp8_e5m2``, and ``mxfp4_mxfp8`` kinds.
The mixed kind uses packed FP4 weights and E4M3 activations, both with
E8M0 scales per 32 values.

All ``sources`` / ``cutlass`` imports are function-local so importing this
module stays CPU-safe (the package ``__init__`` re-exports from here).

Input activations, block scales, routing scores, and the shared workspace
live on the symmetric heap, as does the output for in-kernel reduction.
Routing indices are local int32 tensors aligned to 16 bytes. Workspaces are
128-byte aligned with their leading bytes zeroed before the first launch.
"""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple

import torch

from flashinfer.moe_ep._compat import deprecated_aliases

from . import comm
from ...ep_sizes import sm107_ep_world_size_error
from .dependencies import require_sm107_dsl
from .kernel_helpers import Mxfp8BlockSize, Nvfp4BlockSize, swizzled_flat_sf_size
from .tuning_state import AutoTuningState

if TYPE_CHECKING:
    from flashinfer.moe_ep.tensors import MoEEpTensors

Sm107QuantKind = Literal["nvfp4", "mxfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"]
Sm107TokenBackMode = Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
Sm107WorkIdMode = Literal["grid_stride", "atomic_counter"]
Sm107ScheduleMode = Literal["grouped", "phase_interleave"]

# (act data dtype, act SF dtype, sf_vec_size, 2x-mode instruction K) per kind.
# The mxf8f6f4 UMMA path has instruction K 32 (64 in 2x mode); the fp4 pair
# kinds have 64 (128 in 2x mode) — see quant_def.QuantKind.instruction_k.
_KIND_TABLE: dict = {
    "nvfp4": (None, torch.float8_e4m3fn, Nvfp4BlockSize, 128),
    "mxfp4": (None, torch.float8_e8m0fnu, Mxfp8BlockSize, 128),
    "mxfp8_e4m3": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp8_e5m2": (torch.float8_e5m2, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
    "mxfp4_mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu, Mxfp8BlockSize, 64),
}

TransformedBlockScaledWeights = Tuple[torch.Tensor, torch.Tensor]

_SCALAR_NAMES = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")
# die_split (two-ended expert claim over die-local weights) up to this many expected rows per local expert.
_DIE_SPLIT_MAX_ROWS_PER_EXPERT = 1536

# Batch-size variants (automatic per-shape policy, no flag). Every capacity-derived compile-time choice of the
# kernel (S1 decode landing and the token_in dynamic claim, router data-CTA geometry, the N tile, die_split, the
# tail-reset extents) follows the WORKSPACE capacity, while a server sized for prefill (vLLM decode:
# max_num_batched_tokens 8192 tokens/rank) runs its CUDA-graph steps at a small padded batch. An inference workspace
# therefore also owns one kernel per policy class of batch sizes 1.._VARIANT_MAX_TOKENS: batch sizes whose
# compile-time choices are identical (variant_policy_key) share one kernel, compiled at the class's largest size;
# the live row count is a runtime argument of every kernel (router loads, staging tail). All variants share the
# workspace's allocations: each owns a private reset-state slot, and their data regions start at one common
# address with the post-kernel combine plane placed first at the workspace's own size (see _VariantPlan).
# Measured EP8 NVFP4 (capacity 8192 / capacity = T): T64 189.2 / 154.2 us, T256 193.4 / 167.8, T512 204.3 / 180.4,
# T1024 229.0 / 226.3, T2048 372.4 / 372.1: the capacity costs nothing from 2048 tokens on.
_VARIANT_MAX_TOKENS = 1024


def _variant_eligible(cfg: "Sm107BlockScaledMoeConfig") -> bool:
    """Variants exist for the inference kernel with separate top-k reduction.

    GenPhase and batch-invariant workspaces keep one configuration (their arithmetic receipt covers it); the
    in-kernel reduction peer-writes a symmetric output that variants would have to share at one layout.
    """
    return (
        cfg.kernel_variant == "inference"
        and not cfg.batch_invariant
        and not cfg.reduce_topk_in_kernel
    )


def _die_split_policy(cfg: "Sm107BlockScaledMoeConfig") -> bool:
    """die_split choice of :meth:`Sm107BlockScaledSymmBuffer._build_kernel` (one source of truth)."""
    if cfg.kernel_variant == "genphase" or cfg.schedule_policy[0] != "phase_interleave":
        return False
    from .die_local import locality_domain_count

    rows_per_expert = cfg.max_tokens_per_rank * cfg.world_size * cfg.num_topk / cfg.num_total_experts
    return (
        locality_domain_count(torch.cuda.current_device()) >= 2
        and rows_per_expert <= _DIE_SPLIT_MAX_ROWS_PER_EXPERT
    )


def variant_policy_key(cfg: "Sm107BlockScaledMoeConfig") -> tuple:
    """Compile-time choices of the kernel for ``cfg`` other than its capacity.

    Two capacities with equal keys compile the same code paths; they differ only in region extents and loop
    bounds over the capacity, which a live row count below the capacity skips. Mirrors the trace-time gates
    of the vendored router / token_in (checked against the built kernel in :func:`_check_policy_key`).
    """
    from sources.communication.nvlink_domain.token_comm import (
        TokenCommNonDeterministic,
        _MetadataPushRouter,
    )

    routes = cfg.padded_tokens_per_rank * cfg.num_topk
    cta_capacity = min(16384, 1 << (max(routes, 2048) - 1).bit_length())
    data_ctas = -(-routes // cta_capacity)
    landing = (
        routes <= _MetadataPushRouter.decode_landing_max_routes
        and cfg.world_size <= _MetadataPushRouter.decode_landing_max_world_size
        and data_ctas == 1
    )
    dynamic_claim = routes <= TokenCommNonDeterministic.token_in_dynamic_claim_max_routes
    return (
        dataclasses.replace(cfg, max_tokens_per_rank=1),
        cta_capacity,
        data_ctas,
        landing,
        dynamic_claim,
        _die_split_policy(cfg),
    )


def _check_policy_key(key: tuple, kernel) -> None:
    _, cta_capacity, data_ctas, landing, dynamic_claim, die_split = key
    router = kernel.token_comm._router
    built = (
        router.router_tokens_per_cta,
        router.router_data_cta_count,
        router.decode_landing,
        kernel.token_comm.token_in_dynamic_claim,
        bool(getattr(kernel.scheduler, "die_split", False)),
    )
    if built != (cta_capacity, data_ctas, landing, dynamic_claim, die_split):
        raise RuntimeError(f"variant policy key {key[1:]} does not match the built kernel {built}")


@dataclasses.dataclass(frozen=True)
class _VariantClass:
    """Batch sizes ``lo..hi`` share one kernel compiled at capacity ``hi`` (``config``)."""

    lo: int
    hi: int
    config: "Sm107BlockScaledMoeConfig"
    slot: int


@dataclasses.dataclass
class _VariantPlan:
    """One allocation per workspace space shared by the capacity kernel (slot 0) and every variant.

    Variant ``v`` binds its workspaces at ``base + v * slot_bytes``: its reset prefix (tail-reset copies,
    zero-on-first-allocate barrier and grid-sync state) lies in its own slot, so variants never disturb each
    other's cross-call state, and its data regions start at ``base + slots * slot_bytes`` like everyone's. The
    shared combine plane, which the separate top-k reduction reads AFTER the kernel (when peers may already run
    the next call and write their early regions), is placed first in every kernel at the capacity kernel's size,
    so it never overlaps another kernel's peer-written regions; all other data is read before the kernel's tail
    NVLink barrier and rewritten every call.
    """

    classes: Tuple[_VariantClass, ...]
    slot_bytes: Tuple[int, int]  # (local, shared)
    leading: Tuple[Tuple[str, int], ...]
    total_bytes: Tuple[int, int]
    kernels: tuple = ()  # built kernel per slot (0 = the capacity kernel)

    @property
    def slots(self) -> int:
        return len(self.classes) + 1

    def layout(self, slot: int) -> tuple:
        data_start = tuple((self.slots - slot) * b for b in self.slot_bytes)
        return (data_start, self.leading)


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def variant_classes(cfg, config_for_capacity) -> Tuple[_VariantClass, ...]:
    """Policy classes of batch sizes ``1..min(capacity - 1, _VARIANT_MAX_TOKENS)`` of a workspace.

    ``config_for_capacity(T)`` is the configuration the workspace would have at capacity ``T``. Consecutive
    batch sizes with equal :func:`variant_policy_key` and the same power-of-two capacity bucket form one class,
    compiled once at its largest size. The bucket bounds what a live batch pays for its kernel's capacity
    (tail-reset and worst-case extents): measured EP8 NVFP4, T2..T128 on one capacity-128 kernel ran
    +1.4..+3.2% over a kernel at capacity T; a kernel at most 2x the batch recovers that.

    A batch size whose own configuration is not variant-eligible (e.g. a tuned knob-cache entry selecting the
    in-kernel top-k reduction, whose symmetric output variants cannot share) belongs to no class: it runs on
    the capacity kernel (``variant_for`` returns the workspace for sizes between classes), which serves every
    batch size up to its capacity, so the result is exactly what the workspace computes without variants.
    """
    if not _variant_eligible(cfg):
        return ()
    top = min(cfg.max_tokens_per_rank - 1, _VARIANT_MAX_TOKENS)
    same_problem = ("num_total_experts", "num_topk", "hidden", "intermediate", "rank", "world_size", "quant_kind")
    spans: list = []  # [lo, hi, key, config]; a None entry ends a class at a non-variant batch size
    for tokens in range(1, top + 1):
        vcfg = config_for_capacity(tokens)
        if vcfg.max_tokens_per_rank != tokens or any(
            getattr(vcfg, f) != getattr(cfg, f) for f in same_problem
        ):
            raise RuntimeError(f"configuration for {tokens} tokens is not a capacity variant of {cfg}")
        if not _variant_eligible(vcfg):
            if spans and spans[-1] is not None:
                spans.append(None)
            continue
        key = (variant_policy_key(vcfg), 1 << (tokens - 1).bit_length())
        if spans and spans[-1] is not None and spans[-1][2] == key:
            spans[-1][1] = tokens
            spans[-1][3] = vcfg
        else:
            spans.append([tokens, tokens, key, vcfg])
    return tuple(
        _VariantClass(lo=lo, hi=hi, config=vcfg, slot=index + 1)
        for index, (lo, hi, _key, vcfg) in enumerate(s for s in spans if s is not None)
    )


def _plan_variants(cfg, config_for_capacity) -> Optional[_VariantPlan]:
    """Variant classes of a workspace and their shared layout (None: the workspace has no variants)."""
    classes = variant_classes(cfg, config_for_capacity)
    if not classes:
        return None
    build = Sm107BlockScaledSymmBuffer._build_kernel
    plain = [build(cfg)] + [build(c.config) for c in classes]
    for c, kernel in zip(classes, plain[1:]):
        _check_policy_key(variant_policy_key(c.config), kernel)
    state = [
        max(k._device_workspace.zero_on_allocate_bytes(space) for k in plain)
        for space in ("local", "shared")
    ]
    slot_bytes = tuple(_round_up(max(b, 1), 128) for b in state)
    token_comm = plain[0].token_comm
    workspace = plain[0]._device_workspace
    leading = tuple(
        (name, int(workspace.nbytes(name)))
        for name in (
            token_comm.pre_reduced_activation_region,
            token_comm.pre_reduced_activation_sf_region,
        )
        if name in workspace._region_by_name
    )
    plan = _VariantPlan(classes=classes, slot_bytes=slot_bytes, leading=leading, total_bytes=(0, 0))
    totals = [0, 0]
    kernels = []
    for slot, vcfg in enumerate([cfg] + [c.config for c in classes]):
        kernel = build(vcfg, data_layout=plan.layout(slot))
        kernels.append(kernel)
        for i, space in enumerate(("local", "shared")):
            totals[i] = max(totals[i], slot * slot_bytes[i] + int(kernel._device_workspace.total_bytes(space)))
    plan.total_bytes = tuple(totals)
    plan.kernels = tuple(kernels)
    return plan


def _config_digest(cfg: "Sm107BlockScaledMoeConfig") -> str:
    """Rank-independent digest of a configuration (every field but ``rank``)."""
    import hashlib

    return hashlib.sha256(repr(dataclasses.replace(cfg, rank=0)).encode()).hexdigest()[:16]


def _workspace_fingerprint(cfg: "Sm107BlockScaledMoeConfig", plan: Optional[_VariantPlan]) -> dict:
    """What every EP rank must resolve identically before the symmetric allocation.

    ``knobs=None`` and ``"cache"`` resolve the capacity configuration and every variant class from each rank's
    own read of the knob cache; the symmetric offsets of every kernel's reset slot and data regions follow from
    it, so ranks that read different caches would peer-write at offsets the receiving kernel does not use.
    """
    if plan is None:
        return dict(capacity_config=_config_digest(cfg), variants=None)
    return dict(
        capacity_config=_config_digest(cfg),
        variants=tuple((c.lo, c.hi, c.slot, _config_digest(c.config)) for c in plan.classes),
        slot_bytes=tuple(plan.slot_bytes),
        leading=tuple(plan.leading),
        total_bytes=tuple(plan.total_bytes),
    )


def _staging_view(name: str) -> property:
    """Workspace tensor of the selected live bucket (the workspace's own when none is selected)."""
    own = "_own_" + name

    def fget(self):
        active = self.__dict__.get("_active")
        return getattr(active, name) if active is not None else self.__dict__.get(own)

    def fset(self, value) -> None:
        self.__dict__[own] = value

    return property(fget, fset, doc=f"``{name}`` of the selected live bucket.")


def _require_sm107() -> None:
    cc = torch.cuda.get_device_capability()
    if cc != (10, 7):
        raise RuntimeError(
            f"the SM107 block-scaled mega kernel requires compute capability "
            f"(10, 7) (Rubin); this device reports {cc}."
        )


def _configure_dsl_arch() -> None:
    require_sm107_dsl()


def _fp4_storage_dtype() -> torch.dtype:
    return getattr(torch, "float4_e2m1fn_x2", torch.uint8)


def _to_cute(
    tensor: torch.Tensor, assumed_align: int = 16, dynamic_leading: bool = True
):
    """torch -> cute tensor via dlpack; leading dim marked dynamic by default."""
    import cutlass.torch as cutlass_torch

    cute_tensor = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    if not dynamic_leading:
        return cute_tensor
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 128):
    """Raw cute ``Uint8`` gmem pointer at ``tensor``'s base (opaque workspaces)."""
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import AddressSpace

    return cute.runtime.make_ptr(
        cutlass.Uint8, tensor.data_ptr(), AddressSpace.gmem, assumed_align=assumed_align
    )


def _cu_stream(stream: torch.cuda.Stream):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(stream.cuda_stream)


_MAX_ACTIVE_CLUSTERS_CACHE: dict = {}


def _max_active_clusters(cluster_size: int) -> int:
    """Occupancy probe, cached per device and cluster size."""
    key = (torch.cuda.current_device(), cluster_size)
    cached = _MAX_ACTIVE_CLUSTERS_CACHE.get(key)
    if cached is None:
        import cutlass.utils as cutlass_utils

        cached = int(cutlass_utils.HardwareInfo().get_max_active_clusters(cluster_size))
        _MAX_ACTIVE_CLUSTERS_CACHE[key] = cached
    return cached


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
@dataclasses.dataclass(frozen=True)
class Sm107BlockScaledMoeConfig:
    """Validated construction parameters for Rubin inference and GenPhase."""

    num_total_experts: int
    max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate: int  # width after activation; the FC1 GEMM N is 2*intermediate
    rank: int
    world_size: int
    quant_kind: Sm107QuantKind = "mxfp8_e4m3"
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None  # None -> (256, 128, 2*ik)
    cluster_shape_mn: Tuple[int, int] = (2, 1)
    # Mixed-CGA launch: alongside the preferred clusters, fill leftover SMs with
    # smaller fallback clusters (upstream commit a5b4d33). None = uniform launch.
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None
    # ("grouped" | "phase_interleave", optional hint); typed loosely so backend
    # configs can carry it as a plain (str, int|None) tuple.
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None)
    work_id_mode: Sm107WorkIdMode = "grid_stride"
    fc2_use_bulk: bool = False
    fc2_tma_stages: Optional[int] = None
    epi_flag_batches: Tuple[int, int] = (4, 2)
    token_in_flag_batch: int = 1
    token_padding_block: int = 64
    sf_padding_block: int = 128
    gate_up_clamp: Optional[float] = None
    reduce_topk_in_kernel: bool = False
    token_back_mode: Sm107TokenBackMode = "epi_warps"
    apply_routing_weights_before_fc2: bool = True
    max_sm_count: Optional[int] = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: Optional[float] = None
    situ_linear_beta: Optional[float] = None
    swiglu_alpha: Optional[float] = dataclasses.field(default=None, kw_only=True)
    swiglu_beta: Optional[float] = dataclasses.field(default=None, kw_only=True)

    kernel_variant: Literal["inference", "genphase"] = "inference"
    combine_dtype: Literal["bf16", "nvfp4", "mxfp8"] = "bf16"
    batch_invariant: bool = dataclasses.field(default=False, kw_only=True)
    input_norm_const: float = dataclasses.field(default=1.0, kw_only=True)
    has_fc1_alpha: bool = dataclasses.field(default=True, kw_only=True)
    has_fc2_alpha: bool = dataclasses.field(default=True, kw_only=True)
    has_fc1_norm_const: bool = dataclasses.field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if self.quant_kind not in _KIND_TABLE:
            raise ValueError(f"unsupported quant_kind {self.quant_kind!r}.")
        if self.kernel_variant not in ("inference", "genphase"):
            raise ValueError(f"unsupported kernel_variant {self.kernel_variant!r}.")
        if self.combine_dtype not in ("bf16", "nvfp4", "mxfp8"):
            raise ValueError(f"unsupported combine_dtype {self.combine_dtype!r}.")
        if self.kernel_variant == "genphase" and self.combine_dtype != "bf16":
            raise ValueError("GenPhase requires BF16 combine.")
        if self.combine_dtype != "bf16" and self.reduce_topk_in_kernel:
            raise ValueError("Quantized combine requires separate top-k reduction.")
        if (
            self.combine_dtype == "nvfp4"
            and self.fc2_use_bulk
            and self.token_back_mode == "epi_warps"
        ):
            raise ValueError(
                "NVFP4 combine bulk stores require a token-back data path."
            )
        if (
            not self.batch_invariant
            and type(self).__name__ != "Sm100BlockScaledMoeConfig"
            and (self.swiglu_alpha is not None or self.swiglu_beta is not None)
        ):
            raise ValueError("custom activation constants require batch_invariant=True")
        if self.activation not in ("swiglu", "situ"):
            raise ValueError("activation must be 'swiglu' or 'situ'.")
        if self.activation == "situ" and not self.batch_invariant:
            for name in ("situ_beta", "situ_linear_beta"):
                value = getattr(self, name)
                if value is None or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"SiTU requires positive, finite {name}.")
            if self.gate_up_clamp is not None:
                raise ValueError("SiTU does not support gate_up_clamp.")
        elif self.activation != "situ" and (
            self.situ_beta is not None or self.situ_linear_beta is not None
        ):
            raise ValueError("SiTU beta parameters require activation='situ'.")
        for name in (
            "num_total_experts",
            "max_tokens_per_rank",
            "num_topk",
            "hidden",
            "intermediate",
            "world_size",
            "token_padding_block",
            "sf_padding_block",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive Python integer.")
        if type(self.rank) is not int or not (0 <= self.rank < self.world_size):
            raise ValueError(f"invalid rank/world_size {self.rank}/{self.world_size}.")
        for name in (
            "fc2_use_bulk",
            "reduce_topk_in_kernel",
            "apply_routing_weights_before_fc2",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a Python bool.")
        if self.num_total_experts > 16384:
            raise ValueError("SM107 routing supports at most 16384 global experts.")
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk must not exceed num_total_experts.")
        if self.padded_tokens_per_rank * self.num_topk > 2097152:
            raise ValueError("SM107 routing supports at most 2097152 routes per rank.")
        ep_error = sm107_ep_world_size_error(self.num_total_experts, self.world_size)
        if ep_error is not None:
            raise ValueError(ep_error)
        vec = self.sf_vec_size
        if self.hidden % (4 * vec) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a multiple of {4 * vec} "
                f"for {self.quant_kind}."
            )
        if self.intermediate % (2 * vec) != 0 or self.intermediate % 32 != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a multiple of "
                f"{max(2 * vec, 32)} (gate/up interleave + SF blocks) for "
                f"{self.quant_kind}."
            )
        if self.quant_kind in ("mxfp4", "mxfp4_mxfp8") and self.intermediate % 128:
            raise ValueError(
                f"{self.quant_kind} intermediate must be a multiple of 128."
            )
        tiler = self.resolved_mma_tiler_mnk
        instruction_k = self.instruction_k
        if len(tiler) != 3:
            raise ValueError("mma_tiler_mnk must be a 3-tuple.")
        if any(type(dim) is not int or dim <= 0 for dim in tiler):
            raise ValueError(
                "mma_tiler_mnk dimensions must be positive Python integers."
            )
        if tiler[0] not in (128, 256):
            raise ValueError(f"Rubin instruction M must be 128 or 256, got {tiler[0]}.")
        if tiler[1] not in (64, 128, 256):
            raise ValueError(
                f"Rubin instruction N must be 64, 128, or 256, got {tiler[1]}."
            )
        if tiler[2] not in (2 * instruction_k, 4 * instruction_k):
            raise ValueError(
                f"Rubin {self.quant_kind} tile K must be {2 * instruction_k} or "
                f"{4 * instruction_k}, got {tiler[2]}."
            )
        if (
            self.quant_kind == "nvfp4"
            and tiler[1] == 256
            and tiler[2] == 4 * instruction_k
        ):
            raise ValueError("nvfp4 does not support (N=256, K=4x-instruction) tiles.")
        if len(self.cluster_shape_mn) != 2:
            raise ValueError("cluster_shape_mn must be a 2-tuple.")
        cm, cn = self.cluster_shape_mn
        if type(cm) is not int or cm <= 0 or cm > 16 or cm & (cm - 1):
            raise ValueError("cluster M must be a power of two no greater than 16.")
        if type(cn) is not int or cn != 1:
            raise ValueError("the swap-AB FC12 path requires cluster N=1.")
        if tiler[0] == 256 and self.cluster_shape_mn[0] % 2 != 0:
            raise ValueError("instruction M 256 requires an even cluster M.")
        if self.fallback_cluster_shape_mn is not None:
            fallback = tuple(self.fallback_cluster_shape_mn)
            if fallback == tuple(self.cluster_shape_mn):
                # Kernel-side NonClcMixedCgaConfig collapses this to a uniform
                # launch; normalize here so downstream logic sees one spelling.
                object.__setattr__(self, "fallback_cluster_shape_mn", None)
            else:
                if len(fallback) != 2:
                    raise ValueError("fallback_cluster_shape_mn must be a 2-tuple.")
                if any(type(dim) is not int or dim <= 0 for dim in fallback):
                    raise ValueError(
                        "fallback_cluster_shape_mn dimensions must be positive."
                    )
                if fallback[1] != 1:
                    raise ValueError(
                        "the swap-AB FC12 fallback path requires cluster N=1."
                    )
                if any(
                    preferred % fb != 0
                    for preferred, fb in zip(
                        self.cluster_shape_mn, fallback, strict=True
                    )
                ):
                    raise ValueError(
                        "every preferred cluster dimension must be divisible by "
                        "its fallback dimension."
                    )
                if tiler[0] == 256 and fallback[0] % 2 != 0:
                    raise ValueError(
                        "instruction M 256 requires an even fallback cluster M."
                    )
                if self.max_sm_count is not None:
                    raise ValueError(
                        "max_sm_count is not supported together with "
                        "fallback_cluster_shape_mn (mixed-CGA occupancy is "
                        "resolved from the hardware probe)."
                    )
        if self.token_padding_block % 64 != 0:
            raise ValueError("token_padding_block must be a multiple of 64.")
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")
        if tiler[1] % self.token_padding_block != 0:
            raise ValueError(
                f"mma tiler N ({tiler[1]}) must be a whole number of token "
                f"padding blocks ({self.token_padding_block})."
            )
        if len(self.schedule_policy) != 2:
            raise ValueError("schedule_policy requires (mode, hint).")
        mode, hint = self.schedule_policy
        if mode not in ("grouped", "phase_interleave"):
            raise ValueError(f"unknown schedule mode {mode!r}.")
        if hint is not None and (type(hint) is not int or hint <= 0):
            raise ValueError("schedule hint must be a positive Python integer or None.")
        if self.work_id_mode not in ("grid_stride", "atomic_counter"):
            raise ValueError(f"unknown work_id_mode {self.work_id_mode!r}.")
        if self.token_back_mode not in (
            "epi_warps",
            "standalone_warps",
            "reuse_dispatch_warps",
        ):
            raise ValueError(f"unknown token_back_mode {self.token_back_mode!r}.")
        if mode == "phase_interleave" and self.work_id_mode != "atomic_counter":
            raise ValueError(
                "schedule_policy 'phase_interleave' requires "
                "work_id_mode='atomic_counter'."
            )
        if len(self.epi_flag_batches) != 2:
            raise ValueError("epi_flag_batches requires (FC1, FC2) values.")
        if any(
            type(batch) is not int or not 1 <= batch <= 4
            for batch in self.epi_flag_batches
        ):
            raise ValueError("Rubin epi_flag_batches values must be in [1, 4].")
        if (
            type(self.token_in_flag_batch) is not int
            or not 1 <= self.token_in_flag_batch <= 32
        ):
            raise ValueError("token_in_flag_batch must be in [1, 32].")
        if self.max_sm_count is not None and (
            type(self.max_sm_count) is not int or self.max_sm_count < cm
        ):
            raise ValueError("max_sm_count must accommodate at least one cluster.")
        if (
            not self.batch_invariant
            and self.gate_up_clamp is not None
            and (not math.isfinite(self.gate_up_clamp) or self.gate_up_clamp < 0)
        ):
            raise ValueError("gate_up_clamp must be finite and nonnegative.")
        if self.fc2_tma_stages is not None and not (
            type(self.fc2_tma_stages) is int
            and 1 <= self.fc2_tma_stages <= tiler[1] // 64
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {tiler[1] // 64}] for tiler N "
                f"{tiler[1]}."
            )
        if self.reduce_topk_in_kernel and not self.apply_routing_weights_before_fc2:
            raise ValueError(
                "reduce_topk_in_kernel requires "
                "apply_routing_weights_before_fc2=True (the in-kernel reduce "
                "red-adds already-weighted terms)."
            )

        if self.kernel_variant == "genphase":
            if self.padded_tokens_per_rank > 1024:
                raise ValueError("GenPhase supports at most 1024 tokens per rank.")
            n64_decode = (
                self.cluster_shape_mn in ((2, 1), (4, 1))
                and tiler == (256, 64, 128)
                and (
                    self.quant_kind == "mxfp8_e4m3"
                    or (
                        self.quant_kind == "mxfp4_mxfp8"
                        and self.cluster_shape_mn == (4, 1)
                    )
                )
            )
            if (
                self.cluster_shape_mn != (4, 1) and not n64_decode
            ) or self.fallback_cluster_shape_mn is not None:
                raise ValueError(
                    "GenPhase requires uniform cluster (4, 1), or an admitted N64 decode geometry at (2, 1) or (4, 1)."
                )
            if not self.fc2_use_bulk:
                raise ValueError("GenPhase requires fc2_use_bulk=True.")
            if tiler[1] not in (128, 256) and not n64_decode:
                raise ValueError(
                    "GenPhase requires tile N 128 or 256, or an admitted N64 decode geometry."
                )
            if n64_decode and self.world_size > 1 and mode == "phase_interleave":
                # EP1 phase-interleave coverage does not establish multirank
                # liveness: an EP2 dense/replay case stalls with this geometry.
                # Reject before allocation or compilation; autotune applies
                # this same config validation through is_valid_sm107.
                raise ValueError(
                    "GenPhase N64 with world_size > 1 requires grouped scheduling; "
                    "multirank phase_interleave is not qualified."
                )
            if tiler[2] != 2 * instruction_k:
                raise ValueError(
                    f"GenPhase {self.quant_kind} requires tile K {2 * instruction_k} "
                    "so each gathered row occupies one 128-byte swizzle atom."
                )
            if self.sf_padding_block != 128 or 128 % self.token_padding_block:
                raise ValueError(
                    "GenPhase requires SF padding 128 and token padding 64 or 128."
                )
            if self.token_back_mode != "epi_warps" or self.token_in_flag_batch != 1:
                raise ValueError(
                    "GenPhase does not use token-back or token-in flag knobs."
                )

        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
            validate_block_scaled_batch_invariant,
        )

        if type(self.batch_invariant) is not bool:
            raise ValueError("batch_invariant must be a bool.")
        if self.batch_invariant:
            # NVFP4 binds three expert scalars; MX modes have no such operands.
            expected_scalars = self.quant_kind == "nvfp4"
            for name in ("has_fc1_alpha", "has_fc2_alpha", "has_fc1_norm_const"):
                if getattr(self, name) is not expected_scalars:
                    raise ValueError(
                        f"batch_invariant workspace requires {name}={expected_scalars}."
                    )
        validate_block_scaled_batch_invariant(self, arch="sm_107")

    @property
    def experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def padded_tokens_per_rank(self) -> int:
        """Physical capacity keeps the router's four-int vector loads in bounds."""
        multiple = 4 // math.gcd(self.num_topk, 4)
        return (self.max_tokens_per_rank + multiple - 1) // multiple * multiple

    @property
    def sf_vec_size(self) -> int:
        return _KIND_TABLE[self.quant_kind][2]

    @property
    def instruction_k(self) -> int:
        return _KIND_TABLE[self.quant_kind][3]

    @property
    def resolved_mma_tiler_mnk(self) -> Tuple[int, int, int]:
        if self.mma_tiler_mnk is not None:
            return self.mma_tiler_mnk
        return (256, 128, 2 * self.instruction_k)

    @property
    def torch_act_data_dtype(self) -> torch.dtype:
        dtype = _KIND_TABLE[self.quant_kind][0]
        return _fp4_storage_dtype() if dtype is None else dtype

    @property
    def torch_weight_data_dtype(self) -> torch.dtype:
        if self.quant_kind in ("nvfp4", "mxfp4", "mxfp4_mxfp8"):
            return _fp4_storage_dtype()
        return self.torch_act_data_dtype

    @property
    def torch_act_sf_dtype(self) -> torch.dtype:
        return _KIND_TABLE[self.quant_kind][1]


class Sm107BlockScaledSymmBuffer:
    """Session workspace for the SM107 block-scaled inference mega kernel.

    Owns the staging tensors the backend fills (``x``, ``x_sf``, ``topk_idx``,
    ``topk_weights``), the kernel + device workspaces, and the compile cache.
    Create via :func:`get_symm_buffer_for_sm107_block_scaled_mega_moe`.

    Batch-size variants (``variants=True``, inference kernels with separate
    top-k reduction): besides the capacity kernel, the workspace owns one
    kernel per policy class of batch sizes up to ``_VARIANT_MAX_TOKENS``
    (:func:`variant_policy_key`; ``config_for_capacity`` gives the knobs of a
    capacity, e.g. the backend's per-shape policy). Variants share every
    allocation of the workspace (:class:`_VariantPlan`) and its staging rows.
    :meth:`prepare_batch_sizes` compiles the variants of a list of batch
    sizes (e.g. the CUDA-graph capture sizes) ahead of time; nothing compiles
    later on their behalf. :meth:`select_live_tokens` (host, before staging)
    picks the kernel of one step from the EP-uniform step maximum; the staging
    tensors, ``output_activation``, ``prepare`` and ``launch`` then follow the
    selection. Every EP rank must prepare and select alike.
    """

    x = _staging_view("x")
    x_sf = _staging_view("x_sf")
    topk_idx = _staging_view("topk_idx")
    topk_weights = _staging_view("topk_weights")
    output_activation = _staging_view("output_activation")
    # Class defaults keep partially constructed workspaces (tests, object.__new__) on the full-capacity kernel.
    _active: Optional["Sm107BlockScaledSymmBuffer"] = None
    _variant_plan: Optional[_VariantPlan] = None
    _variants: Tuple["Sm107BlockScaledSymmBuffer", ...] = ()
    _variant_root: Optional["Sm107BlockScaledSymmBuffer"] = None
    _variant_class: Optional[_VariantClass] = None

    def __init__(
        self,
        config: Sm107BlockScaledMoeConfig,
        *,
        process_group=None,
        variants: bool = True,
        config_for_capacity=None,
    ) -> None:
        """Allocate session buffers, including independent NVFP4 default scalars."""
        self._active = None
        self._variants = ()
        self.config = config
        self._process_group = process_group
        self._config_for_capacity = (
            config_for_capacity
            if config_for_capacity is not None
            else (lambda tokens: dataclasses.replace(config, max_tokens_per_rank=tokens))
        )
        self._want_variants = bool(variants)
        prepared = comm.prepare_local(
            self._prepare_session_local,
            config.world_size,
            process_group,
            phase="session local allocation",
            agree=lambda p: _workspace_fingerprint(config, p["variant_plan"]),
        )
        execution = prepared.pop("execution")
        self.device = prepared.pop("device")
        self.kernel = prepared.pop("kernel")
        self.sf_cols = prepared.pop("sf_cols")
        self.topk_idx = prepared.pop("topk_idx")
        self._variant_plan = prepared.pop("variant_plan")
        variant_kernels = prepared.pop("variant_kernels")
        self.__dict__.update(prepared)
        cfg = config
        tokens = cfg.padded_tokens_per_rank

        # Staging tensors. Activation / SF / scores are pulled by peer dispatch
        # warps, so they live on the symmetric heap; the routing indices are a
        # local (16B-aligned) int32 tensor read by this rank's router only.
        if cfg.quant_kind in ("nvfp4", "mxfp4"):
            self.x = comm.sym_zeros((tokens, cfg.hidden // 2), cfg.torch_act_data_dtype)
        else:
            self.x = comm.sym_zeros((tokens, cfg.hidden), cfg.torch_act_data_dtype)
        self.x_sf = comm.sym_zeros((tokens, self.sf_cols), cfg.torch_act_sf_dtype)
        self.topk_weights = comm.sym_zeros((tokens, cfg.num_topk), torch.float32)

        self._allocate_execution_state(prepared=execution)
        self._staged_tokens: Optional[int] = None
        self._destroyed = False
        self._tuning_state = AutoTuningState()
        self._tuning_inputs: tuple[MoEEpTensors, bool] | None = None
        self._routing_checks: dict[tuple, Any] = {}
        # Missing scalar bindings use the workspace's own tensors.
        self._bound_scalars: dict = {}

        self._validate_batch_invariant_scalars()
        if self._variant_plan is not None:
            self._variants = tuple(
                self._make_variant(c, kernel)
                for c, kernel in zip(self._variant_plan.classes, variant_kernels)
            )

    def _make_variant(self, vclass: _VariantClass, kernel) -> "Sm107BlockScaledSymmBuffer":
        """A kernel of this workspace's variant plan: no allocation, views of this workspace only."""
        variant = object.__new__(Sm107BlockScaledSymmBuffer)
        cfg = vclass.config
        own = self.__dict__
        rows = cfg.padded_tokens_per_rank
        variant._active = None
        variant._variants = ()
        variant._variant_root = self
        variant._variant_class = vclass
        variant.config = cfg
        variant._process_group = self._process_group
        variant.device = self.device
        variant.kernel = kernel
        variant.sf_cols = int(kernel.token_comm.activation_sf_hidden_padded)
        if variant.sf_cols != self.sf_cols:
            raise RuntimeError("variant scale layout differs from its workspace")
        # Leading rows of the workspace's staging and output tensors: peers pull a token row at the same symmetric
        # address whichever kernel runs, and the routing tail past the live rows is never read.
        variant.x = own["_own_x"][:rows]
        variant.x_sf = own["_own_x_sf"][:rows]
        variant.topk_weights = own["_own_topk_weights"][:rows]
        variant.topk_idx = own["_own_topk_idx"][:rows]
        variant.output_activation = own["_own_output_activation"][:rows]
        local_slot, shared_slot = self._variant_plan.slot_bytes
        variant.local_workspace = self.local_workspace[vclass.slot * local_slot :]
        variant.shared_workspace = self.shared_workspace[vclass.slot * shared_slot :]
        for name in (*_SCALAR_NAMES, "_default_scalar"):
            setattr(variant, name, getattr(self, name, None))
        variant._staged_tokens = None
        variant._destroyed = False
        variant._tuning_state = AutoTuningState()
        variant._tuning_inputs = None
        variant._routing_checks = {}
        variant._bound_scalars = {}
        variant._bind_execution_state(peers_of=self)
        return variant

    @property
    def variant_capacities(self) -> Tuple[int, ...]:
        """Capacity (largest batch size) of every variant class, ascending."""
        return tuple(v.config.max_tokens_per_rank for v in self._variants)

    @property
    def compiled_variant_capacities(self) -> Tuple[int, ...]:
        return tuple(v.config.max_tokens_per_rank for v in self._variants if v._compiled is not None)

    def variant_for(self, tokens: int) -> "Sm107BlockScaledSymmBuffer":
        """The kernel whose policy class holds ``tokens`` (the capacity kernel past the classes)."""
        for variant in self._variants:
            if tokens <= variant.config.max_tokens_per_rank:
                return variant if tokens >= variant._variant_class.lo else self
        return self

    def prepare_batch_sizes(
        self,
        batch_sizes,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
    ) -> dict:
        """Compile the kernel of every batch size (and the capacity kernel) ahead of time.

        Call outside graph capture with the same sizes on every EP rank, e.g. the CUDA-graph capture sizes;
        batch sizes that share a policy class share one compiled kernel. Returns ``{batch size: capacity of
        its kernel}``. Afterwards a step selects only among compiled kernels (no compile on the request path).
        """
        comm.ensure_not_capturing("SM107 batch-size variant compile")
        selected = {}
        for tokens in sorted({int(t) for t in batch_sizes}):
            if not 0 < tokens <= self.config.max_tokens_per_rank:
                raise ValueError(
                    f"batch size {tokens} outside (0, {self.config.max_tokens_per_rank}]"
                )
            kernel = self.variant_for(tokens)
            if kernel._compiled is None:
                kernel._prepare_self(transformed_l1, transformed_l2, live_tokens=tokens)
            selected[tokens] = kernel.config.max_tokens_per_rank
        if self._compiled is None:
            self._prepare_self(transformed_l1, transformed_l2)
        return selected

    def select_live_tokens(
        self, ep_max_tokens: Optional[int], num_tokens: int
    ) -> Optional[int]:
        """Pick the kernel for the next step; call before staging its inputs (host only, no device sync).

        ``ep_max_tokens`` is the step's largest live token count over the EP
        ranks (the same on every rank, ``MoEEpTensors.runtime_max_tokens_per_rank``);
        the compiled variant whose class holds it runs. ``None`` selects the
        full-capacity kernel, except on a single rank, where ``num_tokens`` is
        that maximum. Returns the selected capacity.
        """
        if ep_max_tokens is None and self.config.world_size == 1:
            ep_max_tokens = num_tokens
        if ep_max_tokens is not None and not (
            num_tokens <= ep_max_tokens <= self.config.max_tokens_per_rank
        ):
            raise ValueError(
                f"runtime_max_tokens_per_rank={ep_max_tokens} must cover this "
                f"rank's {num_tokens} tokens within capacity "
                f"{self.config.max_tokens_per_rank}"
            )
        self._active = None
        if ep_max_tokens is not None and self._variants:
            kernel = self.variant_for(ep_max_tokens)
            if kernel is not self and kernel._compiled is not None:
                self._active = kernel
        return (self._active or self).config.max_tokens_per_rank

    def _prepare_session_local(self) -> dict:
        """Finish every local allocation before entering the symmetric heap."""
        comm.ensure_not_capturing("SM107 mega workspace allocation")
        _require_sm107()
        _configure_dsl_arch()
        cfg = self.config
        device = torch.device("cuda", torch.cuda.current_device())
        plan = (
            _plan_variants(cfg, self._config_for_capacity)
            if self._want_variants
            else None
        )
        kernel = self._build_kernel(cfg) if plan is None else plan.kernels[0]
        prepared = dict(
            device=device,
            kernel=kernel,
            sf_cols=int(
                kernel.token_comm.plain_activation_sf_row_elements
                if cfg.kernel_variant == "genphase"
                else kernel.token_comm.activation_sf_hidden_padded
            ),
            topk_idx=torch.full(
                (cfg.padded_tokens_per_rank, cfg.num_topk),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            execution=self._prepare_execution_local(cfg, kernel, plan),
            variant_plan=plan,
            variant_kernels=[] if plan is None else list(plan.kernels[1:]),
        )
        for name in (*_SCALAR_NAMES, "_default_scalar"):
            prepared[name] = (
                torch.ones(cfg.experts_per_rank, dtype=torch.float32, device=device)
                if cfg.quant_kind == "nvfp4"
                else None
            )
            if (
                name in _SCALAR_NAMES
                and cfg.batch_invariant
                and getattr(cfg, f"has_{name}") is not (prepared[name] is not None)
            ):
                raise ValueError(
                    f"batch_invariant scale presence does not match {name}."
                )
        if prepared["topk_idx"].data_ptr() % 16:
            raise RuntimeError("routing index tensor must be 16-byte aligned.")
        return prepared

    def _validate_batch_invariant_scalars(self) -> None:
        if not self.config.batch_invariant:
            return
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            if getattr(self.config, f"has_{name}") is not (
                getattr(self, name) is not None
            ):
                raise ValueError(
                    f"batch_invariant scale presence does not match {name}."
                )

    @property
    def batch_invariant_fingerprint(self) -> Optional[str]:
        if not self.config.batch_invariant:
            return None
        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
            block_scaled_batch_invariant_fingerprint,
        )

        self._validate_batch_invariant_scalars()
        return block_scaled_batch_invariant_fingerprint(self.config, arch="sm_107")

    def _prepare_execution_local(self, cfg, kernel, plan=None) -> dict:
        workspace = kernel._device_workspace
        local_bytes = int(
            workspace.total_bytes("local") if plan is None else plan.total_bytes[0]
        )
        local = torch.zeros((max(local_bytes, 1),), dtype=torch.uint8, device="cuda")
        if local.data_ptr() % 128:
            raise RuntimeError("SM107 mega workspaces must be 128-byte aligned.")
        return dict(
            local_workspace=local,
            output_activation=(
                None
                if cfg.reduce_topk_in_kernel
                else torch.zeros(
                    (cfg.padded_tokens_per_rank, cfg.hidden),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
            ),
        )

    def _allocate_execution_state(self, *, prepared=None) -> None:
        """Allocate reduction buffers and scratch, then reset cached bindings.

        Peer-written buffers use symmetric allocations; separate-reduce output
        stays local and may be replaced by a caller output at launch time.
        """
        cfg, kernel = self.config, self.kernel
        tokens = cfg.padded_tokens_per_rank
        if prepared is None:
            prepared = comm.prepare_local(
                lambda: self._prepare_execution_local(cfg, kernel),
                cfg.world_size,
                getattr(self, "_process_group", None),
                phase="kernel local allocation",
            )
        # In-kernel reduce: the combine peer-writes (red.add) each topk term
        # straight into the output, so it must live on the SYMMETRIC heap and
        # start zeroed; the separate-reduce path only sees local stores.
        if cfg.reduce_topk_in_kernel:
            self.output_activation = comm.sym_zeros(
                (tokens, cfg.hidden), torch.bfloat16
            )
        else:
            self.output_activation = prepared["output_activation"]

        workspace = kernel._device_workspace
        plan = getattr(self, "_variant_plan", None)
        shared_bytes = int(
            workspace.total_bytes("shared") if plan is None else plan.total_bytes[1]
        )
        # Allocated fully zeroed, which covers require_zero_workspace_leading_bytes.
        self.local_workspace = prepared["local_workspace"]
        self.shared_workspace = comm.sym_zeros((max(shared_bytes, 1),), torch.uint8)
        comm.prepare_local(
            self._bind_execution_state,
            cfg.world_size,
            getattr(self, "_process_group", None),
            phase="symmetric workspace binding",
        )

    def _bind_execution_state(self, peers_of=None) -> None:
        cfg, kernel = self.config, self.kernel
        tokens = cfg.padded_tokens_per_rank
        workspace = kernel._device_workspace
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        if not cfg.reduce_topk_in_kernel:
            region = (
                kernel.pre_reduced_activation_region
                if cfg.kernel_variant == "genphase"
                else kernel.token_comm.pre_reduced_activation_region
            )
            start = workspace.offset(region)
            end = start + workspace.nbytes(region)
            # Byte views also cover packed FP4/FP8 payloads, which do not
            # support Torch masked_fill directly.
            self._pre_reduced_activation = self.shared_workspace[start:end].view(
                tokens, cfg.num_topk, -1
            )
            if cfg.combine_dtype != "bf16":
                region = kernel.token_comm.pre_reduced_activation_sf_region
                start = workspace.offset(region)
                end = start + workspace.nbytes(region)
                self._pre_reduced_activation_sf = self.shared_workspace[start:end].view(
                    tokens, cfg.num_topk, -1
                )
        if (
            self.local_workspace.data_ptr() % 128 != 0
            or self.shared_workspace.data_ptr() % 128 != 0
        ):
            raise RuntimeError("SM107 mega workspaces must be 128-byte aligned.")

        if peers_of is None:
            base, peer_offsets = comm.compute_peer_offsets(
                self.shared_workspace, cfg.world_size
            )
        else:
            # One heap-wide peer delta (comm.compute_peer_offsets) also serves views of the same heap.
            base, peer_offsets = peers_of._symmetric_base, peers_of._peer_offsets
        self._symmetric_base = base
        self._peer_offsets = peer_offsets

        # Cached launch bindings name the previous workspaces and peer offsets.
        self.__dict__.pop("_kwargs_cache", None)
        self._compiled: Optional[Any] = None
        self._launch_key: Optional[tuple] = None
        self._launch_kwargs: Optional[dict] = None
        self._launch_weights: Optional[
            Tuple[TransformedBlockScaledWeights, TransformedBlockScaledWeights]
        ] = None
        self._launch_output: Optional[torch.Tensor] = None

    @staticmethod
    def _build_kernel(cfg: Sm107BlockScaledMoeConfig, data_layout: Optional[tuple] = None):
        import cutlass
        from cutlass.cute.nvgpu import OperandMajorMode
        from sources import RubinInferenceGenphaseMegaMoE, RubinInferenceMegaMoE
        from sources.api import ImplDesc, ProblemDesc
        from sources.quant_def import CombineFormat

        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.block_scaled_batch_invariant import (
            validate_block_scaled_batch_invariant,
        )

        validate_block_scaled_batch_invariant(cfg, arch="sm_107")
        tiler = cfg.resolved_mma_tiler_mnk
        cluster_size = cfg.cluster_shape_mn[0] * cfg.cluster_shape_mn[1]
        preferred_count: Optional[int] = None
        fallback_count: Optional[int] = None
        if cfg.fallback_cluster_shape_mn is None:
            launch_clusters = _max_active_clusters(cluster_size)
            if cfg.max_sm_count is not None:
                requested = cfg.max_sm_count // cluster_size
                if requested <= 0:
                    raise ValueError(
                        "max_sm_count must cover at least one full cluster."
                    )
                launch_clusters = min(launch_clusters, requested)
        else:
            # Mirror the drop's launch_cluster_configuration()/max_active_clusters()
            # mixed recipe (tester/host_utils.py): pack preferred clusters at their
            # occupancy limit, then convert the leftover fallback CTA capacity into
            # whole preferred-sized groups of fallback clusters.
            fallback_size = (
                cfg.fallback_cluster_shape_mn[0] * cfg.fallback_cluster_shape_mn[1]
            )
            if cluster_size % fallback_size != 0:
                raise ValueError(
                    "preferred cluster size must be a multiple of the fallback "
                    "cluster size."
                )
            preferred_count = _max_active_clusters(cluster_size)
            fallback_occupancy = _max_active_clusters(fallback_size)
            fallback_capacity = fallback_occupancy * fallback_size
            preferred_capacity = preferred_count * cluster_size
            if fallback_capacity < preferred_capacity:
                raise ValueError(
                    "fallback cluster CTA capacity must not be smaller than "
                    "preferred cluster CTA capacity."
                )
            split_factor = cluster_size // fallback_size
            remaining = (fallback_capacity - preferred_capacity) // fallback_size
            fallback_count = remaining // split_factor * split_factor
            launch_clusters = preferred_count + fallback_count // split_factor

        from flashinfer.moe_ep.backends.mega.kernel.sm100.common.activation_batch_invariant import (
            batch_invariant_activation,
        )

        activation_variant = (
            batch_invariant_activation(cfg, family="next107", nondefault_only=True)
            if cfg.batch_invariant
            else None
        )
        problem_desc = ProblemDesc(
            {
                "expert_count": cfg.num_total_experts,
                "batch_invariant": cfg.batch_invariant,
                **(
                    {"batch_invariant_activation": activation_variant}
                    if activation_variant is not None
                    else {}
                ),
                "intermediate_gateup_size": 2 * cfg.intermediate,
                "hidden_size": cfg.hidden,
                "quant_kind": cfg.quant_kind,
                "a_major_mode": OperandMajorMode.K,
                "b_major_mode": OperandMajorMode.K,
                "combine_format": CombineFormat.parse(
                    {
                        "bf16": "bf16",
                        "nvfp4": "16e2m1xbf16",
                        "mxfp8": "32e4m3xe8m0",
                    }[cfg.combine_dtype]
                ),
                "gate_up_clamp": None
                if activation_variant is not None
                else cfg.gate_up_clamp,
                "situ_beta": None if activation_variant is not None else cfg.situ_beta,
                "situ_linear_beta": None
                if activation_variant is not None
                else cfg.situ_linear_beta,
                "world_size": cfg.world_size,
                "topk": cfg.num_topk,
                "topk_index_dtype": cutlass.Int32,
                "max_tokens_per_rank": cfg.padded_tokens_per_rank,
                "apply_topk_at_fc1": cfg.apply_routing_weights_before_fc2,
            }
        )
        impl_fields = {
            "mma_instruction_mnk": (tiler[0], tiler[1], cfg.instruction_k),
            "mma_tiler_mnk": tiler,
            "mma_k_mode": "2x",
            "cluster_shape_mn": tuple(cfg.cluster_shape_mn),
            "use_2cta_instrs": tiler[0] == 256,
            "schedule_policy": tuple(cfg.schedule_policy),
            "token_padding_block": cfg.token_padding_block,
            "sf_padding_block": cfg.sf_padding_block,
            "work_id_mode": cfg.work_id_mode,
            "fc2_use_bulk": cfg.fc2_use_bulk,
            "epi_flag_batches": tuple(cfg.epi_flag_batches),
            "launch_cluster_count": launch_clusters,
            "token_in_flag_batch": cfg.token_in_flag_batch,
            "token_back_mode": cfg.token_back_mode,
            "reduce_topk_in_kernel": cfg.reduce_topk_in_kernel,
        }
        if cfg.fc2_tma_stages is not None:
            impl_fields["fc2_tma_stages"] = cfg.fc2_tma_stages
        if data_layout is not None:
            impl_fields["workspace_data_layout"] = data_layout
        # Multi-die GPU: the weight preprocessing places the lower half of the local experts on locality
        # domain 0 and the upper half on domain 1 (die_local.py); the scheduler then lets each die claim
        # from its own end of the expert-ordered work streams.
        # Per-shape: at large expected rows per local expert (deep prefill) the work streams are long and
        # the weight reads amortize; there the two-ended claim measured slower (EP16 T8192, 2340 rows: +3.9%),
        # so it is used only up to _DIE_SPLIT_MAX_ROWS_PER_EXPERT (_die_split_policy).
        if _die_split_policy(cfg):
            impl_fields["die_split"] = True
        if cfg.fallback_cluster_shape_mn is not None:
            impl_fields["fallback_cluster_shape_mn"] = tuple(
                cfg.fallback_cluster_shape_mn
            )
            impl_fields["preferred_cluster_count"] = preferred_count
            impl_fields["fallback_cluster_count"] = fallback_count
        kernel_cls = (
            RubinInferenceGenphaseMegaMoE
            if cfg.kernel_variant == "genphase"
            else RubinInferenceMegaMoE
        )
        return kernel_cls(problem_desc, ImplDesc(impl_fields))

    def note_staged_tokens(self, num_tokens: int) -> None:
        self._staged_tokens = int(num_tokens)

    def staged_tokens(self) -> Optional[int]:
        return self._staged_tokens

    def bind_scalars(self, values: dict) -> None:
        """Bind per-call NVFP4 scalar tensors (``fc1_alpha`` / ``fc2_alpha`` /
        ``fc1_norm_const``: contiguous fp32 ``[experts_per_rank]`` CUDA tensors) as
        the launch arguments, instead of copying them into the workspace tensors
        every call. ``None`` (or a missing name) selects the workspace tensor.
        The tensors must stay alive while launches (or captured graphs) use them."""
        bound = {}
        for name in _SCALAR_NAMES:
            value = values.get(name)
            if value is not None:
                bound[name] = value
        self._bound_scalars = bound

    def _scalar_tensor(self, name: str):
        if self._variant_root is not None:
            # Variants run with their workspace's bound or stored scalars.
            return self._variant_root._scalar_tensor(name)
        bound = getattr(self, "_bound_scalars", None) or {}
        value = bound.get(name)
        return value if value is not None else getattr(self, name)

    def _peer_mapper(self):
        from sources.communication.nvlink_domain.symmetric_buffer import (
            SymmetricBufferHost,
        )

        return SymmetricBufferHost(
            base_address=self._symmetric_base,
            offsets=tuple(self._peer_offsets),
            rank=self.config.rank,
            max_ranks=self.config.world_size,
        )

    def _runtime_kwargs(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        output: Optional[torch.Tensor] = None,
        live_tokens: Optional[int] = None,
    ) -> dict:
        """Bind weights, workspaces, scalars, and output on the current stream.

        Output defaults to the workspace buffer. Its dynamic row extent lets
        one compiled kernel serve caller outputs with different live lengths.
        The routing tensor is bound at its ``live_tokens`` rows (default: the
        capacity): its runtime row extent is the router's live row count.
        """
        stream = torch.cuda.current_stream()
        own = self.__dict__  # this kernel's own staging views, whatever variant is selected
        topk_idx = own["_own_topk_idx"]
        if live_tokens is not None:
            topk_idx = topk_idx[: max(int(live_tokens), 1)]
        return {
            "activation": _to_cute(own["_own_x"]),
            "activation_sf": _to_cute(own["_own_x_sf"]),
            "topk_indices": _to_cute(topk_idx),
            "topk_scores": _to_cute(own["_own_topk_weights"], assumed_align=4),
            "fc1_weight": _to_cute(transformed_l1[0]),
            "fc1_weight_sf": _to_cute(transformed_l1[1]),
            "fc2_weight": _to_cute(transformed_l2[0]),
            "fc2_weight_sf": _to_cute(transformed_l2[1]),
            "output_activation": self._output_kwarg(output),
            "local_workspace": _to_cute_ptr(self.local_workspace),
            "shared_workspace": _to_cute_ptr(self.shared_workspace),
            "peer_rank_ptr_mapper_host": self._peer_mapper(),
            "stream": _cu_stream(stream),
            **{
                name: _to_cute(self._scalar_tensor(name), assumed_align=4)
                for name in _SCALAR_NAMES
                if self.config.quant_kind == "nvfp4"
            },
            **(
                {"sm_locality_domain": _to_cute(self._sm_locality_domain())}
                if getattr(self.kernel.scheduler, "die_split", False)
                else {}
            ),
        }

    def _output_kwarg(self, output: Optional[torch.Tensor]):
        """Output binding (default: the workspace buffer)."""
        if output is None:
            output = self.__dict__["_own_output_activation"]
        # Separate reduction can write only the live rows, directly into the
        # caller's output. Keep the row extent dynamic so warmed kernels also
        # accept another batch size during CUDA graph capture.
        output_cute = _to_cute(output, dynamic_leading=False)
        output_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        return output_cute

    def _sm_locality_domain(self) -> torch.Tensor:
        """Die of every SM (``die_local.sm_locality_domain_map``), for the die_split scheduler."""
        from .die_local import SM_LOCALITY_MAP_SIZE, sm_locality_domain_map

        domains = sm_locality_domain_map(self.device.index)
        if domains.numel() != SM_LOCALITY_MAP_SIZE:
            raise RuntimeError("unexpected SM locality map size")
        return domains

    def _live_tokens(self) -> Optional[int]:
        root = self._variant_root or self
        return getattr(root, "_staged_tokens", None)

    def prepare(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        *,
        output: Optional[torch.Tensor] = None,
        live_tokens: Optional[int] = None,
    ) -> None:
        """Compile and bind the selected kernel without launching it.

        ``live_tokens`` defaults to the staged token count. Variants compile
        only through :meth:`prepare_batch_sizes`.
        """
        (self._active or self)._prepare_self(
            transformed_l1,
            transformed_l2,
            output=output,
            live_tokens=self._live_tokens() if live_tokens is None else live_tokens,
        )

    def _prepare_self(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        *,
        output: Optional[torch.Tensor] = None,
        live_tokens: Optional[int] = None,
    ) -> None:
        import cutlass.cute as cute

        if self._destroyed or (self._variant_root is not None and self._variant_root._destroyed):
            raise RuntimeError("cannot launch a destroyed SM107 workspace.")
        if output is not None and (
            self.config.reduce_topk_in_kernel
            or output.ndim != 2
            or not (0 < output.shape[0] <= self.config.max_tokens_per_rank)
            or output.shape[1] != self.config.hidden
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError(
                "Direct output requires separate reduction and an aligned, "
                "contiguous BF16 matrix with live rows on the workspace device."
            )
        if (
            self.device.type == "cuda"
            and self.device.index != torch.cuda.current_device()
        ):
            raise RuntimeError(
                "SM107 workspace must launch on the CUDA device that owns it."
            )
        if live_tokens is not None:
            live_tokens = int(live_tokens)
            if not 0 <= live_tokens <= self.config.max_tokens_per_rank:
                raise ValueError(
                    f"{live_tokens} live tokens exceed the kernel capacity "
                    f"{self.config.max_tokens_per_rank}"
                )
        self._validate_batch_invariant_scalars()
        binding = (
            self.batch_invariant_fingerprint,
            transformed_l1[0].data_ptr(),
            transformed_l1[1].data_ptr(),
            transformed_l2[0].data_ptr(),
            transformed_l2[1].data_ptr(),
            # Bound per-call scalars: a new tensor re-binds the launch arguments
            # (host-only, no recompile).
            *(
                self._scalar_tensor(name).data_ptr()
                for name in _SCALAR_NAMES
                if self.config.quant_kind == "nvfp4"
            ),
            torch.cuda.current_stream().cuda_stream,
            live_tokens,
        )
        key = binding + (
            None
            if output is None
            else (output.data_ptr(), output.shape, output.stride()),
        )
        if self._compiled is None or self._launch_key != key:
            if self._compiled is None:
                comm.ensure_not_capturing("SM107 mega kernel compile")
            # DLPack metadata binding and stream conversion are host-only. The
            # bindings of the workspace's own buffers are cached per (live row
            # count, layer weights, scalars, stream); a caller output is bound
            # per key change and never cached, so at most the latest one
            # (_launch_output) stays referenced. Cleared whenever the bound
            # workspaces are released or rebound (_bind_execution_state).
            cache = self.__dict__.setdefault("_kwargs_cache", {})
            entry = cache.get(binding)
            if entry is None:
                entry = (
                    self._runtime_kwargs(transformed_l1, transformed_l2, None, live_tokens),
                    (transformed_l1, transformed_l2),
                )
                if len(cache) >= 64:
                    cache.clear()
                cache[binding] = entry
            kwargs = entry[0]
            if output is not None:
                kwargs = {**kwargs, "output_activation": self._output_kwarg(output)}
            if self._compiled is None:
                self._compiled = cute.compile(self.kernel, **kwargs)
            self._launch_key = key
            self._launch_kwargs = kwargs
            # Keep pointers alive until the cached bindings are replaced.
            self._launch_weights = (transformed_l1, transformed_l2)
            self._launch_output = output

    def launch(
        self,
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
        *,
        output: Optional[torch.Tensor] = None,
        live_tokens: Optional[int] = None,
    ) -> None:
        """Compile on first use, then launch the fused mega kernel."""
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self._tuning_state.graph_captured = True
        self.prepare(transformed_l1, transformed_l2, output=output, live_tokens=live_tokens)
        (self._active or self)._launch_prepared()

    def _launch_prepared(self) -> None:
        if self.config.reduce_topk_in_kernel:
            # red.add accumulation base: zero the output every launch.
            self.output_activation.zero_()
        elif not getattr(self.kernel, "masks_invalid_combine_slots", False):
            # Invalid routes do not write a combine slot. Clear those slots on
            # every launch so a masked route cannot reuse an earlier result.
            invalid = (self.topk_idx < 0) | (
                self.topk_idx >= self.config.num_total_experts
            )
            self._pre_reduced_activation.masked_fill_(invalid.unsqueeze(-1), 0)
            if self.config.combine_dtype != "bf16":
                # Clear BF16 amax for NVFP4; E8M0 byte 127 is unit scale for MXFP8.
                sf_byte = 0 if self.config.combine_dtype == "nvfp4" else 127
                self._pre_reduced_activation_sf.masked_fill_(
                    invalid.unsqueeze(-1), sf_byte
                )
        self._compiled(**self._launch_kwargs)

    def _release_execution_state(self) -> None:
        """Drop cached tensor references and free owned symmetric buffers."""
        self.__dict__.pop("_kwargs_cache", None)  # binds the buffers freed below
        self._compiled = None
        self._launch_kwargs = None
        self._launch_weights = None
        self._launch_output = None
        self._launch_key = None
        self._pre_reduced_activation = None
        self._pre_reduced_activation_sf = None
        comm.free_sym_tensor(getattr(self, "shared_workspace", None))
        if self.config.reduce_topk_in_kernel:
            comm.free_sym_tensor(self.output_activation)
        self.output_activation = None
        self.shared_workspace = None
        self.local_workspace = None

    def install_tuned_knobs(
        self,
        knobs: dict[str, Any],
        transformed_l1: TransformedBlockScaledWeights,
        transformed_l2: TransformedBlockScaledWeights,
    ) -> None:
        """Replace output and scratch before capture, retaining staged inputs."""
        import dataclasses

        self._active = None

        def prepare_replacement():
            comm.ensure_not_capturing("SM107 tuning winner installation")
            if self._destroyed or self._tuning_state.graph_captured:
                raise RuntimeError(
                    "cannot tune a destroyed or captured SM107 workspace"
                )
            config = dataclasses.replace(self.config, **knobs)
            if config == self.config:
                return None
            kernel = self._build_kernel(config)
            sf_cols = int(
                kernel.token_comm.plain_activation_sf_row_elements
                if config.kernel_variant == "genphase"
                else kernel.token_comm.activation_sf_hidden_padded
            )
            if sf_cols != self.sf_cols:
                raise ValueError("tuning cannot change the staged scale layout")
            local = self._prepare_execution_local(config, kernel)
            torch.cuda.synchronize()
            return config, kernel, local

        prepared = comm.prepare_local(
            prepare_replacement,
            self.config.world_size,
            self._process_group,
            phase="winner local allocation",
        )
        if prepared is not None:
            config, kernel, local = prepared
            # A tuned winner replaces the layout: tuned (automatic) workspaces own no variants.
            self._variants = ()
            self._variant_plan = None
            self._release_execution_state()
            self.config, self.kernel = config, kernel
            self._allocate_execution_state(prepared=local)
        comm.prepare_local(
            lambda: self.prepare(transformed_l1, transformed_l2),
            self.config.world_size,
            self._process_group,
            phase="winner compilation",
        )

    def destroy(self) -> None:
        """Drain outstanding work and collectively release the session buffers."""
        if self._destroyed:
            return
        comm.ensure_not_capturing("SM107 mega workspace free")
        # Symmetric free is collective and does not wait for in-flight kernels.
        torch.cuda.synchronize()
        self._active = None
        for variant in self._variants:  # views of this workspace: no allocation of their own
            variant._compiled = variant._launch_kwargs = None
            variant.__dict__.pop("_kwargs_cache", None)
            variant._destroyed = True
        self._variants = ()
        self.__dict__.pop("_kwargs_cache", None)
        self._release_execution_state()
        self._routing_checks.clear()
        self._tuning_inputs = None
        self._bound_scalars = {}
        for name in ("x", "x_sf", "topk_weights"):
            comm.free_sym_tensor(getattr(self, name, None))
            setattr(self, name, None)
        if self.config.quant_kind == "nvfp4":
            from ....sm100.cutedsl_megamoe.shim.quant_stage import forget_staged_tokens

            forget_staged_tokens(self.topk_idx)
        self.topk_idx = None
        self.fc1_alpha = self.fc2_alpha = self.fc1_norm_const = None
        self._default_scalar = None
        self._destroyed = True


@deprecated_aliases(apply_topk_at_fc1="apply_routing_weights_before_fc2")
def get_symm_buffer_for_sm107_block_scaled_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    quant_kind: Sm107QuantKind = "mxfp8_e4m3",
    kernel_variant: Literal["inference", "genphase"] = "inference",
    combine_dtype: Literal["bf16", "nvfp4", "mxfp8"] = "bf16",
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    fallback_cluster_shape_mn: Optional[Tuple[int, int]] = None,
    schedule_policy: Tuple[str, Optional[int]] = ("grouped", None),
    work_id_mode: Sm107WorkIdMode = "grid_stride",
    fc2_use_bulk: bool = False,
    fc2_tma_stages: Optional[int] = None,
    epi_flag_batches: Tuple[int, int] = (4, 2),
    token_in_flag_batch: int = 1,
    token_padding_block: int = 64,
    sf_padding_block: int = 128,
    gate_up_clamp: Optional[float] = None,
    reduce_topk_in_kernel: bool = False,
    token_back_mode: Sm107TokenBackMode = "epi_warps",
    apply_routing_weights_before_fc2: bool = True,
    max_sm_count: Optional[int] = None,
    activation: Literal["swiglu", "situ"] = "swiglu",
    situ_beta: Optional[float] = None,
    situ_linear_beta: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    swiglu_beta: Optional[float] = None,
    batch_invariant: bool = False,
    input_norm_const: float = 1.0,
    has_fc1_alpha: bool = True,
    has_fc2_alpha: bool = True,
    has_fc1_norm_const: bool = True,
    process_group=None,
) -> Sm107BlockScaledSymmBuffer:
    """Allocate the SM107 block-scaled mega session workspace.

    Problem sizes positional, tuning knobs keyword-only (the standard mega
    allocator contract). ``intermediate`` is the width after activation. Expert
    weights are NOT owned by the workspace; they are passed per launch.
    """
    config = Sm107BlockScaledMoeConfig(
        num_total_experts=num_total_experts,
        max_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        hidden=hidden,
        intermediate=intermediate,
        rank=rank,
        world_size=world_size,
        quant_kind=quant_kind,
        kernel_variant=kernel_variant,
        combine_dtype=combine_dtype,
        batch_invariant=batch_invariant,
        input_norm_const=input_norm_const,
        has_fc1_alpha=has_fc1_alpha,
        has_fc2_alpha=has_fc2_alpha,
        has_fc1_norm_const=has_fc1_norm_const,
        mma_tiler_mnk=mma_tiler_mnk,
        cluster_shape_mn=cluster_shape_mn,
        fallback_cluster_shape_mn=fallback_cluster_shape_mn,
        schedule_policy=schedule_policy,
        work_id_mode=work_id_mode,
        fc2_use_bulk=fc2_use_bulk,
        fc2_tma_stages=fc2_tma_stages,
        epi_flag_batches=epi_flag_batches,
        token_in_flag_batch=token_in_flag_batch,
        token_padding_block=token_padding_block,
        sf_padding_block=sf_padding_block,
        gate_up_clamp=gate_up_clamp,
        reduce_topk_in_kernel=reduce_topk_in_kernel,
        token_back_mode=token_back_mode,
        apply_routing_weights_before_fc2=apply_routing_weights_before_fc2,
        max_sm_count=max_sm_count,
        activation=activation,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )
    return Sm107BlockScaledSymmBuffer(config, process_group=process_group)


def _expected_weight_shapes(
    cfg: Sm107BlockScaledMoeConfig,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int]:
    """(fc1 shape, fc2 shape, fc1 SF numel/expert, fc2 SF numel/expert)."""
    experts = cfg.experts_per_rank
    fc1_out = 2 * cfg.intermediate
    vec = cfg.sf_vec_size
    if cfg.quant_kind in ("nvfp4", "mxfp4", "mxfp4_mxfp8"):
        fc1_shape = (experts, cfg.hidden // 2, fc1_out)
        fc2_shape = (experts, cfg.intermediate // 2, cfg.hidden)
    else:
        fc1_shape = (experts, cfg.hidden, fc1_out)
        fc2_shape = (experts, cfg.intermediate, cfg.hidden)
    fc1_sf = swizzled_flat_sf_size(fc1_out, cfg.hidden // vec)
    fc2_sf = swizzled_flat_sf_size(cfg.hidden, cfg.intermediate // vec)
    return fc1_shape, fc2_shape, fc1_sf, fc2_sf


def _validate_weight_leg(
    name: str,
    leg: TransformedBlockScaledWeights,
    expected_weight_shape: Tuple[int, ...],
    expected_sf_numel: int,
    cfg: Sm107BlockScaledMoeConfig,
    device: torch.device,
) -> None:
    if (
        not isinstance(leg, tuple)
        or len(leg) != 2
        or not all(isinstance(t, torch.Tensor) for t in leg)
    ):
        raise ValueError(f"{name} must be a (weight, scale) tensor pair.")
    weight, scale = leg
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            f"{name} weight shape {tuple(weight.shape)} != expected "
            f"{expected_weight_shape}."
        )
    if tuple(scale.shape) != (expected_weight_shape[0], expected_sf_numel):
        raise ValueError(
            f"{name} scale numel {scale.numel()} != expected "
            f"{expected_sf_numel * expected_weight_shape[0]}."
        )
    if (
        weight.dtype != cfg.torch_weight_data_dtype
        or scale.dtype != cfg.torch_act_sf_dtype
    ):
        raise ValueError(
            f"{name} requires {cfg.torch_weight_data_dtype} weights and {cfg.torch_act_sf_dtype} scales."
        )
    if not weight.permute(0, 2, 1).is_contiguous() or not scale.is_contiguous():
        raise ValueError(
            f"{name} requires K-major weights and contiguous scale planes."
        )
    if any(t.device != device or t.data_ptr() % 16 for t in leg):
        raise ValueError(f"{name} must be on {device} with 16-byte aligned storage.")


def sm107_block_scaled_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: Sm107BlockScaledSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    fast_math: bool = True,  # accepted for mega API parity; the kernel has no toggle
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Fused dispatch, FC1, SwiGLU/SiTU, FC2, and combine; writes ``y[:num_tokens]``.

    The caller must have staged ``symm_buffer.x`` / ``.x_sf`` and the routing
    slices first. With ``y=None`` returns a workspace view (valid under stream
    ordering until the next launch).
    """
    if not fast_math:
        import warnings

        warnings.warn(
            "fast_math=False is a no-op for the SM107 block-scaled kernel.",
            stacklevel=2,
        )
    cfg = symm_buffer.config
    if y is not None and (
        y.ndim != 2
        or y.shape[1] != cfg.hidden
        or y.dtype != torch.bfloat16
        or y.device != symm_buffer.device
    ):
        raise ValueError(
            f"y must be a BF16 matrix with {cfg.hidden} columns on {symm_buffer.device}."
        )
    if num_tokens is None:
        num_tokens = int(y.shape[0]) if y is not None else symm_buffer.staged_tokens()
    if num_tokens is None:
        raise ValueError(
            "num_tokens unset and no tokens were staged; call the backend's "
            "stage_inputs (or note_staged_tokens) first."
        )
    if not (0 <= num_tokens <= cfg.max_tokens_per_rank):
        raise ValueError(
            f"num_tokens ({num_tokens}) out of range [0, {cfg.max_tokens_per_rank}]."
        )
    if y is not None and y.shape[0] < num_tokens:
        raise ValueError("y does not have enough rows for num_tokens.")

    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )

    if (
        y is not None
        and num_tokens > 0
        and not cfg.reduce_topk_in_kernel
        and y.is_contiguous()
        and y.data_ptr() % 16 == 0
    ):
        # TopkReduce runs locally and needs no symmetric output allocation.
        # REDG still uses the workspace's registered symmetric destination.
        symm_buffer.launch(
            transformed_l1, transformed_l2, output=y[:num_tokens], live_tokens=num_tokens
        )
        if sync:
            torch.cuda.synchronize()
        return None

    symm_buffer.launch(transformed_l1, transformed_l2, live_tokens=num_tokens)
    out = symm_buffer.output_activation[:num_tokens]
    if y is None:
        if sync:
            torch.cuda.synchronize()
        return out
    y[:num_tokens].copy_(out)
    if sync:
        torch.cuda.synchronize()
    return None


def sm107_block_scaled_mega_launch_thunk(
    transformed_l1: TransformedBlockScaledWeights,
    transformed_l2: TransformedBlockScaledWeights,
    symm_buffer: Sm107BlockScaledSymmBuffer,
):
    """Zero-arg relauncher over pre-staged inputs, for steady-state timing loops."""
    cfg = symm_buffer.config
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(cfg)
    _validate_weight_leg(
        "fc1", transformed_l1, fc1_shape, fc1_sf, cfg, symm_buffer.device
    )
    _validate_weight_leg(
        "fc2", transformed_l2, fc2_shape, fc2_sf, cfg, symm_buffer.device
    )

    def _thunk() -> None:
        symm_buffer.launch(transformed_l1, transformed_l2)

    return _thunk


__all__ = [
    "Sm107BlockScaledMoeConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
]
