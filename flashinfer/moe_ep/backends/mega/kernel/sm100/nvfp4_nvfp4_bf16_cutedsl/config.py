"""CuTeDSL NVFP4 mega-MoE kernel config."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    import torch


@dataclass
class Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig:
    """Kernel params for ``kernel_src.cutedsl_megamoe.nvfp4_mega_moe``.

    Expert weights must be NVFP4 at kernel launch; supply bf16 ``MoEWeightPack``
    and enable ``MegaConfig.preprocess_weights`` (default), or pass pre-quantized
    NVFP4 weights with ``w13_scale`` / ``w2_scale``.
    """

    intermediate_size: int
    top_k: int
    kernel_name: str = "sm100_nvfp4_nvfp4_bf16_cutedsl"
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    activation: Literal["swiglu", "situ"] = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    fast_math: bool = True
    apply_topk_in_fc1: bool = True
    # In-flight top-k combine: cross-rank REDG atomic-add collapses the combine
    # as peer data arrives (no per-topk staging / explicit tail reduce).
    # ~1-2% faster and removes the multi-GB combine staging from the symmetric
    # workspace.  Requires apply_topk_in_fc1=True and combine_dtype="bf16";
    # accumulation order is nondeterministic (tolerance-compare outputs).
    in_kernel_fc2_reduce: bool = False
    # Cross-rank combine wire format: "bf16" (exact), "mxfp8" (2x less combine
    # traffic), "nvfp4" (4x less).  Quantized wires trade accuracy for NVLink
    # bandwidth and require in_kernel_fc2_reduce=False.
    combine_dtype: Literal["bf16", "mxfp8", "nvfp4"] = "bf16"
    input_norm_const: float = 1.0
    fc1_alpha: Optional["torch.Tensor"] = None
    fc2_alpha: Optional["torch.Tensor"] = None
    fc1_norm_const: Optional["torch.Tensor"] = None
    shared_hidden_size: int | None = None
    shared_intermediate_size: int | None = None
    # Kernel tuning knobs (see kernel_src.cutedsl_megamoe.shim.tuner); overrides
    # the token-count default heuristic entirely when set, e.g. a winner from the
    # kernel repo's tester sweep. None -> tuner.default_knobs(num_max_tokens).
    # "auto" -> online autotune at the first forward: collectively time the
    # shim.autotune candidate set on the live problem and keep the winner
    # (one cute.compile per candidate, paid once per session).
    knobs: dict | str | None = None

    def __post_init__(self) -> None:
        if (self.shared_hidden_size is None) != (
            self.shared_intermediate_size is None
        ):
            raise ValueError(
                "shared_hidden_size and shared_intermediate_size must be "
                "set together."
            )
        if self.shared_hidden_size is not None:
            assert self.shared_intermediate_size is not None
            if self.shared_hidden_size % 64 or self.shared_intermediate_size % 64:
                raise ValueError(
                    "shared expert dimensions must be multiples of 64."
                )
        if self.activation not in ("swiglu", "situ"):
            raise ValueError(
                f"activation must be 'swiglu' or 'situ', got {self.activation!r}."
            )
        if self.activation == "situ":
            if self.situ_beta is None:
                raise ValueError("activation='situ' requires situ_beta.")
            if not math.isfinite(self.situ_beta) or self.situ_beta <= 0:
                raise ValueError("situ_beta must be positive and finite.")
            if self.situ_linear_beta is not None and (
                not math.isfinite(self.situ_linear_beta) or self.situ_linear_beta <= 0
            ):
                raise ValueError(
                    "situ_linear_beta must be positive and finite when set."
                )
            if self.gate_up_clamp is not None or self.activation_clamp is not None:
                raise ValueError("activation clamps are not supported with SiTU.")
        elif self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError("SiTU parameters require activation='situ'.")
