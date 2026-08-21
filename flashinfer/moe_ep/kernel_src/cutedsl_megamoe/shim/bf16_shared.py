# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused BF16 SiTU shared-expert frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FusedBf16SituMlpConfig:
    capacity: int
    hidden: int
    intermediate: int
    situ_beta: float
    situ_linear_beta: Optional[float]
    max_active_clusters: Optional[int] = None
    single_cta: bool = False

    def __post_init__(self) -> None:
        if self.capacity <= 0 or self.capacity % 64:
            raise ValueError("capacity must be a positive multiple of 64.")
        if self.hidden <= 0 or self.hidden % 32:
            raise ValueError("hidden must be a positive multiple of 32.")
        if self.intermediate <= 0 or self.intermediate % 32:
            raise ValueError("intermediate must be a positive multiple of 32.")
        if self.situ_beta <= 0:
            raise ValueError("situ_beta must be positive.")
        if self.situ_linear_beta is not None and self.situ_linear_beta <= 0:
            raise ValueError("situ_linear_beta must be positive when provided.")


class FusedBf16SituMlpFrontend:
    """One-expert local FC1+SiTU+FC2 runner with reusable storage."""

    def __init__(self, config: FusedBf16SituMlpConfig) -> None:
        self.config = config
        c = config
        self.activation = torch.empty(
            (c.capacity, c.hidden), dtype=torch.bfloat16, device="cuda"
        )
        self.fc1_output = torch.empty(
            (c.capacity, c.intermediate), dtype=torch.bfloat16, device="cuda"
        )
        self.output = torch.empty(
            (c.capacity, c.hidden), dtype=torch.bfloat16, device="cuda"
        )
        self.topk_scores = torch.ones((c.capacity,), dtype=torch.float32, device="cuda")
        cluster_tile_tokens = 128 if c.single_cta else 256
        counter_slots = (
            c.capacity + cluster_tile_tokens - 1
        ) // cluster_tile_tokens + 1
        self.fc1_done_counter = torch.zeros(
            (counter_slots,), dtype=torch.int32, device="cuda"
        )
        self._offs: dict[int, torch.Tensor] = {}
        self._compiled = None
        self._kernel = None

    @staticmethod
    def _to_cute(tensor: torch.Tensor, *, assumed_align: int = 16):
        import cutlass.torch as cutlass_torch

        result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
        return result.mark_layout_dynamic(
            leading_dim=cutlass_torch.get_leading_dim(tensor)
        )

    def _get_offs(self, num_tokens: int) -> torch.Tensor:
        offs = self._offs.get(num_tokens)
        if offs is None:
            offs = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
            self._offs[num_tokens] = offs
        return offs

    def _runtime_kwargs(
        self,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
        num_tokens: int,
    ) -> dict:
        import cuda.bindings.driver as cuda

        return {
            "activation": self._to_cute(self.activation),
            "fc1_weight": self._to_cute(fc1_weight),
            "fc1_output": self._to_cute(self.fc1_output),
            "fc2_weight": self._to_cute(fc2_weight),
            "fc2_output": self._to_cute(self.output),
            "topk_scores": self._to_cute(self.topk_scores),
            "fc1_done_counter": self._to_cute(self.fc1_done_counter, assumed_align=4),
            "offs": self._to_cute(self._get_offs(num_tokens)),
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def _ensure_compiled(
        self, fc1_weight: torch.Tensor, fc2_weight: torch.Tensor, num_tokens: int
    ) -> None:
        if self._compiled is not None:
            return
        import cutlass
        import cutlass.cute as cute
        import cutlass.utils as utils

        from .bf16_situ import (
            bf16_epilogue_compile_context,
            install_situ_metadata,
        )
        from moe_bf16_glu import kernel_bf16_glu_fc12 as fc12

        c = self.config
        max_active_clusters = c.max_active_clusters
        if max_active_clusters is None:
            cluster_size = 1 if c.single_cta else 2
            max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
                cluster_size
            )
        kernel = fc12.Sm100SwigluBf16Fc12Kernel(
            mma_tiler_mnk=(256, 256, 64),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            group_hint=max_active_clusters,
            token_padding_block=64,
            load_balance_mode="static",
            static_expert_shape=(1, 2 * c.intermediate, c.hidden),
            force_static_sched=True,
            ab_dtype=cutlass.BFloat16,
            epi_flag_batch=(1, 1),
            apply_topk_in_fc1=False,
        )
        if c.single_cta:
            kernel.mma_tiler_mnk = (128, 256, 64)
            kernel.mma_tiler = kernel.mma_tiler_mnk
            kernel.cluster_shape_mn = (1, 1)
            kernel.use_2cta_instrs = False
            kernel.cta_group = fc12.tcgen05.CtaGroup.ONE
        install_situ_metadata(
            kernel,
            situ_beta=c.situ_beta,
            situ_linear_beta=c.situ_linear_beta,
        )
        kwargs = self._runtime_kwargs(fc1_weight, fc2_weight, num_tokens)
        compile_kwargs = dict(kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        with bf16_epilogue_compile_context(
            situ_beta=c.situ_beta,
            situ_linear_beta=c.situ_linear_beta,
        ):
            self._compiled = cute.compile(kernel, **compile_kwargs)
        self._kernel = kernel

    def run(
        self,
        hidden_states: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        self._validate(hidden_states, fc1_weight, fc2_weight)
        self._ensure_compiled(fc1_weight, fc2_weight, num_tokens)
        self.activation[:num_tokens].copy_(hidden_states)
        self.fc1_done_counter.zero_()
        kwargs = self._runtime_kwargs(fc1_weight, fc2_weight, num_tokens)
        self._compiled(**kwargs)
        return self.output[:num_tokens]

    def _validate(
        self,
        hidden_states: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc2_weight: torch.Tensor,
    ) -> None:
        c = self.config
        if not 0 < hidden_states.shape[0] <= c.capacity:
            raise ValueError(
                f"token count must be in [1, {c.capacity}], "
                f"got {hidden_states.shape[0]}."
            )
        expected = (
            (hidden_states, (hidden_states.shape[0], c.hidden)),
            (fc1_weight, (1, c.hidden, 2 * c.intermediate)),
            (fc2_weight, (1, c.intermediate, c.hidden)),
        )
        for tensor, shape in expected:
            if (
                not tensor.is_cuda
                or tensor.dtype != torch.bfloat16
                or tuple(tensor.shape) != shape
            ):
                raise ValueError(f"expected CUDA bfloat16 tensor with shape {shape}.")


_FRONTENDS: dict[tuple, FusedBf16SituMlpFrontend] = {}


def fused_bf16_situ_mlp(
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    *,
    capacity: int,
    situ_beta: float,
    situ_linear_beta: Optional[float],
    max_active_clusters: Optional[int] = None,
    single_cta: bool = False,
) -> torch.Tensor:
    """Run a local shared expert with one fused persistent kernel."""
    device = hidden_states.device
    key = (
        device.type,
        device.index,
        capacity,
        hidden_states.shape[1],
        fc2_weight.shape[1],
        situ_beta,
        situ_linear_beta,
        max_active_clusters,
        single_cta,
    )
    frontend = _FRONTENDS.get(key)
    if frontend is None:
        frontend = FusedBf16SituMlpFrontend(
            FusedBf16SituMlpConfig(
                capacity=capacity,
                hidden=hidden_states.shape[1],
                intermediate=fc2_weight.shape[1],
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                max_active_clusters=max_active_clusters,
                single_cta=single_cta,
            )
        )
        _FRONTENDS[key] = frontend
    return frontend.run(hidden_states, fc1_weight, fc2_weight)


__all__ = [
    "FusedBf16SituMlpConfig",
    "FusedBf16SituMlpFrontend",
    "fused_bf16_situ_mlp",
]
