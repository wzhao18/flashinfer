# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SiTU activation adapter for the fused BF16 FC1+FC2 kernel."""

from __future__ import annotations

import contextlib
from types import MethodType
from typing import Iterator, Optional

import cutlass
import cutlass.cute as cute

from ._paths import bootstrap_paths
from .nvfp4_situ import _COMPILE_LOCK, _situ_f32, _tanh_f32

bootstrap_paths()

from moe_bf16_glu.epilogue_bf16 import GluBf16Epilogue  # noqa: E402
from moe_bf16_glu.kernel_bf16_glu_fc12 import (  # noqa: E402
    Sm100SwigluBf16Fc12Kernel,
)


@cute.jit
def _situ_act(
    epilogue,
    output: cute.Tensor,
    up: cute.Tensor,
    gate: cute.Tensor,
    prob: Optional[cutlass.Float32] = None,
) -> None:
    beta = epilogue.situ_beta
    linear_beta = epilogue.situ_linear_beta
    for i in cutlass.range_constexpr(0, cute.size(output), 2):
        gate_pair = (
            _situ_f32(gate[i], beta, fastmath=True),
            _situ_f32(gate[i + 1], beta, fastmath=True),
        )
        up_pair = (up[i], up[i + 1])
        if cutlass.const_expr(linear_beta is not None):
            scale = cutlass.Float32(linear_beta)
            inv_scale = cutlass.Float32(1.0 / linear_beta)
            up_pair = (
                scale * _tanh_f32(up_pair[0] * inv_scale, fastmath=True),
                scale * _tanh_f32(up_pair[1] * inv_scale, fastmath=True),
            )
        result = cute.arch.mul_packed_f32x2(up_pair, gate_pair)
        if cutlass.const_expr(prob is not None):
            result = cute.arch.mul_packed_f32x2(result, (prob, prob))
        output[i], output[i + 1] = result


def _situ_kernel_name(kernel: Sm100SwigluBf16Fc12Kernel) -> str:
    return (
        f"sm100_bf16_fc12_{kernel.mma_tiler_mnk}_{kernel.cluster_shape_mn}"
        f"_situ_beta{kernel.situ_beta}"
        f"_linear_beta{kernel.situ_linear_beta}"
    )


def install_situ_metadata(
    kernel: Sm100SwigluBf16Fc12Kernel,
    *,
    situ_beta: float,
    situ_linear_beta: Optional[float],
) -> None:
    kernel.situ_beta = situ_beta
    kernel.situ_linear_beta = situ_linear_beta
    kernel.name = MethodType(_situ_kernel_name, kernel)


@contextlib.contextmanager
def bf16_epilogue_compile_context(
    *, situ_beta: float, situ_linear_beta: Optional[float]
) -> Iterator[None]:
    """Expose SiTU only while CuTeDSL traces the BF16 kernel."""
    with _COMPILE_LOCK:
        original_method = GluBf16Epilogue._swiglu_act
        old_beta = getattr(GluBf16Epilogue, "situ_beta", None)
        old_linear_beta = getattr(GluBf16Epilogue, "situ_linear_beta", None)
        had_beta = hasattr(GluBf16Epilogue, "situ_beta")
        had_linear_beta = hasattr(GluBf16Epilogue, "situ_linear_beta")
        GluBf16Epilogue._swiglu_act = _situ_act
        GluBf16Epilogue.situ_beta = situ_beta
        GluBf16Epilogue.situ_linear_beta = situ_linear_beta
        try:
            yield
        finally:
            GluBf16Epilogue._swiglu_act = original_method
            if had_beta:
                GluBf16Epilogue.situ_beta = old_beta
            else:
                del GluBf16Epilogue.situ_beta
            if had_linear_beta:
                GluBf16Epilogue.situ_linear_beta = old_linear_beta
            else:
                del GluBf16Epilogue.situ_linear_beta


__all__ = ["bf16_epilogue_compile_context", "install_situ_metadata"]
