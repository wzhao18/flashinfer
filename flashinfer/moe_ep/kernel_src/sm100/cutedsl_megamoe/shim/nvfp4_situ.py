# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SiTU activation adapter for the raw NVFP4 MegaMoE kernel.

The raw kernel resolves its decorated epilogue method while CuTeDSL traces it.
The shim substitutes that method only inside a serialized compile scope and
restores it before returning; runtime launches use the compiled SiTU path.
"""

from __future__ import annotations

import contextlib
import ctypes
import threading
from types import MethodType
from typing import Iterator, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace

from flashinfer.fused_moe.cute_dsl.blackwell.utils import sigmoid_f32

from ._paths import bootstrap_paths

bootstrap_paths()

from moe_nvfp4_swapab.epilogue_refactor import (  # noqa: E402
    SwapABFc1Epilogue,
)
from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel  # noqa: E402


def _tanh_f32(a, *, fastmath: bool):
    return cutlass.Float32(2.0) * sigmoid_f32(
        cutlass.Float32(2.0) * a, fastmath=fastmath
    ) - cutlass.Float32(1.0)


def _f32_reciprocal(value: float) -> float:
    value_f32 = ctypes.c_float(float(value)).value
    return ctypes.c_float(1.0 / value_f32).value


def _situ_f32(a, beta: float, *, fastmath: bool):
    x = cutlass.Float32(a)
    beta_f32 = cutlass.Float32(beta)
    inv_beta = cutlass.Float32(_f32_reciprocal(beta))
    return (
        beta_f32
        * _tanh_f32(x * inv_beta, fastmath=fastmath)
        * sigmoid_f32(x, fastmath=fastmath)
    )


@cute.jit
def _alpha_situ(
    fc1_epilogue,
    gate_rmem: cute.Tensor,
    up_rmem: cute.Tensor,
    alpha_val: Optional[cutlass.Float32],
) -> cute.Tensor:
    for name, tensor in (("gate_rmem", gate_rmem), ("up_rmem", up_rmem)):
        if cutlass.const_expr(tensor.element_type is not cutlass.Float32):
            raise TypeError(
                f"alpha_situ: {name} must be Float32, got {tensor.element_type}"
            )
        if cutlass.const_expr(tensor.memspace != AddressSpace.rmem):
            raise ValueError(
                f"alpha_situ: {name} must be a register tensor, "
                f"got address space {tensor.memspace}"
            )
        if cutlass.const_expr(cute.rank(tensor) != 1):
            raise ValueError(
                f"alpha_situ: {name} must be 1D, got rank {cute.rank(tensor)}"
            )
        if cutlass.const_expr(cute.size(tensor) % 2 != 0):
            raise ValueError(
                f"alpha_situ: {name} element count must be even, "
                f"got {cute.size(tensor)}"
            )
    if cutlass.const_expr(cute.size(gate_rmem) != cute.size(up_rmem)):
        raise ValueError(
            "alpha_situ: gate_rmem and up_rmem must have equal "
            f"size, got {cute.size(gate_rmem)} vs {cute.size(up_rmem)}"
        )

    num_values = cute.size(gate_rmem)
    output = cute.make_rmem_tensor((num_values,), cutlass.Float32)
    situ_beta = fc1_epilogue.situ_beta
    if cutlass.const_expr(fc1_epilogue.situ_linear_beta is not None):
        linear_beta = cutlass.Float32(fc1_epilogue.situ_linear_beta)
        inv_linear_beta = cutlass.Float32(
            _f32_reciprocal(fc1_epilogue.situ_linear_beta)
        )

    for i in cutlass.range_constexpr(0, num_values, 2):
        gate_pair = (gate_rmem[i], gate_rmem[i + 1])
        up_pair = (up_rmem[i], up_rmem[i + 1])

        if cutlass.const_expr(alpha_val is not None):
            alpha_pair = (alpha_val, alpha_val)
            gate_pair = cute.arch.mul_packed_f32x2(gate_pair, alpha_pair)
            up_pair = cute.arch.mul_packed_f32x2(up_pair, alpha_pair)

        situ_gate_pair = (
            _situ_f32(gate_pair[0], situ_beta, fastmath=True),
            _situ_f32(gate_pair[1], situ_beta, fastmath=True),
        )
        if cutlass.const_expr(fc1_epilogue.situ_linear_beta is not None):
            up_pair = (
                linear_beta * _tanh_f32(up_pair[0] * inv_linear_beta, fastmath=True),
                linear_beta * _tanh_f32(up_pair[1] * inv_linear_beta, fastmath=True),
            )
        output_pair = cute.arch.mul_packed_f32x2(up_pair, situ_gate_pair)
        output[i] = output_pair[0]
        output[i + 1] = output_pair[1]

    return output


def _situ_kernel_name(kernel: Sm100MegaMoEKernel) -> str:
    return (
        f"{Sm100MegaMoEKernel.name(kernel)}_situ_beta{kernel.situ_beta}"
        f"_linear_beta{kernel.situ_linear_beta}"
    )


def install_situ_metadata(
    kernel: Sm100MegaMoEKernel,
    *,
    situ_beta: float,
    situ_linear_beta: Optional[float],
) -> None:
    """Add SiTU constants to the raw kernel's compile identity."""
    kernel.situ_beta = situ_beta
    kernel.situ_linear_beta = situ_linear_beta
    kernel.name = MethodType(_situ_kernel_name, kernel)


_COMPILE_LOCK = threading.RLock()


@contextlib.contextmanager
def nvfp4_epilogue_compile_context(
    *, situ_beta: Optional[float], situ_linear_beta: Optional[float]
) -> Iterator[None]:
    """Serialize NVFP4 traces and expose SiTU only for its own compilation."""
    with _COMPILE_LOCK:
        if situ_beta is None:
            yield
            return
        original_method = SwapABFc1Epilogue.alpha_swiglu_clamp
        old_beta = getattr(SwapABFc1Epilogue, "situ_beta", None)
        old_linear_beta = getattr(SwapABFc1Epilogue, "situ_linear_beta", None)
        had_beta = hasattr(SwapABFc1Epilogue, "situ_beta")
        had_linear_beta = hasattr(SwapABFc1Epilogue, "situ_linear_beta")
        SwapABFc1Epilogue.alpha_swiglu_clamp = _alpha_situ
        SwapABFc1Epilogue.situ_beta = situ_beta
        SwapABFc1Epilogue.situ_linear_beta = situ_linear_beta
        try:
            yield
        finally:
            SwapABFc1Epilogue.alpha_swiglu_clamp = original_method
            if had_beta:
                SwapABFc1Epilogue.situ_beta = old_beta
            else:
                del SwapABFc1Epilogue.situ_beta
            if had_linear_beta:
                SwapABFc1Epilogue.situ_linear_beta = old_linear_beta
            else:
                del SwapABFc1Epilogue.situ_linear_beta
