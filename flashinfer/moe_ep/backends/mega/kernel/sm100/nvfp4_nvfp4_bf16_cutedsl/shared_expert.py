"""Local NVFP4 shared-expert workspace for the CuTeDSL MegaMoE backend."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ......kernel_src.cutedsl_megamoe import (
    MegaMoESharedNvfp4Inputs,
    fused_quant_stage,
    get_symm_buffer_for_mega_moe,
    nvfp4_mega_launch_thunk,
)
from .staging import stage_mega_moe_inputs
from .weights import TransformedMegaWeights


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class Nvfp4CutedslSharedExpertSession:
    """Reusable workspace for a replicated NVFP4 SiTU shared expert.

    The session supports both a standalone local MegaMoE launch and staging
    tensors for the shared FC1+FC2 path integrated into a distributed MegaMoE
    launch. A session has a fixed token capacity; only active rows are exposed
    to the kernel on each call.

    Args:
        max_num_tokens: Maximum active tokens in one invocation.
        hidden_size: Input and output hidden dimension.
        intermediate_size: Shared-expert output width after SiTU.
        num_sms: SM budget for standalone local execution.
        situ_beta: SiTU gate clamp parameter.
        situ_linear_beta: Optional SiTU linear clamp parameter.
    """

    def __init__(
        self,
        *,
        max_num_tokens: int,
        hidden_size: int,
        intermediate_size: int,
        num_sms: int,
        situ_beta: float,
        situ_linear_beta: float | None,
    ) -> None:
        if max_num_tokens <= 0:
            raise ValueError("max_num_tokens must be positive.")
        if hidden_size % 64 or intermediate_size % 64:
            raise ValueError(
                "hidden_size and intermediate_size must be multiples of 64."
            )
        if num_sms < 2:
            raise ValueError("num_sms must be at least 2.")

        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.active_num_tokens = 0
        self._integrated_staging: bool | None = None
        self._destroyed = False
        self.buffer = get_symm_buffer_for_mega_moe(
            1,
            max_num_tokens,
            1,
            hidden_size,
            2 * intermediate_size,
            0,
            1,
            activation="situ",
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            knobs={"max_active_clusters": num_sms // 2},
            local_only=True,
        )
        self.topk_ids = torch.zeros(max_num_tokens, 1, dtype=torch.int64, device="cuda")
        self.topk_weights = torch.ones(
            max_num_tokens, 1, dtype=torch.float32, device="cuda"
        )
        scale_rows = _ceil_div(max_num_tokens, 128) * 128
        self.activation_sf = torch.zeros(
            scale_rows,
            _ceil_div(hidden_size // 16, 4) * 4,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self.fc1_output = torch.empty(
            max_num_tokens,
            intermediate_size // 2,
            dtype=torch.float4_e2m1fn_x2,
            device="cuda",
        )
        self.fc1_output_sf = torch.zeros(
            scale_rows,
            _ceil_div(intermediate_size // 16, 4) * 4,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        self.fc1_done_counter = torch.zeros(
            max_num_tokens, dtype=torch.int32, device="cuda"
        )
        self._thunks: dict[tuple[int, ...], Callable[[], None]] = {}

    def stage(self, hidden_states: torch.Tensor, *, integrated: bool) -> None:
        """Quantize active input rows into this session's workspace."""
        self._ensure_live()
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                "Shared-expert input must have shape "
                f"(num_tokens, {self.hidden_size}), got {tuple(hidden_states.shape)}."
            )
        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            raise ValueError("Shared-expert input must contain at least one token.")
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                f"Shared-expert input has {num_tokens} tokens, but the session "
                f"capacity is {self.max_num_tokens}."
            )
        if integrated:
            fused_quant_stage(
                hidden_states,
                self.topk_ids,
                self.topk_weights,
                self.buffer.x,
                self.activation_sf,
                self.buffer.topk_idx,
                self.buffer.topk_weights,
                quant_type="nvfp4",
                norm_const=1.0,
                sf_layout="blocked_128x4",
            )
        else:
            stage_mega_moe_inputs(
                hidden_states,
                self.topk_weights,
                self.topk_ids,
                self.buffer.x,
                self.buffer.x_sf,
                self.buffer.topk_idx,
                self.buffer.topk_weights,
            )
        self.active_num_tokens = num_tokens
        self._integrated_staging = integrated

    def integrated_inputs(
        self, weights: TransformedMegaWeights
    ) -> MegaMoESharedNvfp4Inputs:
        """Build active views consumed by the distributed fused launch."""
        self._ensure_staged(integrated=True)
        fc1, fc2 = weights
        num_tokens = self.active_num_tokens
        scale_rows = _ceil_div(num_tokens, 128) * 128
        return MegaMoESharedNvfp4Inputs(
            activation=self.buffer.x[:num_tokens],
            activation_sf=self.activation_sf[:scale_rows],
            fc1_weight=fc1[0],
            fc1_weight_sf=fc1[1],
            fc1_output=self.fc1_output[:num_tokens],
            fc1_output_sf=self.fc1_output_sf[:scale_rows],
            fc2_weight=fc2[0],
            fc2_weight_sf=fc2[1],
            fc1_alpha=self.buffer.fc1_alpha,
            fc2_alpha=self.buffer.fc2_alpha,
            fc1_norm_const=self.buffer.fc1_norm_const,
            fc1_done_counter=self.fc1_done_counter[:num_tokens],
            output_activation=self.buffer.output_activation[:num_tokens],
        )

    def output(self) -> torch.Tensor:
        """Return the active output view."""
        self._ensure_staged()
        return self.buffer.output_activation[: self.active_num_tokens]

    def run(self, weights: TransformedMegaWeights) -> torch.Tensor:
        """Launch the standalone local shared expert for staged inputs."""
        self._ensure_staged(integrated=False)
        fc1, fc2 = weights
        key = (
            *(tensor.data_ptr() for pair in weights for tensor in pair),
            torch.cuda.current_stream().cuda_stream,
        )
        thunk = self._thunks.get(key)
        if thunk is None:
            thunk = nvfp4_mega_launch_thunk(fc1, fc2, self.buffer)
            self._thunks[key] = thunk
        thunk()
        return self.output()

    def destroy(self) -> None:
        """Release the session's local MegaMoE workspace."""
        if self._destroyed:
            return
        self._thunks.clear()
        self.buffer.destroy()
        self._destroyed = True

    def _ensure_live(self) -> None:
        if self._destroyed:
            raise RuntimeError("Shared-expert session has been destroyed.")

    def _ensure_staged(self, *, integrated: bool | None = None) -> None:
        self._ensure_live()
        if self._integrated_staging is None:
            raise RuntimeError("stage() must be called before using the session.")
        if integrated is not None and self._integrated_staging != integrated:
            mode = "integrated" if integrated else "standalone"
            raise RuntimeError(f"Inputs must be staged for {mode} execution.")
