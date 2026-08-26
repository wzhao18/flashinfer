"""CuTeDSL NVFP4 mega-MoE kernel backend.

The fused kernel consumes NVFP4 expert weights in kernel-ready layout (packed
weights + atom-swizzled scale factors). ``MoEWeightPack`` supplies canonical
bf16 ``w13``/``w2`` by default; ``preprocess_weights()`` quantizes and swizzles
them. Pass pre-quantized NVFP4 weights via ``w13``/``w2`` + ``w13_scale``/``w2_scale``
to skip re-quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ......config import BootstrapConfig, FleetParams
from ......core.kernel.base import MegaKernelBackend
from ......core.kernel.registry import register_mega_kernel
from ......core.runtime import nvfp4_cutedsl_runtime_requirements
from ......core.validation.common import (
    validate_mega_arch,
    validate_mega_fleet_params,
)
from ......weights import MoEWeightPack
from .config import Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
from .staging import stage_mega_moe_inputs, validate_nvfp4_forward_inputs
from .weights import (
    TransformedMegaWeights,
    preprocess_mega_weights,
    validate_transformed_mega_weights,
)

if TYPE_CHECKING:
    from ......tensors import MoEEpTensors


def _resolve_gate_up_clamp(
    config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
) -> float | None:
    if config.gate_up_clamp is not None:
        return config.gate_up_clamp
    return config.activation_clamp


@register_mega_kernel(
    "sm100_nvfp4_nvfp4_bf16_cutedsl", deprecated_aliases=("nvfp4_cutedsl",)
)
class Nvfp4CutedslMegaKernelBackend(MegaKernelBackend):
    supports_output_view = True

    def __init__(self, config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig) -> None:
        super().__init__(config)
        self._kernel_config: Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig = config
        self._thunk_state: tuple | None = None
        self._active_epilogue: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
            None
        )
        # knobs="auto": tune at the first compute() (weights + staged inputs
        # exist there), then keep the winner for the session.
        self._autotune_pending = config.knobs == "auto"
        if self._autotune_pending:
            import warnings

            warnings.warn(
                "knobs='auto' runs a COLLECTIVE multi-minute compile+timing "
                "sweep at the first forward — never use it inside a serving "
                "engine. Tune offline instead (python -m flashinfer.moe_ep.tune"
                "); winners persist in the knob cache and knobs=None then "
                "resolves them with a pure lookup.",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def kernel_name(cls) -> str:
        return "sm100_nvfp4_nvfp4_bf16_cutedsl"

    def runtime_requirements(self, bootstrap: BootstrapConfig) -> frozenset[str]:
        return nvfp4_cutedsl_runtime_requirements(bootstrap)

    def validate_init(
        self,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_mega_arch()
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=self._kernel_config.intermediate_size,
            top_k=self._kernel_config.top_k,
            # cutedsl tiles are tail-safe (ceil-div + predicated epilogue);
            # the binding bound is TMA row alignment, not the dg SF word.
            alignment=64,
        )

    def preprocess_weights(
        self,
        weights: MoEWeightPack,
        fleet_params: FleetParams,
    ) -> TransformedMegaWeights:
        return preprocess_mega_weights(
            weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            gate_up_clamp=_resolve_gate_up_clamp(self._kernel_config),
            activation_clamp=self._kernel_config.activation_clamp,
        )

    def validate_transformed_weights(
        self,
        transformed_weights: TransformedMegaWeights,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_mega_weights(
            transformed_weights,
            intermediate_size=self._kernel_config.intermediate_size,
            hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size,
            num_experts=fleet_params.num_experts,
        )

    def _allocate_workspace(self, fleet_params: FleetParams) -> Any:
        from ......kernel_src.cutedsl_megamoe import get_symm_buffer_for_mega_moe

        k = self._kernel_config
        fp = fleet_params
        return get_symm_buffer_for_mega_moe(
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            2 * k.intermediate_size,
            self.ep_rank,
            self.ep_world_size,
            gate_up_clamp=_resolve_gate_up_clamp(k),
            activation_clamp=k.activation_clamp,
            activation=k.activation,
            situ_beta=k.situ_beta,
            situ_linear_beta=k.situ_linear_beta,
            apply_topk_in_fc1=k.apply_topk_in_fc1,
            in_kernel_fc2_reduce=k.in_kernel_fc2_reduce,
            combine_dtype=k.combine_dtype,
            fc1_alpha=k.fc1_alpha,
            fc2_alpha=k.fc2_alpha,
            fc1_norm_const=k.fc1_norm_const,
            knobs=k.knobs if isinstance(k.knobs, dict) else None,
            shared_hidden=k.shared_hidden_size,
            shared_intermediate=(
                2 * k.shared_intermediate_size
                if k.shared_intermediate_size is not None
                else None
            ),
        )

    def validate_forward(
        self,
        t: "MoEEpTensors",
        fleet_params: FleetParams,
        *,
        quantize_input: bool,
    ) -> None:
        validate_nvfp4_forward_inputs(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            fleet_params,
            top_k=self._kernel_config.top_k,
            quantize_input=quantize_input,
            scales=t.scales,
        )
        shared_values = (
            t.shared_hidden_states,
            t.shared_expert_weights,
            t.shared_expert_output,
        )
        num_shared_values = sum(value is not None for value in shared_values)
        if num_shared_values not in (0, len(shared_values)):
            raise ValueError(
                "shared_hidden_states, shared_expert_weights, and "
                "shared_expert_output must be set together."
            )
        expects_shared = self._kernel_config.shared_hidden_size is not None
        if bool(num_shared_values) != expects_shared:
            requirement = "requires" if expects_shared else "does not support"
            raise ValueError(
                f"This kernel configuration {requirement} shared expert inputs."
            )

    def stage_inputs(
        self,
        t: "MoEEpTensors",
        workspace: Any,
        *,
        quantize_input: bool,
    ) -> None:
        num_tokens = t.hidden_states.shape[0]
        if quantize_input:
            stage_mega_moe_inputs(
                t.hidden_states,
                t.topk_weights,
                t.topk_ids,
                workspace.x,
                workspace.x_sf,
                workspace.topk_idx,
                workspace.topk_weights,
                norm_const=self._kernel_config.input_norm_const,
                token_padding_info=t.is_padding,
            )
        else:
            # Backend talks only to the cutedsl_megamoe shim (never src/ directly).
            from ......kernel_src.cutedsl_megamoe import (
                Nvfp4BlockSize,
                ceil_div,
                round_up,
            )

            hidden = workspace.hidden
            hidden_sf_cols = ceil_div(hidden, Nvfp4BlockSize)
            hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)

            workspace.x[:num_tokens].copy_(t.hidden_states)
            assert t.scales is not None
            workspace.x_sf[:num_tokens].zero_()
            workspace.x_sf[:num_tokens, :hidden_sf_cols].copy_(
                t.scales[:num_tokens, :hidden_sf_cols]
            )
            if t.scales.shape[1] >= hidden_sf_cols_padded:
                workspace.x_sf[
                    :num_tokens, hidden_sf_cols:hidden_sf_cols_padded
                ].zero_()
            workspace.topk_idx[:num_tokens].copy_(t.topk_ids)
            workspace.topk_weights[:num_tokens].copy_(t.topk_weights)
            capacity = workspace.x.shape[0]
            if num_tokens < capacity:
                workspace.topk_idx[num_tokens:capacity].fill_(-1)
            from ......kernel_src.cutedsl_megamoe import note_staged_tokens

            note_staged_tokens(workspace.topk_idx, num_tokens)

        def bind_epilogue(
            source: torch.Tensor | None, target: torch.Tensor
        ) -> torch.Tensor:
            if source is None:
                return target
            if (
                source.shape == target.shape
                and source.dtype == target.dtype
                and source.device == target.device
                and source.is_contiguous()
            ):
                return source
            target.copy_(source)
            return target

        self._active_epilogue = (
            bind_epilogue(t.fc1_alpha, workspace.fc1_alpha),
            bind_epilogue(t.fc2_alpha, workspace.fc2_alpha),
            bind_epilogue(t.fc1_norm_const, workspace.fc1_norm_const),
        )
        if t.shared_hidden_states is not None:
            assert t.shared_expert_weights is not None
            assert t.shared_expert_output is not None
            workspace._stage_shared(
                t.shared_hidden_states,
                t.shared_expert_weights,
                t.shared_expert_output,
            )
        else:
            workspace._shared_inputs = None

    def compute(
        self,
        workspace: Any,
        transformed_weights: TransformedMegaWeights,
        *,
        output: torch.Tensor | None,
    ) -> torch.Tensor:
        from ......kernel_src.cutedsl_megamoe import staged_tokens

        if output is not None:
            num_tokens = output.shape[0]
        else:
            if self._autotune_pending:
                raise ValueError(
                    "compute(output=None) is incompatible with knobs='auto' "
                    "(the autotune sweep needs a caller output buffer)"
                )
            staged = staged_tokens(workspace.topk_idx)
            if staged is None:
                raise ValueError(
                    "compute(output=None) requires stage_inputs() to have "
                    "staged this workspace first"
                )
            num_tokens = staged

        kcfg = self._kernel_config
        if self._autotune_pending:
            # COLLECTIVE: every EP rank reaches this first compute() together,
            # so the candidate sweep stays in lockstep (see shim/autotune.py).
            from ......kernel_src.cutedsl_megamoe import autotune_nvfp4_mega_moe

            autotune_nvfp4_mega_moe(
                output,
                transformed_weights[0],
                transformed_weights[1],
                workspace,
                num_tokens=num_tokens,
                gate_up_clamp=_resolve_gate_up_clamp(kcfg),
                activation_clamp=kcfg.activation_clamp,
            )
            # Cleared only on success: if the collective tune raises, a retried
            # compute() re-attempts it (all ranks fail together, so lockstep holds).
            self._autotune_pending = False
        # Steady-state launch thunk: nvfp4_mega_moe re-validates, re-resolves
        # the clamp, and rebuilds the 12-field inputs bundle on every call
        # (~70us of loop-invariant host Python at 43 layers x 4 ranks — the
        # measured arrival-skew generator; see vllm_e2e RUNS.md run 27/28).
        # Build once per (workspace, weights, compiled-session, STREAM) and
        # reuse. The stream is part of the key because the thunk's launch
        # kwargs bind it at build time — a graph capture runs on a capture
        # stream and must get its own thunk or the kernel launch escapes the
        # graph. A knobs/clamp change nulls the frontend's compiled session,
        # changing the key and forcing a rebuild through the validated path.
        shared_inputs = workspace._shared_inputs
        active_epilogue = self._active_epilogue
        if active_epilogue is None:
            raise ValueError("compute() requires stage_inputs() to run first")
        fc1_alpha, fc2_alpha, fc1_norm_const = active_epilogue
        fe = workspace._frontend
        clamp = _resolve_gate_up_clamp(kcfg)
        if clamp is not None:
            fe.set_gate_up_clamp(clamp)
        mega = fe._mega
        stream = torch.cuda.current_stream().cuda_stream
        # IKR writes only epilogue tiles that cover live rows. Clear the same
        # 64-row extent instead of the capacity-sized output workspace.
        clear_tokens = min(
            ((max(num_tokens, 1) + 63) // 64) * 64,
            workspace.x.shape[0],
        )
        key = (
            id(workspace),
            id(transformed_weights[0][0]),
            id(mega.compiled) if mega is not None and mega.compiled else None,
            stream,
            clear_tokens,
            tuple(
                (tensor.data_ptr(), tuple(tensor.shape))
                for tensor in active_epilogue
            ),
            tuple(
                (tensor.data_ptr(), tuple(tensor.shape))
                for tensor in vars(shared_inputs).values()
                if isinstance(tensor, torch.Tensor)
            )
            if shared_inputs is not None
            else (),
        )
        state = self._thunk_state
        if state is None or state[0] != key or key[2] is None:
            from ......kernel_src.cutedsl_megamoe import (
                MegaMoENvfp4Inputs,
            )

            inputs = MegaMoENvfp4Inputs(
                activation=workspace.x,
                activation_sf=workspace.x_sf,
                topk_idx=workspace.topk_idx,
                topk_weights=workspace.topk_weights,
                fc1_weight=transformed_weights[0][0],
                fc1_weight_sf=transformed_weights[0][1],
                fc2_weight=transformed_weights[1][0],
                fc2_weight_sf=transformed_weights[1][1],
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                output_activation=workspace.output_activation,
                shared=shared_inputs,
            )
            # Full validation happens inside make_launch_thunk's
            # _prepare_launch_inputs (run()'s slow-path validator).
            thunk = fe.make_launch_thunk(inputs, zero_num_tokens=clear_tokens)
            mega = fe._mega
            key = (*key[:2], id(mega.compiled), *key[3:])
            state = (key, thunk, workspace.output_activation)
            self._thunk_state = state

        _, thunk, out_buf = state
        thunk()
        if output is not None:
            output.copy_(out_buf[:num_tokens])
            return output
        # Zero-copy: the caller consumes the [:n] view under stream ordering
        # (valid until the next launch on this session's buffers).
        return out_buf[:num_tokens]

    def _workspace_pool_key(self, fleet_params: FleetParams) -> Any:
        k = self._kernel_config
        if k.knobs == "auto":
            # Autotune retunes (and recompiles) the workspace's shared
            # frontend at first compute; give each session its own buffer.
            return None
        import torch

        from ......core.kernel.workspace_pool import epilogue_pool_key, knobs_pool_key

        fp = fleet_params
        return (
            "sm100_nvfp4_nvfp4_bf16_cutedsl",
            torch.cuda.current_device(),
            self.ep_rank,
            self.ep_world_size,
            id(self._ep_comm_group),
            fp.num_experts,
            fp.max_tokens_per_rank,
            k.top_k,
            fp.token_hidden_size,
            2 * k.intermediate_size,
            _resolve_gate_up_clamp(k),
            k.activation,
            k.situ_beta,
            k.situ_linear_beta,
            k.apply_topk_in_fc1,
            k.in_kernel_fc2_reduce,
            k.combine_dtype,
            k.shared_hidden_size,
            k.shared_intermediate_size,
            epilogue_pool_key(k.fc1_alpha),
            epilogue_pool_key(k.fc2_alpha),
            epilogue_pool_key(k.fc1_norm_const),
            knobs_pool_key(k.knobs),
        )

    def _forget_workspace_state(self, workspace) -> None:
        # The fused-stage memos key on topk_idx.data_ptr(); the symmetric
        # heap reuses freed addresses, so evict before the buffer dies.
        # sys.modules lookup (not an import): if the shim was never loaded,
        # no memo exists and the heavy import must not happen.
        import sys

        quant_stage = sys.modules.get(
            "flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.quant_stage"
        )
        topk_idx = getattr(workspace, "topk_idx", None)
        if quant_stage is not None and topk_idx is not None:
            quant_stage.forget_staged_tokens(topk_idx)
