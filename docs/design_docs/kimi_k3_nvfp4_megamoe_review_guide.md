# Kimi K3 NVFP4 MegaMoE review guide

This document describes the current changes on
`wzhao/k3-mega-moe-shared-fc12-balanced-fc1`, relative to its merge base with
`origin/main`, `fb28d724`. The implementation changes 21 files.

## Scope

The branch extends the SM100 CuTe DSL NVFP4 MegaMoE backend for the Kimi K3
MoE path. It adds:

- SiTU as an FC1 activation, including Kimi's optional linear-branch clamp.
- A rank-local shared expert whose FC1 and FC2 tiles execute inside the same
  persistent kernel as distributed routed MoE work.
- NVFP4 staging directly into the 128-by-4 blocked scale-factor layout needed
  by the integrated shared FC1.
- Kimi-specific correctness coverage and an EP16 benchmark.

The host integration deliberately extends the existing MegaMoE lifecycle
instead of adding a second shared-expert API. The existing pooled
`MegaMoESymmBuffer` owns shared scratch when shared dimensions are configured.
The normal per-forward tensor bundle supplies the shared input, transformed
weights, and caller-owned output. There is no public session object, separate
frontend, local-only allocation mode, or additional destruction protocol.

The shared-expert output remains a separate tensor. This branch does not add
it to the routed-MoE output, and it does not contain the vLLM model wiring that
performs the model-level residual/mixing operation.

Two size conventions are important throughout the review:

- `shared_intermediate_size` in the backend config is the post-SiTU width.
- `shared_intermediate` in the raw kernel config is the FC1 gate-plus-up width,
  so the backend passes `2 * shared_intermediate_size`.

## Data flow

The integrated path is:

1. The caller sets `shared_hidden_states`, `shared_expert_weights`, and
   `shared_expert_output` on the normal `MoEEpTensors` object.
2. The existing backend `stage_inputs()` stages routed inputs as before, then
   asks its pooled `MegaMoESymmBuffer` to quantize the BF16 shared input into
   its local packed-data and blocked-scale scratch.
3. The workspace combines active scratch views, transformed weights, counters,
   and the caller-owned output into a private launch-argument group.
4. The existing NVFP4 frontend compiles or reuses one configuration containing
   both routed and shared geometry.
5. The persistent scheduler emits shared FC1 work before waiting for routed
   dispatch, executes normal routed FC1/FC2 work, and emits shared FC2 work
   afterward. Per-token-tile completion counters prevent shared FC2 from
   reading an incomplete shared FC1 result.
6. Routed output is returned through the normal MegaMoE output buffer. Shared
   FC2 writes directly to `shared_expert_output`.

A configuration with shared dimensions requires all three shared fields on
every forward. A normal configuration without shared dimensions follows the
unchanged routed-only path. This compile-time distinction avoids dynamic
frontend switching and duplicate workspaces.

## Recommended review order

### 1. Public contract and orchestration

Start here to establish what callers provide and what the backend promises.

1. `flashinfer/moe_ep/backends/mega/kernel/sm100/`
   `nvfp4_nvfp4_bf16_cutedsl/config.py`
2. `flashinfer/moe_ep/tensors.py`
3. `flashinfer/moe_ep/backends/mega/kernel/sm100/`
   `nvfp4_nvfp4_bf16_cutedsl/backend.py`

The main questions in this stage are whether the size conventions are clear,
the three optional per-forward fields form a clear contract, and normal
routed-only configurations remain behaviorally identical.

### 2. Python shim and compilation boundary

These files translate the public tensor contract into the raw CuTe DSL ABI.

1. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/nvfp4.py`
2. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/nvfp4_situ.py`
3. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/quant_stage.py`
4. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/knob_cache.py`
5. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/autotune.py`

Pay particular attention to validation of every shared tensor, compile-cache
identity, the serialized SiTU compile-time method substitution, and reuse of
the normal pooled workspace across model layers.

### 3. Raw persistent-kernel implementation

Review the ABI entry point and scheduler before the large mainloop diff.

1. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/megamoe_kernel.py`
2. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/fc1_fc2_fuse_sched.py`
3. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/custom_ext.py`
4. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/kernel_fc12.py`
5. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/epilogue_refactor.py`
6. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/`
   `moe_nvfp4_swapab/epilogue.py`
7. `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/src/inputs_process.py`

The critical correctness properties are that shared work cannot consume the
routed communication state, shared FC2 cannot overtake shared FC1, all TMA
layouts agree with the packed data and scale-factor storage, and the scheduler
publishes exactly one terminal sentinel to every consumer pipeline.

### 4. Tuning path

1. `flashinfer/moe_ep/backends/mega/kernel/sm100/`
   `nvfp4_nvfp4_bf16_cutedsl/tuner.py`
2. `flashinfer/moe_ep/tune.py`

Verify that SiTU and SwiGLU results cannot collide in the persistent knob
cache and that the CLI passes all activation parameters into dummy sessions.

### 5. Correctness tests

1. `tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py`
2. `tests/moe_ep/test_fused_quant_stage.py`
3. `tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py`

The single-rank oracle is the fastest way to review the activation algebra.
The blocked-stage test isolates scale layout. The multirank suite then covers
real dispatch/combine, integrated shared execution, repeated launches, and
in-kernel reduction behavior.

### 6. Performance benchmark

Finish with `benchmarks/bench_kimi_k3_nvfp4_ep.py`. It is intentionally last:
it makes more sense after the launch boundaries and output contracts are
understood.

## File-by-file summary

### Public API and backend

#### `flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/config.py`

Adds `activation`, `situ_beta`, and `situ_linear_beta` to the NVFP4 backend
configuration. It validates SiTU parameters and disallows mixing SiTU with the
SwiGLU clamp fields. It also adds paired shared-expert dimensions and validates
their 64-element alignment.

#### `flashinfer/moe_ep/tensors.py`

Adds three optional fields to the existing per-forward tensor bundle:
`shared_hidden_states`, `shared_expert_weights`, and `shared_expert_output`.
They are backend-specific inputs, must be supplied together, and do not expose
the kernel's internal scratch layout.

#### `flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/backend.py`

Propagates activation and shared geometry into workspace construction. It
validates the three shared fields, and stages them through the normal workspace.
It includes active shared pointers/shapes in the launch-thunk cache key and
forwards the private argument group into `MegaMoENvfp4Inputs`. It also limits
the in-kernel-reduction output clear to the live 64-row tile extent and extends
the workspace-pool key with all compile-relevant activation/shared fields.

### NVFP4 shim and staging

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/nvfp4.py`

This is the main Python-side implementation. It:

- Extends the raw config with SiTU, shared geometry, and an optional
  active-cluster limit.
- Keeps the shared launch-argument grouping private to this module and embeds
  it in the normal launch bundle.
- Validates shared data, scale, weight, epilogue, counter, and output tensors.
- Adds shared tensors and configuration to compile and launch cache keys.
- Builds the shared TMA launch arguments and passes them to the raw kernel.
- Allocates local shared scratch as part of the existing pooled
  `MegaMoESymmBuffer` and stages BF16 shared input there.
- Clears shared FC1 readiness counters before each launch.
- Supports a bounded output clear for in-kernel reduction.
- Adds activation to knob resolution and dummy-input construction.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/nvfp4_situ.py`

Implements Kimi's SiTU algebra in CuTe DSL:

`SiTU(gate) = beta * tanh(gate / beta) * sigmoid(gate)`

and, when configured:

`linear(up) = linear_beta * tanh(up / linear_beta)`.

The adapter installs SiTU constants in the kernel compile identity and swaps
the raw FC1 epilogue method only within a process-wide reentrant compile lock.
The original method and class attributes are restored after tracing, so the
runtime launch and later SwiGLU compilations do not retain Python mutation.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/quant_stage.py`

Adds the `blocked_128x4` NVFP4 scale layout to fused BF16-to-NVFP4 staging.
The layout participates in the compiled stager and launch cache keys. It is
validated as NVFP4-only and passes the full padded scale plane to the kernel,
while the existing row-major path remains unchanged.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/knob_cache.py`

Adds activation type to the tuning-cache identity, preventing a SwiGLU winner
from being reused for SiTU or vice versa. Existing version-1 entries are
normalized to `activation="swiglu"` for backward compatibility.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/shim/autotune.py`

Records the active NVFP4 activation in automatically selected knob-cache
entries.

### Raw CuTe DSL kernel

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/megamoe_kernel.py`

Extends the MegaMoE wrapper constructor with optional shared geometry and adds
all shared tensors to the kernel call signature. The wrapper forwards those
arguments into the fused FC1/FC2 implementation.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/fc1_fc2_fuse_sched.py`

Adds `SharedLinear1` and `SharedLinear2` phases. Scheduler parameters carry the
shared `(tokens, gate_up_width, hidden)` shape, including MLIR serialization.
The scheduler derives shared token/output tile counts and `gen_shared_work()`
maps a linear shared task index to phase, token tile, output tile, and valid
tail rows.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/custom_ext.py`

Adds phase predicates to the work-tile object. `is_linear1` treats routed and
shared FC1 uniformly, while `is_shared` distinguishes shared tasks from routed
tasks. This keeps phase branching consistent across mainloop and epilogue
code.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/kernel_fc12.py`

Contains the core fused implementation. It creates a shared epilogue, maps
shared packed tensors and blocked scales into GEMM layouts, constructs shared
FC1/FC2 TMA descriptors, and passes the shared task space to the persistent
scheduler. Shared FC1 tiles are distributed across persistent clusters before
routed dispatch work; shared FC2 tiles run after routed work. Common helper
paths issue TMA A-side weight/scale loads and B-side activation/scale loads.
Shared FC2 waits on the corresponding FC1 completion counter before consuming
the quantized intermediate. MMA and epilogue warps select routed or shared
state from the work phase.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/epilogue_refactor.py`

Creates separate shared FC1 and FC2 epilogue objects when shared execution is
compiled. It dispatches each completed accumulator tile to the routed or
shared epilogue and signals the matching completion tracker. The shared FC2
epilogue writes directly to the separate rank-local shared output and does not
use routed token communication.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/moe_nvfp4_swapab/epilogue.py`

Updates the older epilogue path to use the generalized `is_linear1` predicate,
so the phase classification remains consistent with the new scheduler enum.

#### `flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/src/inputs_process.py`

Extends `DataPreprocess` with a scale-layout selector. For
`blocked_128x4`, it uses CUTLASS's block-scaled layout construction and writes
each NVFP4 scale to the physical location consumed directly by shared FC1.
The existing row-major NVFP4 and MXFP8 behavior is preserved.

### Tuning

#### `flashinfer/moe_ep/backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/tuner.py`

Passes activation and SiTU parameters into dummy NVFP4 sessions and includes
activation when resolving the base schedule for a tuning sweep.

#### `flashinfer/moe_ep/tune.py`

Adds CLI flags for activation, SiTU beta, and optional linear beta. It rejects
SiTU for non-NVFP4 tuners and keeps NVFP4/MXFP8 tuner imports type-checker-safe.

### Tests and benchmark

#### `tests/moe_ep/test_nvfp4_cutedsl_kernel_vs_reference.py`

Generalizes the pure-Torch NVFP4 oracle from SwiGLU to selectable gated
activation. It adds SiTU coverage for a small geometry and the Kimi DEP16
routed-expert geometry `(hidden=3584, intermediate=3072, experts=56,
topk=16)`.

#### `tests/moe_ep/test_fused_quant_stage.py`

Adds a focused test proving that fused blocked-layout staging is byte-identical
to applying the canonical `to_blocked` transform to row-major scale factors,
while packed activations and routing outputs remain identical.

#### `tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py`

Adds SiTU to the existing multirank reference framework and introduces an
integrated shared-expert test. It compares the caller-owned shared output with
an existing distributed NVFP4 reference path, checks repeated launches, and
checks that bounded in-kernel-reduction clears leave the unused capacity tail
untouched. It also adds two-GPU SiTU oracle coverage and config validation.
Test geometry is expressed through helper arguments rather than environment
variables.

#### `benchmarks/bench_kimi_k3_nvfp4_ep.py`

Adds a standalone distributed benchmark for Kimi K3 decode shapes. It compares
three providers:

- `mega-fused`: FlashInfer routed MegaMoE plus integrated shared FC1/FC2.
- `mega-routed`: the same FlashInfer routed path without shared work.
- `trtllm`: TensorRT-LLM NVFP4 routed MoE behind rectangular all-gather and
  reduce-scatter.

The benchmark uses Kimi's routed geometry `(896 experts, topk=16, hidden=3584,
intermediate=3072)`, shared geometry `(hidden=7168, intermediate=6144)`, SiTU
parameters `(beta=4, linear_beta=25)`, and production EP16 scheduling knobs. It
sweeps global token counts, records rectangular padding, and reports the
cross-rank maximum of CUDA-event and synchronized wall latency. An optional
CUDA profiler range brackets only measured iterations. Shared input staging is
part of the measured fused path, matching normal backend execution.

## Highest-risk review points

1. **Scheduler ordering and liveness:** shared FC1 is intentionally placed
   before the routed-dispatch wait, while shared FC2 is placed after routed
   work. Review sentinel publication and FC1 counter thresholds for partial
   token/output tiles.
2. **TMA and scale layouts:** shared activations and FC1 intermediates use
   packed FP4 data with 128-by-4 blocked scale factors. Shape padding and
   physical addressing must agree between staging and `kernel_fc12.py`.
3. **Compile isolation for SiTU:** `nvfp4_situ.py` performs a guarded temporary
   method replacement because the raw epilogue is resolved during tracing.
   Verify that every compile path enters the context and that the compile key
   includes both SiTU constants.
4. **Pooled workspace state:** model layers reuse one workspace sequentially.
   Review that every forward refreshes shared weights, active views, output,
   counters, and thunk identity without retaining stale per-layer state.
5. **CUDA graph identity:** launch thunks are stream-specific and include
   shared data pointers, shapes, active clear extent, and compiled-session
   identity. Missing any of these can replay a graph with stale buffers.
6. **Output contract:** shared output is deliberately separate from routed
   output. Integration code must consume both exactly once.
