"""
Test for AllReduce + Residual + RMSNorm + Per-Token-Group FP8 Quant fusion.

Pattern: kARResidualRMSNormGroupFP8Quant = 6

Validates that the fused kernel produces the same result as the sequential
reference: NCCL allreduce -> residual add -> RMSNorm -> per-group FP8 quant
with UE8M0 packed scales (DeepGEMM layout).
"""

import math
import multiprocessing as mp
import socket
import struct
from typing import Any

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

MAX_TOKEN_NUM = 2048
GROUP_SIZE = 128
FP8_E4M3_MAX = 448.0


def _float_to_ue8m0_exponent(val: float) -> int:
    """Extract 8-bit IEEE biased exponent from a float (UE8M0 format)."""
    bits = struct.unpack("I", struct.pack("f", val))[0]
    return (bits >> 23) & 0xFF


def _reference_group_fp8_quant(
    norm_out: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, list[list[int]]]:
    """
    Reference per-token-group FP8 quantization with UE8M0 packed scales.

    Args:
        norm_out: [token_num, hidden_dim] float32 tensor (RMSNorm output)
        group_size: number of elements per quantization group (128)

    Returns:
        quant_ref: [token_num, hidden_dim] float8_e4m3fn tensor
        exponents: list[list[int]] — per-token, per-group UE8M0 exponent values
    """
    token_num, hidden_dim = norm_out.shape
    assert hidden_dim % group_size == 0
    groups_per_row = hidden_dim // group_size

    norm_f32 = norm_out.float()
    grouped = norm_f32.view(token_num, groups_per_row, group_size)

    # Compute per-group absmax
    absmax = grouped.abs().amax(dim=-1)  # [token_num, groups_per_row]

    # UE8M0 scale: round up to next power of 2
    raw_scale = absmax / FP8_E4M3_MAX
    raw_scale = raw_scale.clamp(min=1e-10)
    ue8m0_scale = torch.exp2(torch.ceil(torch.log2(raw_scale)))

    # Quantize
    inv_scale = 1.0 / ue8m0_scale  # [token_num, groups_per_row]
    scaled = grouped * inv_scale.unsqueeze(-1)
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quant_ref = scaled.view(token_num, hidden_dim).to(torch.float8_e4m3fn)

    # Extract exponents for each group
    exponents = []
    for t in range(token_num):
        row = []
        for g in range(groups_per_row):
            row.append(_float_to_ue8m0_exponent(ue8m0_scale[t, g].item()))
        exponents.append(row)

    return quant_ref, exponents


def _run_correctness_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    hidden_dim: int,
    distributed_init_port: int,
    gpu_offset: int = 0,
):
    device = torch.device(f"cuda:{rank + gpu_offset}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{distributed_init_port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=MAX_TOKEN_NUM,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

        token_nums = [1, 128, 512]
        use_oneshots = [True, None]
        rms_eps = 1e-5

        for token_num in token_nums:
            for use_oneshot in use_oneshots:
                if (
                    token_num < world_size
                    and use_oneshot is not None
                    and not use_oneshot
                ):
                    continue

                dist.barrier(group=group)
                torch.cuda.synchronize()

                groups_per_row = hidden_dim // GROUP_SIZE
                k_num_packed_sf_k = (groups_per_row + 3) // 4
                tma_aligned_mn = ((token_num + 3) // 4) * 4

                # Create input tensors
                allreduce_in = torch.randn(
                    token_num, hidden_dim, dtype=dtype, device=device
                )
                allreduce_in_clone = allreduce_in.clone()

                residual_in = torch.randn(
                    token_num, hidden_dim, dtype=dtype, device=device
                )
                residual_in_clone = residual_in.clone()

                rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

                # Create output tensors
                residual_out = torch.empty_like(allreduce_in)
                quant_out = torch.empty(
                    token_num, hidden_dim, dtype=torch.float8_e4m3fn, device=device
                )
                scale_out = torch.empty_strided(
                    (token_num, k_num_packed_sf_k),
                    (1, tma_aligned_mn),
                    device=device,
                    dtype=torch.int32,
                )
                # Zero-fill scale_out for padding bytes
                scale_out.zero_()

                # Run fused kernel
                comm.allreduce_fusion(
                    input=allreduce_in,
                    workspace=workspace,
                    pattern=comm.AllReduceFusionPattern.kARResidualRMSNormGroupFP8Quant,
                    residual_in=residual_in,
                    residual_out=residual_out,
                    quant_out=quant_out,
                    scale_out=scale_out,
                    rms_gamma=rms_gamma,
                    rms_eps=rms_eps,
                    group_size=GROUP_SIZE,
                    fp32_acc=True,
                    use_oneshot=use_oneshot,
                )
                torch.cuda.synchronize()

                # Compute reference:
                # 1. AllReduce via NCCL
                dist.all_reduce(allreduce_in_clone, group=group)
                ref_ar = allreduce_in_clone.float()

                # 2. Residual add
                ref_residual = ref_ar + residual_in_clone.float()

                # 3. RMSNorm
                variance = ref_residual.pow(2).mean(dim=-1, keepdim=True)
                ref_norm = ref_residual * torch.rsqrt(variance + rms_eps)
                ref_norm = rms_gamma.float() * ref_norm

                # 4. Per-token-group FP8 quant (Python reference)
                ref_quant, ref_exponents = _reference_group_fp8_quant(
                    ref_norm, GROUP_SIZE
                )

                # Check residual_out
                tolerance = 1e-1
                torch.testing.assert_close(
                    residual_out.float(),
                    ref_residual.to(dtype).float(),
                    atol=tolerance,
                    rtol=1e-2,
                )

                # Check quant_out: compare dequantized values.
                # The fused kernel's RMSNorm runs in bf16 (with fp32 acc)
                # while the reference uses pure fp32. This precision gap
                # can shift values across UE8M0 power-of-2 scale boundaries
                # and FP8 quantization buckets. Allow a small fraction of
                # large mismatches.
                fused_deq = quant_out.float()
                ref_deq = ref_quant.float()
                abs_diff = (fused_deq - ref_deq).abs()
                num_large = (abs_diff > 1.0).sum().item()
                total = abs_diff.numel()
                mismatch_ratio = num_large / total
                assert mismatch_ratio < 0.05, (
                    f"Quant mismatch ratio {mismatch_ratio:.4f} exceeds 5% "
                    f"(max abs diff={abs_diff.max().item()}) at "
                    f"token_num={token_num}, hidden_dim={hidden_dim}"
                )

                # Check packed scales at the logical group level.
                # Extract exponents from the strided kernel output tensor
                # by reading each int32 element's 4 packed bytes.
                # scale_out shape: (token_num, k_num_packed_sf_k),
                # strides: (1, tma_aligned_mn), dtype: int32
                scale_cpu = scale_out.cpu()
                num_bad = 0
                num_valid = 0
                for t in range(token_num):
                    for g in range(groups_per_row):
                        pack_idx = g // 4
                        pos = g % 4
                        # Read the int32 at [t, pack_idx]
                        packed_val = scale_cpu[t, pack_idx].item()
                        # Extract byte at position pos
                        fused_exp = (packed_val >> (pos * 8)) & 0xFF
                        ref_exp = ref_exponents[t][g]
                        if abs(fused_exp - ref_exp) > 1:
                            num_bad += 1
                        num_valid += 1
                bad_ratio = num_bad / max(num_valid, 1)
                assert bad_ratio < 0.05, (
                    f"Scale exponent mismatch >1 at {bad_ratio:.2%} of "
                    f"positions ({num_bad}/{num_valid}), "
                    f"token_num={token_num}, hidden_dim={hidden_dim}"
                )

                if rank == 0:
                    print(
                        f"  PASS: token_num={token_num}, hidden_dim={hidden_dim}, "
                        f"dtype={dtype}, use_oneshot={use_oneshot}"
                    )

    finally:
        dist.barrier(group=group)
        workspace.destroy()
        dist.destroy_process_group(group=group)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multi_process_parallel(
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=_run_correctness_worker,
            args=(world_size, i, dtype, hidden_dim, port, gpu_offset),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_dim", [1024, 4096])
def test_allreduce_rmsnorm_group_fp8_quant(world_size, dtype, hidden_dim):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")
    if hidden_dim % GROUP_SIZE != 0:
        pytest.skip(f"hidden_dim={hidden_dim} not divisible by group_size={GROUP_SIZE}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(
        f"\nTesting AR+RMSNorm+GroupFP8Quant: world_size={world_size}, "
        f"dtype={dtype}, hidden_dim={hidden_dim}"
    )
    _multi_process_parallel(world_size, dtype, hidden_dim)
    print(f"  All checks passed!")


if __name__ == "__main__":
    test_allreduce_rmsnorm_group_fp8_quant(2, torch.bfloat16, 1024)
    test_allreduce_rmsnorm_group_fp8_quant(2, torch.bfloat16, 4096)
