"""
Test for AllReduce + Residual + RMSNorm + Per-Token-Group FP8 Quant fusion.

Pattern: kARResidualRMSNormPerTokenGroupFP8PackedQuant = 6

Validates the fused kernel against sequential reference:
  NCCL allreduce -> residual add -> RMSNorm -> per-group FP8 quant

Test shapes cover:
  - MN padding (token_num not multiple of 4)
  - K padding (groups_per_row not multiple of 4)
  - Both MN and K padding
  - Various group sizes
  - Scale buffer always pre-filled with garbage to verify padding zeroed
"""

import multiprocessing as mp
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Reference quantization helpers
# ---------------------------------------------------------------------------


def _scale_storage_size(token_num: int, groups_per_row: int) -> int:
    """Number of int32 elements in the TMA-aligned packed scale storage."""
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    return token_num + (k_num_packed - 1) * tma_aligned_mn


def _unpack_scales(
    packed_scales: torch.Tensor, token_num: int, groups_per_row: int
) -> np.ndarray:
    """Unpack UE8M0 exponent bytes from TMA-aligned packed int32 layout.

    Returns uint8 array of shape [token_num, groups_per_row].
    """
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    num_elems = _scale_storage_size(token_num, groups_per_row)
    raw_bytes = (
        torch.as_strided(packed_scales, (num_elems,), (1,))
        .cpu()
        .view(torch.uint8)
        .numpy()
    )
    exponents = np.empty((token_num, groups_per_row), dtype=np.uint8)
    for t in range(token_num):
        for g in range(groups_per_row):
            elem_idx = (g // 4) * tma_aligned_mn + t
            exponents[t, g] = raw_bytes[elem_idx * 4 + g % 4]
    return exponents


def _pack_scales(
    exponents: np.ndarray,
    token_num: int,
    groups_per_row: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack UE8M0 exponent bytes into TMA-aligned int32 layout with zero padding."""
    k_num_packed = (groups_per_row + 3) // 4
    tma_aligned_mn = ((token_num + 3) // 4) * 4
    num_elems = _scale_storage_size(token_num, groups_per_row)

    buf = np.zeros(num_elems * 4, dtype=np.uint8)
    for t in range(token_num):
        for g in range(groups_per_row):
            elem_idx = (g // 4) * tma_aligned_mn + t
            buf[elem_idx * 4 + g % 4] = exponents[t, g]

    flat = torch.from_numpy(buf.view(np.int32).copy()).to(device)
    out = torch.empty_strided(
        (token_num, k_num_packed),
        (1, tma_aligned_mn),
        device=device,
        dtype=torch.int32,
    )
    torch.as_strided(out, (num_elems,), (1,)).copy_(flat[:num_elems])
    return out


def ref_per_token_group_quant_fp8_packed(
    x: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference per-token-group FP8 quantization with UE8M0 packed scales.

    Mirrors the fused kernel:
      1. Per-group absmax
      2. UE8M0 scale = 2^ceil(log2(max(absmax / 448, 1e-10)))
      3. Quantize: clamp(x / scale, -448, 448) -> fp8_e4m3
      4. Pack exponents 4-per-int32 in TMA-aligned column-major layout
    """
    token_num, hidden_dim = x.shape
    groups_per_row = hidden_dim // group_size

    x_grouped = x.float().reshape(token_num, groups_per_row, group_size)

    # Per-group absmax -> UE8M0 power-of-2 scale
    absmax = x_grouped.abs().amax(dim=-1)
    y_s = torch.exp2(torch.ceil(torch.log2((absmax / FP8_E4M3_MAX).clamp(min=1e-10))))

    # Quantize to FP8 e4m3 using division (matches kernel)
    q = (x_grouped / y_s.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quant_out = q.reshape(token_num, hidden_dim).to(torch.float8_e4m3fn)

    # Extract UE8M0 exponent bytes via numpy reinterpret
    exponents = ((y_s.cpu().numpy().view(np.uint32) >> 23) & 0xFF).astype(np.uint8)

    return quant_out, _pack_scales(exponents, token_num, groups_per_row, x.device)


def ref_quantize_fp8_with_packed_scales(
    x: torch.Tensor, packed_scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Quantize x to FP8 e4m3 using pre-computed packed UE8M0 scales.

    Isolates the FP8 cast from the scale computation: given the same
    scale, do fused and reference produce the same FP8 values?
    """
    token_num, hidden_dim = x.shape
    groups_per_row = hidden_dim // group_size

    exponents = _unpack_scales(packed_scales, token_num, groups_per_row)
    scales = np.where(
        exponents > 0,
        np.ldexp(1.0, exponents.astype(np.int32) - 127),
        0.0,
    ).astype(np.float32)
    scales_t = torch.from_numpy(scales).to(x.device)

    x_grouped = x.float().reshape(token_num, groups_per_row, group_size)
    q = (x_grouped / scales_t.unsqueeze(-1)).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    return q.reshape(token_num, hidden_dim).to(torch.float8_e4m3fn)


# ---------------------------------------------------------------------------
# Multi-process test worker
# ---------------------------------------------------------------------------


def _run_correctness_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    hidden_dim: int,
    token_num: int,
    group_size: int,
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
            max_token_num=token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

        groups_per_row = hidden_dim // group_size
        k_num_packed = (groups_per_row + 3) // 4
        tma_aligned_mn = ((token_num + 3) // 4) * 4
        num_scale_elems = _scale_storage_size(token_num, groups_per_row)

        # Input tensors (scaled up to exercise more of the FP8 range)
        allreduce_in = (
            torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
        )
        allreduce_in_clone = allreduce_in.clone()
        residual_in = torch.randn(token_num, hidden_dim, dtype=dtype, device=device) * 8
        residual_in_clone = residual_in.clone()
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

        # Output tensors
        residual_out = torch.empty_like(allreduce_in)
        quant_out = torch.empty(
            token_num,
            hidden_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        scale_out = torch.empty_strided(
            (token_num, k_num_packed),
            (1, tma_aligned_mn),
            device=device,
            dtype=torch.int32,
        )

        # Fill with garbage to verify kernel zeroes padding bytes
        torch.as_strided(scale_out, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)

        # Run fused kernel
        comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=comm.AllReduceFusionPattern.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
            residual_in=residual_in,
            residual_out=residual_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=rms_gamma,
            rms_eps=1e-5,
            block_quant_group_size=group_size,
            fp32_acc=True,
            use_oneshot=True,
        )
        torch.cuda.synchronize()

        # Reference: allreduce + residual + rmsnorm (no quant)
        ref_residual_out = torch.empty_like(allreduce_in_clone)
        ref_norm_out = torch.empty_like(allreduce_in_clone)
        comm.allreduce_fusion(
            input=allreduce_in_clone,
            workspace=workspace,
            pattern=comm.AllReduceFusionPattern.kARResidualRMSNorm,
            residual_in=residual_in_clone,
            residual_out=ref_residual_out,
            norm_out=ref_norm_out,
            rms_gamma=rms_gamma,
            rms_eps=1e-5,
            fp32_acc=True,
            use_oneshot=True,
        )
        torch.cuda.synchronize()

        # --- Verify ---

        # 1. Residual add
        torch.testing.assert_close(
            residual_out.float(),
            ref_residual_out.float(),
            atol=1e-1,
            rtol=1e-2,
        )

        # 2. Quantization: apply reference quant to ref_norm_out
        _, ref_scale = ref_per_token_group_quant_fp8_packed(
            ref_norm_out,
            group_size,
        )

        # 2a. Scales: compare exponent bytes. atol=1 because the
        # fused and reference RMSNorm can differ slightly, which
        # may shift the group absmax across a UE8M0 power-of-2
        # boundary, changing the exponent by 1.
        fused_scale_bytes = (
            torch.as_strided(scale_out, (num_scale_elems,), (1,))
            .cpu()
            .view(torch.uint8)
        )
        ref_scale_bytes = (
            torch.as_strided(ref_scale, (num_scale_elems,), (1,))
            .cpu()
            .view(torch.uint8)
        )
        torch.testing.assert_close(
            fused_scale_bytes.to(torch.int16),
            ref_scale_bytes.to(torch.int16),
            atol=1,
            rtol=0,
        )

        # Padding bytes must be exactly zero (ref is zero by
        # construction; fused buffer was pre-filled with garbage).
        padding_mask = ref_scale_bytes == 0
        assert (fused_scale_bytes[padding_mask] == 0).all(), (
            "Scale padding bytes not zeroed"
        )

        # 2b. FP8 values: re-quantize ref_norm_out with the fused
        # kernel's scales to isolate the FP8 cast.
        ref_quant_fused_scale = ref_quantize_fp8_with_packed_scales(
            ref_norm_out,
            scale_out,
            group_size,
        )
        torch.testing.assert_close(
            quant_out.float(),
            ref_quant_fused_scale.float(),
            atol=torch.finfo(torch.float8_e4m3fn).tiny,
            rtol=torch.finfo(torch.float8_e4m3fn).eps,
        )

    finally:
        dist.barrier(group=group)
        workspace.destroy()
        dist.destroy_process_group(group=group)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _multi_process_parallel(
    world_size: int,
    dtype: torch.dtype,
    hidden_dim: int,
    token_num: int,
    group_size: int,
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=_run_correctness_worker,
            args=(
                world_size,
                i,
                dtype,
                hidden_dim,
                token_num,
                group_size,
                port,
                gpu_offset,
            ),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, f"Process {i} failed with exit code {proc.exitcode}"


@pytest.mark.parametrize(
    "hidden_dim,token_num,group_size",
    [
        # ===================== group_size=128 =====================
        # hidden=7168, groups_per_row=56 (56%4=0, no K padding)
        (7168, 4, 128),  # no padding (mn%4=0, groups%4=0)
        (7168, 1, 128),  # MN padding only (tma_aligned_mn=4)
        (7168, 3, 128),  # MN padding only (tma_aligned_mn=4)
        (7168, 64, 128),  # larger, no padding
        (7168, 127, 128),  # larger, MN padding
        (7168, 512, 128),  # large token count (triggers twoshot)
        # hidden=768, groups_per_row=6 (6%4=2, K padding)
        (768, 4, 128),  # K padding only
        (768, 3, 128),  # both MN and K padding
        (768, 1, 128),  # MN + K padding, single token
        # hidden=640, groups_per_row=5 (5%4=1, K padding)
        (640, 4, 128),  # K padding only
        (640, 3, 128),  # both MN and K padding
        (640, 253, 128),  # larger, both padding
        # hidden=384, groups_per_row=3, k_num_packed=1
        (384, 4, 128),  # single packed column, no MN padding
        (384, 1, 128),  # both MN and K padding
        # hidden=256, groups_per_row=2, k_num_packed=1
        (256, 4, 128),  # 2 groups per row, no padding
        (256, 1, 128),  # 2 groups, MN padding
        # ===================== group_size=64 ======================
        # hidden=7168, groups_per_row=112 (112%4=0, no K padding)
        (7168, 4, 64),  # no padding
        (7168, 3, 64),  # MN padding
        (7168, 512, 64),  # large token count (triggers twoshot)
        # hidden=640, groups_per_row=10 (10%4=2, K padding)
        (640, 4, 64),  # K padding only
        (640, 3, 64),  # both MN and K padding
        # hidden=384, groups_per_row=6 (6%4=2, K padding)
        (384, 4, 64),  # K padding only
        (384, 1, 64),  # both MN and K padding
        # hidden=192, groups_per_row=3, k_num_packed=1
        (192, 4, 64),  # single packed column, no MN padding
        (192, 1, 64),  # both MN and K padding
    ],
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_allreduce_rmsnorm_group_fp8_quant(
    world_size,
    dtype,
    hidden_dim,
    token_num,
    group_size,
):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    _multi_process_parallel(world_size, dtype, hidden_dim, token_num, group_size)


if __name__ == "__main__":
    for args in test_allreduce_rmsnorm_group_fp8_quant.pytestmark[0].args[1]:
        test_allreduce_rmsnorm_group_fp8_quant(2, torch.bfloat16, *args)
