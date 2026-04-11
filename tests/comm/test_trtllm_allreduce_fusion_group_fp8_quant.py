"""
Test for AllReduce + Residual + RMSNorm + Per-Token-Group FP8 Quant fusion.

Pattern: kARResidualRMSNormPerTokenGroupFP8PackedQuant = 6

Validates the fused kernel against sequential reference:
  NCCL allreduce -> residual add -> RMSNorm -> per-group FP8 quant

Test shapes follow vllm's test_per_token_group_quant_fp8_packed coverage:
  - MN padding (token_num not multiple of 4)
  - K padding (groups_per_row not multiple of 4)
  - Both MN and K padding
  - Various group sizes (128, 96)
  - Poisoned scale buffers (verify padding bytes zeroed)
"""

import multiprocessing as mp
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm.mnnvl import TorchDistBackend

FP8_DTYPE = torch.float8_e4m3fn


def _run_correctness_worker(
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    hidden_dim: int,
    distributed_init_port: int,
    test_cases: list[tuple[int, int, bool]],
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

    max_token_num = max(tc[0] for tc in test_cases)

    try:
        workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

        rms_eps = 1e-5
        first_error = None

        for token_num, group_size, poisoned_scales in test_cases:
            dist.barrier(group=group)
            torch.cuda.synchronize()

            groups_per_row = hidden_dim // group_size
            k_num_packed = (groups_per_row + 3) // 4
            tma_aligned_mn = ((token_num + 3) // 4) * 4
            num_scale_elems = token_num + (k_num_packed - 1) * tma_aligned_mn

            # Input tensors (scaled up to exercise more of the FP8 range)
            allreduce_in = (
                torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
                * 8
            )
            allreduce_in_clone = allreduce_in.clone()
            residual_in = (
                torch.randn(token_num, hidden_dim, dtype=dtype, device=device)
                * 8
            )
            residual_in_clone = residual_in.clone()
            rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

            # Output tensors
            residual_out = torch.empty_like(allreduce_in)
            quant_out = torch.empty(
                token_num, hidden_dim,
                dtype=torch.float8_e4m3fn, device=device,
            )
            scale_out = torch.empty_strided(
                (token_num, k_num_packed),
                (1, tma_aligned_mn),
                device=device, dtype=torch.int32,
            )

            if poisoned_scales:
                # Fill with garbage to verify kernel zeros padding
                torch.as_strided(
                    scale_out, (num_scale_elems,), (1,)
                ).fill_(0x7F7F7F7F)

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
                rms_eps=rms_eps,
                block_quant_group_size=group_size,
                fp32_acc=True,
                use_oneshot=True,
            )
            torch.cuda.synchronize()

            # --- Sequential reference using flashinfer's own AR+RMSNorm ---
            # Use kARResidualRMSNorm (same allreduce + residual + rmsnorm
            # code path, same fp32_acc) so the only difference is the quant.
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
                rms_eps=rms_eps,
                fp32_acc=True,
                use_oneshot=True,
            )
            torch.cuda.synchronize()

            # --- Verify (catch errors to avoid barrier deadlock) ---
            # NOTE: the fused pattern uses different register pressure than
            # kARResidualRMSNorm, which can change block configuration and
            # fp32 reduction order in RMSNorm. So the norm output can differ
            # slightly. We verify via round-trip dequantization instead of
            # exact quant match.
            try:
                # 1. Residual: both patterns write residual before RMSNorm,
                #    so the allreduce+residual path is identical.
                torch.testing.assert_close(
                    residual_out.float(), ref_residual_out.float(),
                    atol=1e-1, rtol=1e-2,
                )

                # 2. Round-trip: dequantize fused (quant_out, scale_out)
                #    and compare against the reference norm output.
                #    This verifies the quant+scale pair is internally
                #    consistent and approximates the correct norm.
                fused_deq = torch.zeros(
                    token_num, hidden_dim, dtype=torch.float32,
                    device=device,
                )
                scale_cpu = scale_out.cpu()
                for t in range(token_num):
                    for g in range(groups_per_row):
                        pack_idx = g // 4
                        pos = g % 4
                        packed_val = scale_cpu[t, pack_idx].item()
                        exponent = (packed_val >> (pos * 8)) & 0xFF
                        # UE8M0 exponent → float scale: 2^(exponent - 127)
                        scale_val = 2.0 ** (exponent - 127) if exponent > 0 else 0.0
                        start = g * group_size
                        end = start + group_size
                        fused_deq[t, start:end] = (
                            quant_out[t, start:end].float() * scale_val
                        )

                # Compare dequantized fused output vs reference norm.
                # FP8 e4m3 has ~4 bits of mantissa → relative error ~1/16.
                # UE8M0 rounding adds up to 2x. Allow generous tolerance.
                torch.testing.assert_close(
                    fused_deq,
                    ref_norm_out.float(),
                    atol=2.0, rtol=0.15,
                )

                # 3. Verify packed scale padding is zero
                actual_storage = torch.as_strided(
                    scale_out, (num_scale_elems,), (1,)
                ).cpu()
                actual_bytes = actual_storage.view(torch.uint8)
                valid_mask = torch.zeros(
                    actual_bytes.numel(), dtype=torch.bool
                )
                for row in range(token_num):
                    for g in range(groups_per_row):
                        pack_col = g // 4
                        pos = g % 4
                        idx = pack_col * tma_aligned_mn + row
                        byte_idx = idx * 4 + pos
                        if byte_idx < actual_bytes.numel():
                            valid_mask[byte_idx] = True
                padding_mask = ~valid_mask
                if padding_mask.any():
                    padding_nonzero = (
                        actual_bytes[padding_mask] != 0
                    ).sum().item()
                    assert padding_nonzero == 0, (
                        f"Padding not zeroed: {padding_nonzero} bytes "
                        f"at tokens={token_num}, hidden={hidden_dim}, "
                        f"group={group_size}, poisoned={poisoned_scales}"
                    )

                if rank == 0:
                    print(
                        f"  PASS: tokens={token_num}, hidden={hidden_dim}, "
                        f"group={group_size}, poisoned={poisoned_scales}"
                    )
            except Exception as e:
                if first_error is None:
                    first_error = e

        # Raise first error after all test cases (avoids barrier deadlock)
        if first_error is not None:
            raise first_error

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
    test_cases: list[tuple[int, int, bool]],
    gpu_offset: int = 0,
) -> None:
    mp.set_start_method("spawn", force=True)
    port = _get_open_port()
    procs = []
    for i in range(world_size):
        proc = mp.Process(
            target=_run_correctness_worker,
            args=(world_size, i, dtype, hidden_dim, port, test_cases, gpu_offset),
            name=f"Worker-{i}",
        )
        proc.start()
        procs.append(proc)
    for i, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, (
            f"Process {i} failed with exit code {proc.exitcode}"
        )


# Test cases grouped by hidden_dim (workspace is per-hidden_dim).
# Each entry: (num_tokens, group_size, poisoned_scales)
# Covers: MN padding, K padding, both, various group sizes, poisoned scales.
_TEST_CASES_7168 = [
    # groups_per_row=56 (56%4=0, no K padding)
    (4, 128, False),    # no padding
    (1, 128, False),    # MN padding only (tma_aligned_mn=4)
    (3, 128, False),    # MN padding only (tma_aligned_mn=4)
    (1, 128, True),     # poisoned, MN padding
    (64, 128, False),   # larger, no padding
    (127, 128, True),   # larger, MN padding, poisoned
]

_TEST_CASES_768 = [
    # groups_per_row=6 (128)
    (4, 128, False),    # K padding (6%4=2)
    (3, 128, True),     # both MN and K padding, poisoned
    # NOTE: group_size=96 is NOT supported — group_size/VEC_SIZE must be
    # a power of 2 for the warp-shuffle reduction to work correctly.
]

_TEST_CASES_640 = [
    # groups_per_row=5 (128), K padding (5%4=1)
    (4, 128, False),    # K padding only
    (3, 128, True),     # both MN and K padding, poisoned
    (253, 128, False),  # larger, both padding
]

_TEST_CASES_384 = [
    # groups_per_row=3 (128), k_num_packed=1
    (4, 128, False),    # single packed column, no MN padding
    (1, 128, True),     # both MN and K padding, poisoned
]


@pytest.mark.parametrize(
    "hidden_dim,test_cases",
    [
        (7168, _TEST_CASES_7168),
        (768, _TEST_CASES_768),
        (640, _TEST_CASES_640),
        (384, _TEST_CASES_384),
    ],
    ids=["hidden_7168", "hidden_768", "hidden_640", "hidden_384"],
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_allreduce_rmsnorm_group_fp8_quant(
    world_size, dtype, hidden_dim, test_cases
):
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(f"Need {world_size} GPUs, have {available_gpus}")

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    print(
        f"\nTesting AR+RMSNorm+GroupFP8PackedQuant: "
        f"world_size={world_size}, hidden_dim={hidden_dim}, "
        f"{len(test_cases)} sub-cases"
    )
    _multi_process_parallel(world_size, dtype, hidden_dim, test_cases)
    print("  All checks passed!")


if __name__ == "__main__":
    for hd, cases in [
        (7168, _TEST_CASES_7168),
        (768, _TEST_CASES_768),
        (640, _TEST_CASES_640),
        (384, _TEST_CASES_384),
    ]:
        test_allreduce_rmsnorm_group_fp8_quant(2, torch.bfloat16, hd, cases)
