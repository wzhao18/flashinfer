"""Die-local placement of expert weights on multi-die GPUs (SM107 / VR200 locality domains).

VR200 is two dies, each with its own HBM stacks and L2. A default allocation is striped over both dies, so half of
every weight stream crosses the die-to-die stitch; reads that stay on the reading SM's die run at the die-local rate
(measured on our nodes with ``e6_ldbw``: 17.3 TB/s die-local vs 12.7 TB/s striped vs 7.0 TB/s all-remote).

``localize_expert_major`` moves an expert-major weight (or scale-factor) tensor into one virtual range whose lower
half of experts is backed by locality domain 0 and whose upper half by locality domain 1 (``cuMemCreate`` with
``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN``). The bytes, shape, strides and dtype are unchanged, so the kernel
reads exactly the same values; only the physical placement changes. The MegaMoE scheduler then lets the clusters of
die 0 claim work from the lowest expert upward and the clusters of die 1 from the highest expert downward
(``die_split`` in ``PhaseInterleavedFc12Scheduler``).

On a device with one locality domain every function here is a no-op.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

__all__ = [
    "expert_split_index",
    "locality_domain_count",
    "localize_expert_major",
    "localize_weight_pair",
    "sm_locality_domain_map",
]


def _driver():
    try:
        from cuda.bindings import driver
    except ImportError:  # pragma: no cover - cuda-python is a CuTe DSL dependency
        return None
    if not hasattr(driver.CUmemLocationType, "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN"):
        return None
    return driver


def _check(result):
    error = result[0]
    if int(error) != 0:
        raise RuntimeError(f"CUDA driver call failed: {error}")
    return result[1] if len(result) == 2 else result[1:]


@functools.lru_cache(maxsize=None)
def locality_domain_count(device_index: int) -> int:
    """Number of locality domains (dies) of a CUDA device; 1 when unknown."""
    driver = _driver()
    if driver is None:
        return 1
    try:
        torch.cuda.init()
        device = _check(driver.cuDeviceGet(device_index))
        return int(
            _check(
                driver.cuDeviceGetAttribute(
                    driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device
                )
            )
        )
    except Exception:
        return 1


def expert_split_index(experts: int) -> int:
    """First expert placed on locality domain 1 (experts below it live on domain 0)."""
    return experts // 2


class _LocalizedAllocation:
    """Owns one reserved virtual range mapped onto two locality-domain physical allocations."""

    def __init__(self, device_index: int, nbytes: int, split_bytes: int) -> None:
        driver = _driver()
        self._driver = driver
        self._handles = []
        self._va = 0
        self._size = 0
        props = []
        for domain in (0, 1):
            prop = driver.CUmemAllocationProp()
            prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
            prop.location.localized.deviceId = device_index
            prop.location.localized.localityDomainId = domain
            props.append(prop)
        granularity = max(
            int(
                _check(
                    driver.cuMemGetAllocationGranularity(
                        prop, driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
                    )
                )
            )
            for prop in props
        )
        size = -(-nbytes // granularity) * granularity
        split = -(-split_bytes // granularity) * granularity
        split = min(max(split, granularity), size - granularity)
        try:
            self._va = int(_check(driver.cuMemAddressReserve(size, granularity, 0, 0)))
            self._size = size
            for domain, (offset, length) in enumerate(((0, split), (split, size - split))):
                handle = _check(driver.cuMemCreate(length, props[domain], 0))
                self._handles.append((handle, offset, length))
                _check(driver.cuMemMap(self._va + offset, length, 0, handle, 0))
            access = driver.CUmemAccessDesc()
            access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = device_index
            access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check(driver.cuMemSetAccess(self._va, size, [access], 1))
        except Exception:
            self.release()
            raise
        self.split_bytes = split

    @property
    def __cuda_array_interface__(self):
        return {"shape": (self._size,), "typestr": "|u1", "data": (self._va, False), "version": 3, "stream": None}

    def release(self) -> None:
        driver = self._driver
        for handle, offset, length in self._handles:
            try:
                driver.cuMemUnmap(self._va + offset, length)
                driver.cuMemRelease(handle)
            except Exception:  # pragma: no cover - interpreter shutdown
                pass
        self._handles = []
        if self._va:
            try:
                driver.cuMemAddressFree(self._va, self._size)
            except Exception:  # pragma: no cover
                pass
            self._va = 0

    def __del__(self) -> None:
        self.release()


def localize_expert_major(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` with experts [0, E/2) on locality domain 0 and [E/2, E) on domain 1.

    ``tensor`` must be a dense CUDA tensor whose dim 0 (experts) has the largest stride, as the SM107 transformed
    weights and scale planes are (possibly a permuted view of a contiguous expert-major buffer). The result has the
    same dtype, shape, strides and bytes. Returns ``tensor`` unchanged on single-domain devices.
    """
    if not tensor.is_cuda or tensor.dim() < 1 or tensor.shape[0] < 2:
        return tensor
    device_index = tensor.device.index if tensor.device.index is not None else torch.cuda.current_device()
    if locality_domain_count(device_index) < 2:
        return tensor
    experts = tensor.shape[0]
    element_bytes = tensor.element_size()
    nbytes = tensor.numel() * element_bytes
    expert_elements = tensor.numel() // experts
    if tensor.stride(0) != expert_elements or max(tensor.stride()) != tensor.stride(0):
        raise ValueError("die-local placement needs a dense expert-major tensor.")
    span = torch.empty(0, dtype=torch.uint8, device=tensor.device)
    span.set_(tensor.untyped_storage(), tensor.storage_offset() * element_bytes, (nbytes,), (1,))
    split_bytes = expert_split_index(experts) * expert_elements * element_bytes
    allocation = _LocalizedAllocation(device_index, nbytes, split_bytes)
    target = torch.as_tensor(allocation, device=tensor.device)
    target[:nbytes].copy_(span)
    return target[:nbytes].view(tensor.dtype).as_strided(tuple(tensor.shape), tuple(tensor.stride()))


def localize_weight_pair(pair):
    """``(weight, scale)`` -> both localized (see ``localize_expert_major``)."""
    weight, scale = pair
    return localize_expert_major(weight), localize_expert_major(scale)


def pointer_locality_domain(tensor: torch.Tensor, byte_offset: int = 0) -> Optional[int]:
    """Locality domain ordinal backing ``tensor.data_ptr() + byte_offset`` (None when not localized/unknown)."""
    import ctypes

    driver = _driver()
    if driver is None:
        return None
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        ordinal = ctypes.c_int(-2)
        attribute = int(driver.CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)
        status = libcuda.cuPointerGetAttribute(
            ctypes.byref(ordinal), ctypes.c_int(attribute), ctypes.c_uint64(tensor.data_ptr() + byte_offset)
        )
    except Exception:
        return None
    if status != 0 or ordinal.value < 0:
        return None
    return ordinal.value


# Per-SM locality domain, measured once per device: one CUDA block per SM slot times dependent loads into a
# domain-0 and a domain-1 allocation (other-die loads cross the die-to-die stitch: ~1300 vs ~900 cycles on VR200,
# 200/200 agreement with green-context locality-domain SM splits). The PTX register %locality_domain_id that
# would give this directly needs a newer ptxas than the 13.4 toolchain.
_SM_PROBE_SOURCE = r"""
extern "C" __global__ void sm_locality_probe(const unsigned* __restrict__ domain0, const unsigned* __restrict__ domain1,
                                             unsigned long long words, unsigned* out) {
  if (threadIdx.x != 0) return;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  const unsigned* buffers[2] = {domain0, domain1};
  unsigned latency[2];
  unsigned sink = 0;
  for (int d = 0; d < 2; ++d) {
    unsigned long long idx = ((unsigned long long)blockIdx.x * 8178031ull + (unsigned long long)d * 977ull) % words;
    long long begin = clock64();
    for (int i = 0; i < 32; ++i) {
      unsigned v;
      asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(v) : "l"(buffers[d] + idx));
      sink += v;
      idx = (idx + 262147ull + (v & 1u)) % words;
    }
    latency[d] = (unsigned)((clock64() - begin) / 32);
  }
  out[3 * blockIdx.x + 0] = sm;
  out[3 * blockIdx.x + 1] = latency[0];
  out[3 * blockIdx.x + 2] = latency[1] + (sink == 0x9e3779b9u ? 1u : 0u);
}
"""

SM_LOCALITY_MAP_SIZE = 1024


@functools.lru_cache(maxsize=None)
def _sm_locality_domain_map_host(device_index: int) -> tuple:
    import ctypes

    from cuda.bindings import nvrtc

    driver = _driver()
    program = _check(nvrtc.nvrtcCreateProgram(_SM_PROBE_SOURCE.encode(), b"sm_locality_probe.cu", 0, [], []))
    try:
        compiled = nvrtc.nvrtcCompileProgram(program, 1, [b"--gpu-architecture=compute_100"])
        if int(compiled[0]) != 0:
            log_size = _check(nvrtc.nvrtcGetProgramLogSize(program))
            log = b" " * log_size
            nvrtc.nvrtcGetProgramLog(program, log)
            raise RuntimeError(f"sm_locality_probe compile failed: {log.decode(errors='replace')}")
        ptx_size = _check(nvrtc.nvrtcGetPTXSize(program))
        ptx = b" " * ptx_size
        _check(nvrtc.nvrtcGetPTX(program, ptx))
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    module = _check(driver.cuModuleLoadData(ptx))
    try:
        function = _check(driver.cuModuleGetFunction(module, b"sm_locality_probe"))
        half = 256 << 20
        allocation = _LocalizedAllocation(device_index, 2 * half, half)
        base = allocation.__cuda_array_interface__["data"][0]
        split = allocation.split_bytes
        sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
        blocks = 8 * sm_count
        out = torch.zeros(3 * blocks, dtype=torch.int32, device=f"cuda:{device_index}")
        words = (split // 4) - 1024
        stream = torch.cuda.current_stream(device_index)
        args = (
            (base, base + split, words, out.data_ptr()),
            (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_void_p),
        )
        _check(driver.cuLaunchKernel(function, blocks, 1, 1, 32, 1, 1, 0, stream.cuda_stream, args, 0))
        stream.synchronize()
        samples = out.view(blocks, 3).cpu().tolist()
        allocation.release()
    finally:
        driver.cuModuleUnload(module)
    sums = {}
    for sm, latency0, latency1 in samples:
        entry = sums.setdefault(sm, [0, 0])
        entry[0] += latency0
        entry[1] += latency1
    domains = [0] * SM_LOCALITY_MAP_SIZE
    for sm, (latency0, latency1) in sums.items():
        if 0 <= sm < SM_LOCALITY_MAP_SIZE and latency1 < latency0:
            domains[sm] = 1
    return tuple(domains)


def sm_locality_domain_map(device_index: int) -> torch.Tensor:
    """uint8 CUDA tensor ``[SM_LOCALITY_MAP_SIZE]``: locality domain (die) of each ``%smid`` (0 for unseen SMs)."""
    cache = _SM_MAP_TENSORS
    if device_index not in cache:
        domains = _sm_locality_domain_map_host(device_index)
        cache[device_index] = torch.tensor(domains, dtype=torch.uint8, device=f"cuda:{device_index}")
    return cache[device_index]


_SM_MAP_TENSORS: dict = {}
