"""Expert-parallel group sizes of the SM107 (Rubin / VR200) MegaMoE kernels.

One rule for every SM107 MegaMoE entry point (the block-scaled workspaces and the isolated Kimi K3 backend):

* ``num_experts % world_size == 0``: every rank owns ``num_experts // world_size`` whole experts. The kernels
  have no logical-to-physical expert map, so an uneven split (896 experts at EP12 / EP20 / EP24) is not
  expressible.
* ``world_size <= SM107_MAX_EP_WORLD_SIZE`` (32): the decode router's landing path gives one warp lane to each
  source rank (``_MetadataPushRouter.decode_landing_max_world_size``; the token_in ring table and its ballot are
  one 32-lane warp). Above 32 every batch size would silently fall back to the two-round push router, which
  no EP configuration exercises, and its helper CTA holds a ``world_size x num_experts`` int32 size matrix in
  shared memory (896 experts: 119 KiB at 32 ranks, over the 227 KiB budget at 64).
* ``world_size`` is one of the single-node sizes (1 and 2 for the generic workspaces, 1 for Kimi K3) or a
  multiple of ``SM107_GPUS_PER_NODE`` (4): a multi-GPU EP group is made of whole 4-GPU VR200 nodes inside one
  NVL72 NVLink domain.

For Kimi K3 / K3.1 (896 experts) this admits EP 1, 4, 8, 16, 28 and 32 (and 2 for the generic workspaces).
"""

from __future__ import annotations

from typing import Optional, Tuple

__all__ = [
    "SM107_GPUS_PER_NODE",
    "SM107_MAX_EP_WORLD_SIZE",
    "sm107_ep_world_size_error",
    "sm107_supported_ep_world_sizes",
]

SM107_MAX_EP_WORLD_SIZE = 32
SM107_GPUS_PER_NODE = 4
_GENERIC_SINGLE_NODE_SIZES = (1, 2)


def sm107_supported_ep_world_sizes(
    num_experts: int, *, single_node_sizes: Tuple[int, ...] = _GENERIC_SINGLE_NODE_SIZES
) -> Tuple[int, ...]:
    """Every EP world size the SM107 MegaMoE kernels support for ``num_experts`` experts."""
    return tuple(
        size
        for size in range(1, SM107_MAX_EP_WORLD_SIZE + 1)
        if num_experts % size == 0
        and (size in single_node_sizes or size % SM107_GPUS_PER_NODE == 0)
    )


def sm107_ep_world_size_error(
    num_experts: int,
    world_size: int,
    *,
    single_node_sizes: Tuple[int, ...] = _GENERIC_SINGLE_NODE_SIZES,
) -> Optional[str]:
    """Why ``world_size`` ranks cannot share ``num_experts`` experts on SM107; None when they can."""
    if type(world_size) is not int or type(num_experts) is not int or num_experts <= 0:
        return f"world_size and num_experts must be positive integers, got {world_size!r} and {num_experts!r}"
    supported = sm107_supported_ep_world_sizes(num_experts, single_node_sizes=single_node_sizes)
    if world_size in supported:
        return None
    sizes = " or ".join(str(s) for s in single_node_sizes)
    return (
        f"SM107 MegaMoE does not support EP world_size={world_size} with {num_experts} experts: world_size must "
        f"divide num_experts (whole experts per rank), be at most {SM107_MAX_EP_WORLD_SIZE} (one decode-router "
        f"lane per source rank) and be {sizes} or a multiple of {SM107_GPUS_PER_NODE} (whole 4-GPU VR200 nodes); "
        f"supported for {num_experts} experts: {', '.join(str(s) for s in supported) or 'none'}"
    )
