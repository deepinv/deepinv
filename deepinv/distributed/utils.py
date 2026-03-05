from __future__ import annotations

from typing import Sequence
import itertools
import torch

Index = tuple[slice, ...]


def tiling_reduce_fn(
    out_local: torch.Tensor,
    local_pairs: list[tuple[int, torch.Tensor]],
    global_metadata: dict,
) -> None:
    r"""
    Default reduction function for tiling strategy (N-dimensional).

    This function fills the `out_local` tensor with the processed patches from `local_pairs`,
    using the metadata to determine where each patch should be placed in the output tensor.
    `out_local` is modified in place.

    :param torch.Tensor out_local: the output tensor to fill (should be initialized to zeros).
    :param list[tuple[int, torch.Tensor]] local_pairs: list of (global_index, processed_tensor) pairs from this rank.
    :param dict global_metadata: metadata from the splitting strategy containing crop_slices and target_slices.
    """
    crop_slices = global_metadata["crop_slices"]
    target_slices = global_metadata["target_slices"]

    for idx, processed_tensor in local_pairs:
        c_sl = crop_slices[idx]
        t_sl = target_slices[idx]
        # Extract the valid part of the processed patch and place it in the output
        out_local[t_sl] = processed_tensor[c_sl]


def tiling_splitting_strategy(
    img_size: Sequence[int],
    *,
    patch_size: int | tuple[int, ...],
    overlap: int | tuple[int, ...] = 0,
    stride: int | tuple[int, ...] | None = None,
    tiling_dims: int | tuple[int, ...] | None = None,
    pad_mode: str = "reflect",
) -> tuple[list[Index], dict]:
    r"""
    Generalized uniform-batching tiler with global padding for N dimensions.

    Produces:
        - global_slices: list of slices into the *padded* tensor,
        - crop_slices: slices relative to the *padded window* that grab only the inner patch portion (removing halo),
        - target_slices: slices into the *original-shape* output where that patch should be placed,
        - global_padding: padding tuple for F.pad to be applied to the whole image before slicing.

    :param Sequence[int] img_size: shape of the input signal tensor.
    :param int | tuple[int, ...] patch_size: size of each patch (inner size).
    :param int | tuple[int, ...] overlap: padding radius around each patch for receptive field (halo).
    :param int | tuple[int, ...] | None stride: stride between patches. If `None`, uses patch_size.
    :param int | tuple[int, ...] | None tiling_dims: dimensions to tile.
        -   If `None`, defaults to last N dimensions where N is `len(patch_size)` if `patch_size` is `tuple`, else `2`.
        -   If `int`, tiles only that dimension.
        -   If `tuple[int]`, tiles the specified dimensions.

    :param str pad_mode: padding mode.
    :return: tuple of (global_slices, metadata).
    """
    shape = list(img_size)
    ndim = len(shape)

    # Determine dimensions to tile
    if tiling_dims is None:
        if isinstance(patch_size, tuple):
            n_tiled = len(patch_size)
            tiling_dims = tuple(range(ndim - n_tiled, ndim))
        else:
            # Default to 2D (-2, -1) if patch_size is int
            tiling_dims = (-2, -1)
    elif isinstance(tiling_dims, int):
        tiling_dims = (tiling_dims,)
    elif not isinstance(tiling_dims, tuple):
        raise ValueError("tiling_dims must be None, int, or tuple of ints.")

    num_tiled_dims = len(tiling_dims)

    # Helper to normalize int/tuple args
    def to_tuple(val, name):
        if isinstance(val, int):
            return (val,) * num_tiled_dims
        if len(val) != num_tiled_dims:
            raise ValueError(f"{name} must have length {num_tiled_dims}")
        return tuple(val)

    p_sizes = to_tuple(patch_size, "patch_size")
    rf_sizes = to_tuple(overlap, "overlap")

    if stride is None:
        strides = p_sizes
    else:
        strides = to_tuple(stride, "stride")

    # Initialize padding for all dims to 0
    # F.pad expects padding for last dim first: (last_left, last_right, 2nd_last_left, ...)
    pads = [0] * (ndim * 2)

    dim_starts = []

    for i, dim_idx in enumerate(tiling_dims):
        # Handle negative indices
        if dim_idx < 0:
            dim_idx += ndim

        D = shape[dim_idx]
        p = p_sizes[i]

        if D <= p:
            raise ValueError(
                f"Dimension {dim_idx} size {D} is smaller than or equal to patch size {p}."
            )

        rf = rf_sizes[i]
        s = strides[i]

        tile_size = p + 2 * rf
        D_pad = D + 2 * rf

        # Update padding list for F.pad
        # pad index for dim_idx:
        # (ndim - 1 - dim_idx) * 2  -> left/front/top
        # (ndim - 1 - dim_idx) * 2 + 1 -> right/back/bottom
        pad_idx_base = (ndim - 1 - dim_idx) * 2
        pads[pad_idx_base] = rf
        pads[pad_idx_base + 1] = rf

        # Calculate starts
        starts = [0]
        while starts[-1] + tile_size < D_pad:
            starts.append(starts[-1] + s)
        # Force last tile to end at D_pad
        starts[-1] = D_pad - tile_size
        starts = sorted(list(set(starts)))
        dim_starts.append(starts)

    # Pre-calculate trim information for each dimension to avoid overlaps
    dim_configs = []
    for i, dim_idx in enumerate(tiling_dims):
        starts = dim_starts[i]
        p = p_sizes[i]
        rf = rf_sizes[i]

        configs = []
        for j, st in enumerate(starts):
            if j == 0:
                trim = 0
            else:
                prev_st = starts[j - 1]
                trim = max(0, (prev_st + p) - st)

            configs.append(
                {"start": st, "trim": trim, "p": p, "rf": rf, "tile_size": p + 2 * rf}
            )
        dim_configs.append(configs)

    global_slices: list[Index] = []
    crop_slices: list[Index] = []
    target_slices: list[Index] = []

    for config_tuple in itertools.product(*dim_configs):
        # config_tuple contains the config dict for each tiled dimension

        # Initialize slices as full slices
        g_sl = [slice(None)] * ndim
        c_sl = [slice(None)] * ndim
        t_sl = [slice(None)] * ndim

        for i, dim_idx in enumerate(tiling_dims):
            if dim_idx < 0:
                dim_idx += ndim

            cfg = config_tuple[i]
            st = cfg["start"]
            trim = cfg["trim"]
            p = cfg["p"]
            rf = cfg["rf"]
            tile_size = cfg["tile_size"]

            # Global slice (into padded)
            g_sl[dim_idx] = slice(st, st + tile_size)

            # Crop slice (remove halo AND trim overlap)
            c_sl[dim_idx] = slice(rf + trim, rf + p)

            # Target slice (into original)
            t_sl[dim_idx] = slice(st + trim, st + p)

        global_slices.append(tuple(g_sl))
        crop_slices.append(tuple(c_sl))
        target_slices.append(tuple(t_sl))

    metadata: dict = {
        "img_size": torch.Size(img_size),
        "tiling_dims": tiling_dims,
        "crop_slices": crop_slices,
        "target_slices": target_slices,
        "global_padding": tuple(pads),
        "pad_mode": pad_mode,
        "window_shape": tuple(p + 2 * rf for p, rf in zip(p_sizes, rf_sizes)),
        "inner_patch_size": p_sizes,
        "grid_shape": tuple(len(s) for s in dim_starts),
        "overlap": rf_sizes if len(set(rf_sizes)) > 1 else rf_sizes[0],
        "stride": strides,
    }
    return global_slices, metadata
