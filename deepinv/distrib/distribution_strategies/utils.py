from __future__ import annotations

from typing import Sequence, Optional

import torch
import torch.nn.functional as F

Index = tuple[slice, ...]


def tiling3d_reduce_fn(
    out_local: torch.Tensor,
    local_pairs: list[tuple[int, torch.Tensor]],
    global_metadata: dict,
) -> None:
    r"""
    Default reduction function for tiling3d strategy.

    This function fills the out_local tensor with the processed patches from local_pairs,
    using the metadata to determine where each patch should be placed in the output tensor.

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


def tiling2d_reduce_fn(
    out_local: torch.Tensor,
    local_pairs: list[tuple[int, torch.Tensor]],
    global_metadata: dict,
) -> None:
    r"""
    Default reduction function for tiling2d strategy.

    This function fills the out_local tensor with the processed patches from local_pairs,
    using the metadata to determine where each patch should be placed in the output tensor.

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


def extract_and_pad_patch(
    X: torch.Tensor, idx: int, global_slices: list[Index], global_metadata: dict
) -> torch.Tensor:
    r"""
    Extract and pad a single patch from the input tensor.

    :param torch.Tensor X: input tensor.
    :param int idx: global index of the patch.
    :param list[Index] global_slices: list of slice objects for extracting patches.
    :param dict global_metadata: metadata containing padding specifications.
    :return: (:class:`torch.Tensor`) extracted and padded patch.
    """
    slc = global_slices[idx]
    piece = X[slc]

    # Apply padding if specifications are available
    pad_specs = global_metadata.get("pad_specs", [])
    if pad_specs and idx < len(pad_specs):
        pad_mode = global_metadata.get("pad_mode", "reflect")
        pad_spec = pad_specs[idx]

        # Validate padding specifications to avoid PyTorch errors
        if len(pad_spec) >= 4:  # (w_left, w_right, h_top, h_bottom)
            w_left, w_right, h_top, h_bottom = pad_spec[:4]

            # Check if padding is reasonable compared to tensor dimensions
            if len(piece.shape) >= 2:
                h_dim, w_dim = piece.shape[-2], piece.shape[-1]

                # Clamp padding to be at most the dimension size
                w_left = min(w_left, w_dim)
                w_right = min(w_right, w_dim)
                h_top = min(h_top, h_dim)
                h_bottom = min(h_bottom, h_dim)

                # Use the clamped padding
                safe_pad_spec = (w_left, w_right, h_top, h_bottom) + pad_spec[4:]

                try:
                    piece = F.pad(piece, pad=safe_pad_spec, mode=pad_mode)
                except RuntimeError as e:
                    raise RuntimeError(f"Padding failed for patch {idx}: {e}")
            else:
                # For 1D or scalar tensors, apply padding as-is if possible
                try:
                    piece = F.pad(piece, pad=pad_spec, mode=pad_mode)
                except RuntimeError as e:
                    raise RuntimeError(f"Padding failed for patch {idx}: {e}")

    return piece


def _normalize_hw_args(
    signal_shape: Sequence[int],
    patch_size: int | tuple[int, int],
    stride: Optional[tuple[int, int]],
    hw_dims: tuple[int, int],
    *,
    receptive_field_radius: int = 0,
    non_overlap: bool = True,
    end_align: Optional[bool] = None,  # only used if non_overlap=False
) -> tuple[
    int,
    int,
    int,
    int,
    int,  # ndim, H_dim, W_dim, H, W
    int,
    int,
    int,
    int,  # ph, pw, sh, sw
    int,
    int,  # H_pad, W_pad
    int,
    int,
    int,
    int,  # pad_bottom, pad_right, (pad_top=0), (pad_left=0)
    int,
    int,  # win_h, win_w
]:
    """
    Returns:
        (ndim, H_dim, W_dim, H, W, ph, pw, sh, sw,
         H_pad, W_pad, pad_bottom, pad_right, pad_top, pad_left, win_h, win_w)

    Notes:
        • H_pad/W_pad are the virtual canvas sizes used to place a uniform tile grid.
        • We conceptually pad only on the bottom/right to reach multiples; receptive-field
          padding is handled per-tile (returned by tiler) so pad_top/left are 0 here.
    """
    shape = list(signal_shape)
    ndim = len(shape)
    H_dim, W_dim = hw_dims
    H = int(shape[H_dim])
    W = int(shape[W_dim])

    if isinstance(patch_size, int):
        ph, pw = patch_size, patch_size
    else:
        ph, pw = int(patch_size[0]), int(patch_size[1])

    if stride is None:
        sh, sw = ph, pw
    else:
        sh, sw = int(stride[0]), int(stride[1])

    # Compute padded canvas for a uniform grid of tile *starts*.
    if non_overlap:
        # grid at 0, ph, 2ph, ..., H_pad - ph  (same for W)
        H_pad = ((H + ph - 1) // ph) * ph
        W_pad = ((W + pw - 1) // pw) * pw
    else:
        # end-aligned sliding windows: last start is ceil((H-ph)/sh)*sh
        if end_align is None:
            end_align = True
        if end_align:
            last_h = max(0, ((max(H - ph, 0) + sh - 1) // sh) * sh)
            last_w = max(0, ((max(W - pw, 0) + sw - 1) // sw) * sw)
            H_pad = last_h + ph
            W_pad = last_w + pw
        else:
            # no extra padding needed; grid stops before end
            H_pad = H
            W_pad = W

    pad_bottom = max(0, H_pad - H)
    pad_right = max(0, W_pad - W)
    pad_top = 0
    pad_left = 0

    win_h = ph + 2 * receptive_field_radius
    win_w = pw + 2 * receptive_field_radius

    return (
        ndim,
        H_dim,
        W_dim,
        H,
        W,
        ph,
        pw,
        sh,
        sw,
        H_pad,
        W_pad,
        pad_bottom,
        pad_right,
        pad_top,
        pad_left,
        win_h,
        win_w,
    )


def tiling_splitting_strategy(
    signal_shape: Sequence[int],
    *,
    patch_size: int | tuple[int, int],
    receptive_field_radius: int = 0,
    stride: Optional[tuple[int, int]] = None,
    hw_dims: tuple[int, int] = (-2, -1),
    non_overlap: bool = True,
    end_align: Optional[bool] = None,  # used when non_overlap=False
    pad_mode: str = "reflect",  # advisory; not applied here
) -> tuple[list[Index], dict]:
    r"""
    Uniform-batching tiler with per-tile padding.

    Produces:
      • global_slices: list of slices into the *original* tensor (no out-of-bounds),
      • crop_slices:   slices relative to the *padded window* that grab only the inner patch
                       portion that overlaps the original image,
      • target_slices: slices into the *original-shape* output where that patch should be placed,
      • pad_specs:     per-tile padding tuple for F.pad: (w_left, w_right, h_top, h_bottom),
                       so that after padding, each window is (ph+2rf, pw+2rf).

    :param Sequence[int] signal_shape: shape of the input signal tensor.
    :param int, tuple[int, int] patch_size: size of each patch. If int, assumes square patches.
    :param int receptive_field_radius: padding radius around each patch for receptive field.
    :param None, tuple[int, int] stride: stride between patches. If `None`, uses patch_size for non-overlapping.
    :param tuple[int, int] hw_dims: dimensions corresponding to height and width.
    :param bool non_overlap: whether patches should be non-overlapping.
    :param None, bool end_align: alignment strategy for overlapping patches.
    :param str pad_mode: padding mode (advisory; not applied here).
    :return: (tuple) tuple of (global_slices, metadata).

    |sep|

    :Examples:

        Create tiling strategy for an image:

        >>> signal_shape = (1, 3, 512, 512)
        >>> global_slices, metadata = tiling_splitting_strategy(
        ...     signal_shape, patch_size=256, receptive_field_radius=32
        ... )
        >>> # Use for batching:
        >>> piece = X[global_slices[i]]
        >>> piece = torch.nn.functional.pad(piece, pad=metadata['pad_specs'][i], mode='reflect')
    """
    (
        ndim,
        H_dim,
        W_dim,
        H,
        W,
        ph,
        pw,
        sh,
        sw,
        H_pad,
        W_pad,
        _pad_bottom,
        _pad_right,
        _pad_top,
        _pad_left,
        win_h,
        win_w,
    ) = _normalize_hw_args(
        signal_shape,
        patch_size,
        stride,
        hw_dims,
        receptive_field_radius=receptive_field_radius,
        non_overlap=non_overlap,
        end_align=end_align,
    )
    signal_shape = torch.Size(signal_shape)

    # Uniform grid of patch starts over the padded canvas
    if non_overlap:
        h_starts = list(range(0, H_pad, ph))
        w_starts = list(range(0, W_pad, pw))
    else:
        if end_align is None:
            end_align = True
        if end_align:
            last_h = H_pad - ph
            last_w = W_pad - pw
            h_starts = list(range(0, last_h + 1, sh))
            w_starts = list(range(0, last_w + 1, sw))
        else:
            h_starts = list(range(0, max(H - ph, 0) + 1, sh))
            w_starts = list(range(0, max(W - pw, 0) + 1, sw))

    global_slices: list[Index] = []
    crop_slices: list[Index] = []
    target_slices: list[Index] = []
    pad_specs: list[tuple[int, int, int, int]] = []

    rf = receptive_field_radius

    for hs in h_starts:
        for ws in w_starts:
            # Inner patch in padded-canvas coords
            p_top, p_left = hs, ws
            p_bot, p_right = hs + ph, ws + pw

            # Amount of valid patch that lies inside the *original* image
            # (ragged last tiles get truncated)
            patch_h = max(0, min(ph, H - p_top))
            patch_w = max(0, min(pw, W - p_left))
            if patch_h == 0 or patch_w == 0:
                # This tile lies entirely in the padded extension; skip
                continue

            # Desired window in original coords (before per-tile padding)
            w_top_desired = p_top - rf
            w_left_desired = p_left - rf
            w_bot_desired = p_bot + rf
            w_right_desired = p_right + rf

            # Clip window to original tensor; compute how much padding is needed to
            # restore the uniform window size (win_h, win_w).
            w_top = max(0, w_top_desired)
            w_left = max(0, w_left_desired)
            w_bot = min(H, w_bot_desired)
            w_right = min(W, w_right_desired)

            need_h_top = max(0, 0 - w_top_desired)  # == rf - p_top if p_top<rf
            need_h_bot = max(0, w_bot_desired - H)  # == p_bot+rf - H if beyond bottom
            need_w_left = max(0, 0 - w_left_desired)  # == rf - p_left if p_left<rf
            need_w_right = max(
                0, w_right_desired - W
            )  # == p_right+rf - W if beyond right

            # Slices into original tensor (no OOB)
            win_sl = [slice(None)] * ndim
            win_sl[H_dim] = slice(w_top, w_bot)
            win_sl[W_dim] = slice(w_left, w_right)
            win_sl = tuple(win_sl)

            # After padding this window by the amounts above, its size will be (win_h, win_w).
            # In that padded window, the inner patch normally sits at [rf:rf+ph, rf:rf+pw],
            # but when the patch is truncated at the image boundary, we only take the valid subregion.
            c_top = rf
            c_left = rf
            c_bot = rf + patch_h
            c_right = rf + patch_w

            crop_sl = [slice(None)] * ndim
            crop_sl[H_dim] = slice(c_top, c_bot)
            crop_sl[W_dim] = slice(c_left, c_right)
            crop_sl = tuple(crop_sl)

            # Target region inside the *original-shape* output tensor
            tgt_sl = [slice(None)] * ndim
            tgt_sl[H_dim] = slice(p_top, p_top + patch_h)
            tgt_sl[W_dim] = slice(p_left, p_left + patch_w)
            tgt_sl = tuple(tgt_sl)

            # Pad spec for F.pad: (w_left, w_right, h_top, h_bottom)
            pad_specs.append((need_w_left, need_w_right, need_h_top, need_h_bot))

            global_slices.append(win_sl)
            crop_slices.append(crop_sl)
            target_slices.append(tgt_sl)

    metadata: dict = {
        "signal_shape": torch.Size(signal_shape),  # original (unpadded) output shape
        "hw_dims": (H_dim, W_dim),
        "crop_slices": crop_slices,
        "target_slices": target_slices,
        "pad_specs": pad_specs,  # per-tile padding for uniform windows
        "window_shape": (win_h, win_w),  # uniform H×W after per-tile pad
        "inner_patch_size": (ph, pw),
        "grid_shape": (len(h_starts), len(w_starts)),
        "pad_mode": pad_mode,
        "receptive_field_radius": rf,
        "non_overlap": non_overlap,
        "stride": (sh, sw),
    }
    return global_slices, metadata


def _normalize_dhw_args(
    signal_shape: Sequence[int],
    patch_size: int | tuple[int, int, int],
    stride: Optional[tuple[int, int, int]],
    dhw_dims: tuple[int, int, int],
    *,
    receptive_field_radius: int = 0,
    non_overlap: bool = True,
    end_align: Optional[bool] = None,
) -> tuple[
    int,
    int,
    int,
    int,  # ndim, D_dim, H_dim, W_dim
    int,
    int,
    int,  # D, H, W
    int,
    int,
    int,  # pd, ph, pw
    int,
    int,
    int,  # sd, sh, sw
    int,
    int,
    int,  # D_pad, H_pad, W_pad
    int,
    int,
    int,
    int,
    int,
    int,  # pad_back, pad_bottom, pad_right, pad_front, pad_top, pad_left
    int,
    int,
    int,  # win_d, win_h, win_w
]:
    """
    Normalize arguments for 3D tiling.

    Returns:
        (ndim, D_dim, H_dim, W_dim, D, H, W, pd, ph, pw, sd, sh, sw,
         D_pad, H_pad, W_pad, pad_back, pad_bottom, pad_right, 
         pad_front, pad_top, pad_left, win_d, win_h, win_w)

    Notes:
        • D_pad/H_pad/W_pad are the virtual canvas sizes for uniform tile grid.
        • Conceptual padding on back/bottom/right; receptive-field padding per-tile.
    """
    shape = list(signal_shape)
    ndim = len(shape)
    D_dim, H_dim, W_dim = dhw_dims
    D = int(shape[D_dim])
    H = int(shape[H_dim])
    W = int(shape[W_dim])

    if isinstance(patch_size, int):
        pd, ph, pw = patch_size, patch_size, patch_size
    else:
        pd, ph, pw = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])

    if stride is None:
        sd, sh, sw = pd, ph, pw
    else:
        sd, sh, sw = int(stride[0]), int(stride[1]), int(stride[2])

    # Compute padded canvas for uniform grid
    if non_overlap:
        D_pad = ((D + pd - 1) // pd) * pd
        H_pad = ((H + ph - 1) // ph) * ph
        W_pad = ((W + pw - 1) // pw) * pw
    else:
        if end_align is None:
            end_align = True
        if end_align:
            last_d = max(0, ((max(D - pd, 0) + sd - 1) // sd) * sd)
            last_h = max(0, ((max(H - ph, 0) + sh - 1) // sh) * sh)
            last_w = max(0, ((max(W - pw, 0) + sw - 1) // sw) * sw)
            D_pad = last_d + pd
            H_pad = last_h + ph
            W_pad = last_w + pw
        else:
            D_pad = D
            H_pad = H
            W_pad = W

    pad_back = max(0, D_pad - D)
    pad_bottom = max(0, H_pad - H)
    pad_right = max(0, W_pad - W)
    pad_front = 0
    pad_top = 0
    pad_left = 0

    win_d = pd + 2 * receptive_field_radius
    win_h = ph + 2 * receptive_field_radius
    win_w = pw + 2 * receptive_field_radius

    return (
        ndim,
        D_dim,
        H_dim,
        W_dim,
        D,
        H,
        W,
        pd,
        ph,
        pw,
        sd,
        sh,
        sw,
        D_pad,
        H_pad,
        W_pad,
        pad_back,
        pad_bottom,
        pad_right,
        pad_front,
        pad_top,
        pad_left,
        win_d,
        win_h,
        win_w,
    )


def tiling3d_splitting_strategy(
    signal_shape: Sequence[int],
    *,
    patch_size: int | tuple[int, int, int],
    receptive_field_radius: int = 0,
    stride: Optional[tuple[int, int, int]] = None,
    dhw_dims: tuple[int, int, int] = (-3, -2, -1),
    non_overlap: bool = True,
    end_align: Optional[bool] = None,
    pad_mode: str = "reflect",
) -> tuple[list[Index], dict]:
    r"""
    Uniform-batching 3D tiler with per-cube padding.

    Produces:
      • global_slices: list of slices into the *original* tensor (no out-of-bounds),
      • crop_slices:   slices relative to the *padded window* that grab only the inner cube
                       portion that overlaps the original volume,
      • target_slices: slices into the *original-shape* output where that cube should be placed,
      • pad_specs:     per-cube padding tuple for F.pad: (w_left, w_right, h_top, h_bottom, d_front, d_back),
                       so that after padding, each window is (pd+2rf, ph+2rf, pw+2rf).

    :param Sequence[int] signal_shape: shape of the input signal tensor (e.g., [B, C, D, H, W]).
    :param int, tuple[int, int, int] patch_size: size of each cube patch. If int, assumes cubic patches.
    :param int receptive_field_radius: padding radius around each patch for receptive field.
    :param None, tuple[int, int, int] stride: stride between patches. If `None`, uses patch_size for non-overlapping.
    :param tuple[int, int, int] dhw_dims: dimensions corresponding to depth, height, and width.
    :param bool non_overlap: whether patches should be non-overlapping.
    :param None, bool end_align: alignment strategy for overlapping patches.
    :param str pad_mode: padding mode (advisory; not applied here).
    :return: (tuple) tuple of (global_slices, metadata).

    |sep|

    :Examples:

        Create tiling strategy for a 3D volume:

        >>> signal_shape = (1, 1, 64, 64, 64)
        >>> global_slices, metadata = tiling3d_splitting_strategy(
        ...     signal_shape, patch_size=32, receptive_field_radius=8
        ... )
        >>> # Use for batching:
        >>> piece = X[global_slices[i]]
        >>> piece = torch.nn.functional.pad(piece, pad=metadata['pad_specs'][i], mode='reflect')
    """
    (
        ndim,
        D_dim,
        H_dim,
        W_dim,
        D,
        H,
        W,
        pd,
        ph,
        pw,
        sd,
        sh,
        sw,
        D_pad,
        H_pad,
        W_pad,
        _pad_back,
        _pad_bottom,
        _pad_right,
        _pad_front,
        _pad_top,
        _pad_left,
        win_d,
        win_h,
        win_w,
    ) = _normalize_dhw_args(
        signal_shape,
        patch_size,
        stride,
        dhw_dims,
        receptive_field_radius=receptive_field_radius,
        non_overlap=non_overlap,
        end_align=end_align,
    )
    signal_shape = torch.Size(signal_shape)

    # Uniform grid of patch starts over the padded canvas
    if non_overlap:
        d_starts = list(range(0, D_pad, pd))
        h_starts = list(range(0, H_pad, ph))
        w_starts = list(range(0, W_pad, pw))
    else:
        if end_align is None:
            end_align = True
        if end_align:
            last_d = D_pad - pd
            last_h = H_pad - ph
            last_w = W_pad - pw
            d_starts = list(range(0, last_d + 1, sd))
            h_starts = list(range(0, last_h + 1, sh))
            w_starts = list(range(0, last_w + 1, sw))
        else:
            d_starts = list(range(0, max(D - pd, 0) + 1, sd))
            h_starts = list(range(0, max(H - ph, 0) + 1, sh))
            w_starts = list(range(0, max(W - pw, 0) + 1, sw))

    global_slices: list[Index] = []
    crop_slices: list[Index] = []
    target_slices: list[Index] = []
    pad_specs: list[tuple[int, int, int, int, int, int]] = []

    rf = receptive_field_radius

    for ds in d_starts:
        for hs in h_starts:
            for ws in w_starts:
                # Inner patch in padded-canvas coords
                p_front, p_top, p_left = ds, hs, ws
                p_back, p_bot, p_right = ds + pd, hs + ph, ws + pw

                # Amount of valid patch that lies inside the *original* volume
                patch_d = max(0, min(pd, D - p_front))
                patch_h = max(0, min(ph, H - p_top))
                patch_w = max(0, min(pw, W - p_left))
                if patch_d == 0 or patch_h == 0 or patch_w == 0:
                    continue

                # Desired window in original coords
                w_front_desired = p_front - rf
                w_top_desired = p_top - rf
                w_left_desired = p_left - rf
                w_back_desired = p_back + rf
                w_bot_desired = p_bot + rf
                w_right_desired = p_right + rf

                # Clip window to original tensor
                w_front = max(0, w_front_desired)
                w_top = max(0, w_top_desired)
                w_left = max(0, w_left_desired)
                w_back = min(D, w_back_desired)
                w_bot = min(H, w_bot_desired)
                w_right = min(W, w_right_desired)

                need_d_front = max(0, 0 - w_front_desired)
                need_d_back = max(0, w_back_desired - D)
                need_h_top = max(0, 0 - w_top_desired)
                need_h_bot = max(0, w_bot_desired - H)
                need_w_left = max(0, 0 - w_left_desired)
                need_w_right = max(0, w_right_desired - W)

                # Slices into original tensor
                win_sl = [slice(None)] * ndim
                win_sl[D_dim] = slice(w_front, w_back)
                win_sl[H_dim] = slice(w_top, w_bot)
                win_sl[W_dim] = slice(w_left, w_right)
                win_sl = tuple(win_sl)

                # Crop slices in padded window
                c_front = rf
                c_top = rf
                c_left = rf
                c_back = rf + patch_d
                c_bot = rf + patch_h
                c_right = rf + patch_w

                crop_sl = [slice(None)] * ndim
                crop_sl[D_dim] = slice(c_front, c_back)
                crop_sl[H_dim] = slice(c_top, c_bot)
                crop_sl[W_dim] = slice(c_left, c_right)
                crop_sl = tuple(crop_sl)

                # Target region in output tensor
                tgt_sl = [slice(None)] * ndim
                tgt_sl[D_dim] = slice(p_front, p_front + patch_d)
                tgt_sl[H_dim] = slice(p_top, p_top + patch_h)
                tgt_sl[W_dim] = slice(p_left, p_left + patch_w)
                tgt_sl = tuple(tgt_sl)

                # Pad spec for F.pad: (w_left, w_right, h_top, h_bottom, d_front, d_back)
                pad_specs.append(
                    (
                        need_w_left,
                        need_w_right,
                        need_h_top,
                        need_h_bot,
                        need_d_front,
                        need_d_back,
                    )
                )

                global_slices.append(win_sl)
                crop_slices.append(crop_sl)
                target_slices.append(tgt_sl)

    metadata: dict = {
        "signal_shape": torch.Size(signal_shape),
        "dhw_dims": (D_dim, H_dim, W_dim),
        "crop_slices": crop_slices,
        "target_slices": target_slices,
        "pad_specs": pad_specs,
        "window_shape": (win_d, win_h, win_w),
        "inner_patch_size": (pd, ph, pw),
        "grid_shape": (len(d_starts), len(h_starts), len(w_starts)),
        "pad_mode": pad_mode,
        "receptive_field_radius": rf,
        "non_overlap": non_overlap,
        "stride": (sd, sh, sw),
    }
    return global_slices, metadata
