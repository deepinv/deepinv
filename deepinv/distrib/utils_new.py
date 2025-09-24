from typing import Sequence, Tuple, Optional, List, Dict
import torch

Index = Tuple[slice, ...]


def tiling2d_reduce_fn(out_local: torch.Tensor, local_pairs: List[Tuple[int, torch.Tensor]], global_metadata: Dict) -> None:
    """
    Default reduction function for tiling2d strategy.
    
    This function fills the out_local tensor with the processed patches from local_pairs,
    using the metadata to determine where each patch should be placed in the output tensor.
    
    Parameters
    ----------
    out_local : torch.Tensor
        The output tensor to fill (should be initialized to zeros).
    local_pairs : List[Tuple[int, torch.Tensor]]
        List of (global_index, processed_tensor) pairs from this rank.
    global_metadata : Dict
        Metadata from the splitting strategy containing crop_slices and target_slices.
    """
    crop_slices = global_metadata["crop_slices"]
    target_slices = global_metadata["target_slices"]
    
    for idx, processed_tensor in local_pairs:
        c_sl = crop_slices[idx]
        t_sl = target_slices[idx]
        # Extract the valid part of the processed patch and place it in the output
        out_local[t_sl] = processed_tensor[c_sl]

def _normalize_hw_args(
    signal_shape: Sequence[int],
    patch_size: int | Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    hw_dims: Tuple[int, int],
    *,
    receptive_field_radius: int = 0,
    non_overlap: bool = True,
    end_align: Optional[bool] = None,  # only used if non_overlap=False
) -> Tuple[
    int, int, int, int, int,  # ndim, H_dim, W_dim, H, W
    int, int, int, int,       # ph, pw, sh, sw
    int, int,                 # H_pad, W_pad
    int, int, int, int,       # pad_bottom, pad_right, (pad_top=0), (pad_left=0)
    int, int                  # win_h, win_w
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
    pad_right  = max(0, W_pad - W)
    pad_top = 0
    pad_left = 0

    win_h = ph + 2 * receptive_field_radius
    win_w = pw + 2 * receptive_field_radius

    return (
        ndim, H_dim, W_dim, H, W,
        ph, pw, sh, sw,
        H_pad, W_pad,
        pad_bottom, pad_right, pad_top, pad_left,
        win_h, win_w,
    )


def tiling_splitting_strategy(
    signal_shape: Sequence[int],
    *,
    patch_size: int | Tuple[int, int],
    receptive_field_radius: int = 0,
    stride: Optional[Tuple[int, int]] = None,
    hw_dims: Tuple[int, int] = (-2, -1),
    non_overlap: bool = True,
    end_align: Optional[bool] = None,  # used when non_overlap=False
    pad_mode: str = "reflect",         # advisory; not applied here
) -> Tuple[List[Index], Dict]:
    """
    Uniform-batching tiler with per-tile padding.

    Produces:
      • global_slices: list of slices into the *original* tensor (no out-of-bounds),
      • crop_slices:   slices relative to the *padded window* that grab only the inner patch
                       portion that overlaps the original image,
      • target_slices: slices into the *original-shape* output where that patch should be placed,
      • pad_specs:     per-tile padding tuple for F.pad: (w_left, w_right, h_top, h_bottom),
                       so that after padding, each window is (ph+2rf, pw+2rf).

    Usage for batching:
      piece = X[global_slices[i]]
      piece = torch.nn.functional.pad(piece, pad=pad_specs[i], mode=pad_mode)
      # piece now has uniform window size across all tiles; stack and run Prior.
    """
    (
        ndim, H_dim, W_dim, H, W,
        ph, pw, sh, sw,
        H_pad, W_pad,
        _pad_bottom, _pad_right, _pad_top, _pad_left,
        win_h, win_w,
    ) = _normalize_hw_args(
        signal_shape, patch_size, stride, hw_dims,
        receptive_field_radius=receptive_field_radius,
        non_overlap=non_overlap, end_align=end_align,
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

    global_slices: List[Index] = []
    crop_slices: List[Index] = []
    target_slices: List[Index] = []
    pad_specs: List[Tuple[int, int, int, int]] = []

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

            need_h_top = max(0, 0 - w_top_desired)         # == rf - p_top if p_top<rf
            need_h_bot = max(0, w_bot_desired - H)         # == p_bot+rf - H if beyond bottom
            need_w_left = max(0, 0 - w_left_desired)       # == rf - p_left if p_left<rf
            need_w_right = max(0, w_right_desired - W)     # == p_right+rf - W if beyond right

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

    metadata: Dict = {
        "signal_shape": torch.Size(signal_shape),   # original (unpadded) output shape
        "hw_dims": (H_dim, W_dim),
        "crop_slices": crop_slices,
        "target_slices": target_slices,
        "pad_specs": pad_specs,                     # per-tile padding for uniform windows
        "window_shape": (win_h, win_w),             # uniform H×W after per-tile pad
        "inner_patch_size": (ph, pw),
        "grid_shape": (len(h_starts), len(w_starts)),
        "pad_mode": pad_mode,
        "receptive_field_radius": rf,
        "non_overlap": non_overlap,
        "stride": (sh, sw),
    }
    return global_slices, metadata
