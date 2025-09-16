from typing import Sequence, Tuple, Optional, List, Dict
import torch

Index = Tuple[slice, ...]

def _normalize_hw_args(
    signal_shape: Sequence[int],
    patch_size: int | Tuple[int, int],
    stride: Optional[Tuple[int, int]],
    hw_dims: Tuple[int, int],
) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Returns: (ndim, H_dim, W_dim, H, W, ph, pw, sh, sw)
    """
    shape = list(signal_shape)
    ndim = len(shape)
    H_dim, W_dim = hw_dims
    H = shape[H_dim]
    W = shape[W_dim]
    if isinstance(patch_size, int):
        ph, pw = patch_size, patch_size
    else:
        ph, pw = patch_size
    if stride is None:
        sh, sw = ph, pw
    else:
        sh, sw = stride
    return ndim, H_dim, W_dim, H, W, ph, pw, sh, sw


def tiling_splitting_strategy(
    signal_shape: Sequence[int],
    *,
    patch_size: int | Tuple[int, int],
    receptive_field_radius: int = 0,
    stride: Optional[Tuple[int, int]] = None,
    hw_dims: Tuple[int, int] = (-2, -1),
    non_overlap: bool = True,
    end_align: Optional[bool] = None, # ignored when non_overlap=True
) -> Tuple[List[Index], Dict]:
    """
    Tiling splitting strategy for distributed processing.
    
    This strategy creates windows around patches that include receptive field context.
    
    The strategy works by:
    1. Dividing the signal into non-overlapping patches
    2. Expanding each patch by receptive_field_radius in all directions
    3. Clipping the expanded windows to valid signal boundaries
    4. Storing crop information to extract the original patch from processed windows

    Parameters:
    -----------
    signal_shape : Sequence[int]
        Shape of the signal tensor (e.g., [B, C, H, W])
    patch_size : int | Tuple[int, int]
        Size of the non-overlapping patches
    receptive_field_radius : int, optional
        Radius to expand patches for context (default: 0)
    stride : Optional[Tuple[int, int]], optional
        Custom stride (default: patch_size for non-overlapping)
    hw_dims : Tuple[int, int], optional
        Dimensions for height and width (default: (-2, -1))
    non_overlap : bool, optional
        If True, creates non-overlapping patches (default: True)
    end_align : Optional[bool], optional
        Ignored when non_overlap=True

    Returns:
    --------
    Tuple[List[Index], Dict]
        - List of global slices to extract windows from the signal
        - Metadata including crop_slices and target_slices for reconstruction

    If non_overlap=True (default):
        - grid is non-overlapping (stride := patch_size)
        - last tiles are ragged (no end-alignment append)
        - matches old behavior: inner patches do NOT overlap, so no merge/blend needed

    If non_overlap=False:
        - stride as provided
        - optionally end-align last start if end_align=True
    """
    ndim, H_dim, W_dim, H, W, ph, pw, sh, sw = _normalize_hw_args(
        signal_shape, patch_size, stride, hw_dims
    )
    signal_shape = torch.Size(signal_shape)

    if non_overlap:
        # - non-overlapping inner patches
        # - ragged last tile (no end-alignment)
        sh, sw = ph, pw
        h_starts = list(range(0, H, sh))
        w_starts = list(range(0, W, sw))
    else:
        if end_align is None:
            end_align = True 
        if end_align:
            h_starts = list(range(0, max(H - ph, 0) + 1, sh))
            if h_starts[-1] != H - ph:
                h_starts.append(H - ph)
            w_starts = list(range(0, max(W - pw, 0) + 1, sw))
            if w_starts[-1] != W - pw:
                w_starts.append(W - pw)
        else:
            h_starts = list(range(0, H, sh))
            w_starts = list(range(0, W, sw))

    global_slices: List[Index] = []
    crop_slices: List[Index] = []
    target_slices: List[Index] = []

    for hs in h_starts:
        for ws in w_starts:
            # inner patch bounds (ragged at the end if needed)
            p_top, p_left = hs, ws
            p_bot, p_right = min(hs + ph, H), min(ws + pw, W)

            # window (with rf), clamped to image bounds
            w_top = max(0, p_top - receptive_field_radius)
            w_left = max(0, p_left - receptive_field_radius)
            w_bot = min(H, p_bot + receptive_field_radius)
            w_right = min(W, p_right + receptive_field_radius)

            # window slice in full tensor space (select all other dims)
            win_sl = [slice(None)] * ndim
            win_sl[H_dim] = slice(w_top, w_bot)
            win_sl[W_dim] = slice(w_left, w_right)
            win_sl = tuple(win_sl)

            # crop slice relative to the window
            c_top = p_top - w_top
            c_left = p_left - w_left
            c_bot = c_top + (p_bot - p_top)
            c_right = c_left + (p_right - p_left)

            crop_sl = [slice(None)] * ndim
            crop_sl[H_dim] = slice(c_top, c_bot)
            crop_sl[W_dim] = slice(c_left, c_right)
            crop_sl = tuple(crop_sl)

            # target slice in the full output
            tgt_sl = [slice(None)] * ndim
            tgt_sl[H_dim] = slice(p_top, p_bot)
            tgt_sl[W_dim] = slice(p_left, p_right)
            tgt_sl = tuple(tgt_sl)

            global_slices.append(win_sl)
            crop_slices.append(crop_sl)
            target_slices.append(tgt_sl)

    metadata: Dict = {
        "signal_shape": signal_shape,
        "hw_dims": (H_dim, W_dim),
        "crop_slices": crop_slices,
        "target_slices": target_slices,
    }
    return global_slices, metadata


def tiling_reduce_fn(
    pieces: List[torch.Tensor],
    metadata: Dict,
) -> torch.Tensor:
    """
    Reduction function for tiling strategy.
    
    This function reassembles processed window pieces back into the original signal shape.
    It crops each processed window to extract only the relevant patch region and places
    it in the correct location in the output tensor.
    
    Parameters:
    -----------
    pieces : List[torch.Tensor]
        List of processed window tensors
    metadata : Dict
        Metadata from tiling_splitting_strategy containing:
        - signal_shape: Original signal shape
        - crop_slices: How to crop each piece to get the patch
        - target_slices: Where to place each patch in the output
        
    Returns:
    --------
    torch.Tensor
        Reconstructed signal with original shape
    """
    if len(pieces) == 0:
        raise ValueError("pieces is empty.")

    device = pieces[0].device
    dtype = pieces[0].dtype
    out = torch.zeros(metadata["signal_shape"], device=device, dtype=dtype)

    crop_slices: List[Index] = metadata["crop_slices"]
    target_slices: List[Index] = metadata["target_slices"]

    if len(crop_slices) != len(pieces) or len(target_slices) != len(pieces):
        raise ValueError("Length mismatch among pieces/crop_slices/target_slices.")

    for tile, c_sl, t_sl in zip(pieces, crop_slices, target_slices):
        out[t_sl] = tile[c_sl]  # equivalent to concatenation when no overlap

    return out
