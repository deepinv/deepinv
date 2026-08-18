from __future__ import annotations

from typing import Literal
from torch import Tensor
import torch
from deepinv.utils._internal import _as_pair, _add_tuple


def _unity_partition_function_1d(
    image_size: int,
    patch_size: int,
    overlap: int,
    mode: Literal["bump", "linear"] = "bump",
    device="cpu",
    dtype=torch.float32,
) -> Tensor:
    r"""Create a 1D unity partition function for smooth patch blending.

    Creates masks that sum to 1 across all patches, enabling smooth blending
    in overlap regions. The function is 1 on [-a, a] and decreases to 0 on
    -(a+b) and (a+b), where a = patch_size/2 - overlap and b = overlap.

    :param int image_size: Size of the image dimension.
    :param int patch_size: Size of each patch.
    :param int overlap: Overlap between adjacent patches.
    :param str mode: Blending mode - 'bump' (smooth) or 'linear'.
    :return: Tensor of shape (n_patches, max_size) with partition masks.
    """
    n_patch = (image_size - patch_size) // (patch_size - overlap) + 1
    max_size = patch_size + (n_patch - 1) * (patch_size - overlap)
    t = torch.linspace(
        -max_size // 2, max_size // 2, max_size, device=device, dtype=dtype
    )

    if mode.lower() == "bump":
        from deepinv.physics.generator.blur import bump_function

        mask = bump_function(t, patch_size / 2 - overlap, overlap).roll(
            shifts=-max_size // 2 + patch_size // 2
        )
    elif mode.lower() == "linear":
        a = patch_size / 2 - overlap
        b = overlap
        mask = ((a + b) / b - t.abs().clip(0, a + b) / b).abs().clip(0, 1)
        mask = mask.roll(shifts=-max_size // 2 + patch_size // 2)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'bump' or 'linear'.")

    # Create masks for each patch
    masks = torch.stack(
        [mask.roll(shifts=(patch_size - overlap) * i) for i in range(n_patch)], dim=0
    )

    # Handle boundary patches
    masks[0, :overlap] = 1.0
    masks[-1, -overlap:] = 1.0

    # Normalize to sum to 1
    masks = masks / (masks.sum(dim=0, keepdims=True) + 1e-8)
    return masks


def _crop_unity_partition_2d(
    masks: Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> Tensor:
    r"""
    Crop unity partition masks to patch regions.

    Extracts the relevant portion of each partition mask corresponding to
    its patch location.
    """

    n_rows, n_cols = masks.size(0), masks.size(1)

    # Create grid indices for vectorized extraction
    row_idx = torch.arange(n_rows, device=masks.device)
    col_idx = torch.arange(n_cols, device=masks.device)
    h_starts = row_idx * stride[0]
    w_starts = col_idx * stride[1]

    # Extract cropped masks using unfold for efficiency
    # Reshape masks from (K_h, K_w, H, W) to process each patch location
    cropped_masks = torch.zeros(
        n_rows,
        n_cols,
        patch_size[0],
        patch_size[1],
        device=masks.device,
        dtype=masks.dtype,
    )
    for i in range(n_rows):
        for j in range(n_cols):
            h_start, w_start = int(h_starts[i]), int(w_starts[j])
            cropped_masks[i, j] = masks[
                i,
                j,
                h_start : h_start + patch_size[0],
                w_start : w_start + patch_size[1],
            ]

    return cropped_masks


def generate_tiled_multipliers(
    img_size: int | tuple[int, int],
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    mode: Literal["bump", "linear"] = "bump",
    device="cpu",
    dtype=torch.float32,
) -> Tensor:
    r"""
    Generate tiled multipliers for patch blending.

    It is used in tiled product convolution, see :class:`deepinv.physics.TiledSpaceVaryingBlur` to smoothly blend overlapping patches.

    :param img_size: Size of the image (height, width) or single `int` for square images.
    :param patch_size: Size of each patch (height, width) or single `int` for square patches.
    :param stride: Stride between adjacent patches (height, width). If a single `int` is provided, it is used for both dimensions.
    :param mode: Blending mode - `'bump'` (smooth) or `'linear'`.
    :return: Tensor of shape `(1, 1, K, H, W)` with multipliers for each patch.
    """
    img_size = _as_pair(img_size)
    patch_size = _as_pair(patch_size)
    stride = _as_pair(stride)
    overlap = _add_tuple(patch_size, stride, -1)

    masks_x = _unity_partition_function_1d(
        img_size[0], patch_size[0], overlap[0], mode, device=device, dtype=dtype
    )
    masks_y = _unity_partition_function_1d(
        img_size[1], patch_size[1], overlap[1], mode, device=device, dtype=dtype
    )

    # Combine 1D masks into 2D via outer product
    masks = torch.tensordot(masks_x, masks_y, dims=0)
    masks = masks.permute(0, 2, 1, 3)

    # Normalize to sum to 1
    masks = masks / (masks.sum(dim=(0, 1), keepdims=True) + 1e-8)

    w = _crop_unity_partition_2d(masks, patch_size, stride)
    return w.flatten(0, 1).unsqueeze(0).unsqueeze(0)
