from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ._internal import _add_tuple, _as_pair


def _resolve_tiling_params(
    patch_size: int | tuple[int, int],
    stride: int | tuple[int, int] | None = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    patch_size_2d = _as_pair(patch_size)
    stride_2d = (
        _as_pair(stride) if stride is not None else tuple(p // 2 for p in patch_size_2d)
    )

    if stride_2d[0] > patch_size_2d[0] or stride_2d[1] > patch_size_2d[1]:
        raise ValueError(
            f"Stride {stride_2d} must be smaller or equal than patch_size {patch_size_2d}."
        )

    return patch_size_2d, stride_2d


def _compute_needed_pad(
    img_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    n_h = abs(img_size[0] - patch_size[0]) // stride[0] + 1
    n_w = abs(img_size[1] - patch_size[1]) // stride[1] + 1

    pad_h = (patch_size[0] + n_h * stride[0] - img_size[0]) % stride[0]
    pad_w = (patch_size[1] + n_w * stride[1] - img_size[1]) % stride[1]

    return pad_h, pad_w


def _compute_compatible_img_size(
    img_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
) -> tuple[int, int]:
    return _add_tuple(img_size, _compute_needed_pad(img_size, patch_size, stride))


def _compute_num_patches(
    img_size: tuple[int, int],
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    pad_if_needed: bool,
) -> tuple[int, int]:
    compatible_size = (
        _compute_compatible_img_size(img_size, patch_size, stride)
        if pad_if_needed
        else img_size
    )
    n_h = (compatible_size[0] - patch_size[0]) // stride[0] + 1
    n_w = (compatible_size[1] - patch_size[1]) // stride[1] + 1
    return n_h, n_w


def _image_to_patches_impl(
    image: Tensor,
    patch_size: tuple[int, int],
    stride: tuple[int, int],
    pad_if_needed: bool = True,
    extra_pad: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tensor:
    if image.ndim != 4:
        raise ValueError(
            f"Input image must have shape (B, C, H, W), got {tuple(image.shape)}."
        )

    if pad_if_needed:
        pad_h, pad_w = _compute_needed_pad(image.shape[-2:], patch_size, stride)
        pad = (
            extra_pad[0],
            extra_pad[1] + pad_w,
            extra_pad[2],
            extra_pad[3] + pad_h,
        )
    else:
        pad = extra_pad

    if any(p > 0 for p in pad):
        image = F.pad(image, pad, mode="constant", value=0)

    patch_size = _add_tuple(
        patch_size, (extra_pad[2] + extra_pad[3], extra_pad[0] + extra_pad[1])
    )
    patches = image.unfold(2, patch_size[0], stride[0]).unfold(
        3, patch_size[1], stride[1]
    )
    return patches.contiguous()


def _patches_to_image_impl(
    patches: Tensor,
    stride: tuple[int, int],
    img_size: tuple[int, int] | None = None,
    reduce_overlap: str = "mean",
) -> Tensor:
    if reduce_overlap not in ["sum", "mean"]:
        raise ValueError(
            f"Invalid reduce_overlap option: {reduce_overlap}. Must be 'sum' or 'mean'."
        )

    B, C, num_patches_h, num_patches_w, h, w = patches.size()

    output_size = (
        h + (num_patches_h - 1) * stride[0],
        w + (num_patches_w - 1) * stride[1],
    )

    num_patches = num_patches_h * num_patches_w
    patches_reshaped = (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(B, num_patches, C, h, w)
        .view(B, num_patches, C * h * w)
        .permute(0, 2, 1)
    )

    output = F.fold(
        patches_reshaped,
        output_size=output_size,
        kernel_size=(h, w),
        stride=stride,
    )

    if reduce_overlap == "mean":
        mask = torch.ones_like(patches_reshaped)
        overlap_count = F.fold(
            mask,
            output_size=output_size,
            kernel_size=(h, w),
            stride=stride,
        ).clamp_min_(1)
        output = output / overlap_count

    if img_size is not None:
        img_size = _as_pair(img_size)
        output = output[:, :, : img_size[0], : img_size[1]]

    return output.contiguous()
