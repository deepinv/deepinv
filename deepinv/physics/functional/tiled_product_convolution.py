from __future__ import annotations

from dataclasses import dataclass, field, InitVar
from typing import Literal

from torch import Tensor

import torch
import torch.nn.functional as F
from deepinv.physics.functional.utils import _as_pair, _add_tuple


@dataclass
class TiledPConv2dConfig:
    """Configuration for tiled patch-based operations. Used in :class:`deepinv.physics.TiledSpaceVaryingBlur`

    This dataclass centralizes all patch-related parameters and provides
    computed properties for derived values like stride and margin.

    :param patch_size: Size of each patch (height, width) or single `int` for square patches.
    :param stride: Stride between adjacent patches (height, width). If a single `int` is provided, it is used for both dimensions.
    :param psf_size_init: The initial size of the PSF/kernel (height, width) or single `int`. Optional.
    """

    patch_size: int | tuple[int, int]
    stride: int | tuple[int, int]
    psf_size_init: InitVar[int | tuple[int, int] | None] = None
    _psf_size: int | tuple[int, int] | None = field(init=False, default=None)

    def __post_init__(self, psf_size_init):
        """
        Normalize all sizes to tuples after initialization.
        """
        self.patch_size = _as_pair(self.patch_size)
        self.stride = _as_pair(self.stride)
        self.overlap = _add_tuple(self.patch_size, self.stride, -1)
        self._psf_size = _as_pair(psf_size_init) if psf_size_init is not None else None

    @property
    def psf_size(self) -> tuple[int, int] | None:
        """
        Size of the PSF/kernel (height, width).
        """
        return self._psf_size

    @psf_size.setter
    def psf_size(self, value: int | tuple[int, int] | None):
        if value is not None:
            self._psf_size = _as_pair(value)
        else:
            self._psf_size = None

    @property
    def margin(self) -> tuple[int, int]:
        """
        Compute margin based on PSF size: psf_size - 1.
        """
        if self.psf_size is None:
            return (0, 0)
        return (self.psf_size[0] - 1, self.psf_size[1] - 1)

    def get_num_patches(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """
        Compute the number of patches that fit in the image.
        If the image size is not compatible, we pad it beforehand.

        :param img_size: Image size (height, width).
        :return: Number of patches (n_rows, n_cols).
        """
        img_size = _as_pair(img_size)
        img_size = self._get_compatible_img_size(img_size)[0]
        stride = self.stride

        return (
            (img_size[0] - self.patch_size[0]) // stride[0] + 1,
            (img_size[1] - self.patch_size[1]) // stride[1] + 1,
        )

    def _get_compatible_img_size(
        self, img_size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Get compatible image size and required padding.

        :param img_size: Original image size (height, width).
        :return: Tuple of (compatible_size, padding).
        """
        # Compute number of patches and required padding
        n_h = (img_size[0] - self.patch_size[0]) // self.stride[0] + 1
        n_w = (img_size[1] - self.patch_size[1]) // self.stride[1] + 1

        pad_h = self.patch_size[0] + n_h * self.stride[0] - img_size[0]
        pad_w = self.patch_size[1] + n_w * self.stride[1] - img_size[1]

        return (img_size[0] + pad_h, img_size[1] + pad_w), (pad_h, pad_w)

    def adjoint_config(self) -> "TiledPConv2dConfig":
        """
        Create a new config for adjoint operations.
        """
        if self.psf_size is None:
            return self
        expansion = self.margin
        return TiledPConv2dConfig(
            patch_size=_add_tuple(self.patch_size, expansion),
            stride=self.stride,
            psf_size_init=self.psf_size,
        )


# =============================================================================
# PATCH HANDLER CLASS
# =============================================================================
class TiledPConv2dHandler:
    """
    Handler for extraction and reconstruction of image patches.

    It is helpful for implementing tiled operations, such as :class:`deepinv.physics.TiledSpaceVaryingBlur`.

    :param TiledPConv2dConfig config: a :class:`deepinv.physics.functional.TiledPConv2dConfig` instance with patch parameters.

    """

    def __init__(self, config: TiledPConv2dConfig):
        self.config = config

    def image_to_patches(self, image: Tensor) -> Tensor:
        r"""
        Split an image into overlapping patches.

        The image will be padded if necessary to ensure all patches have the same size.

        :param torch.Tensor image: Input image tensor of shape `(B, C, H, W)`.
        :return: Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
        """
        patch_size = self.config.patch_size
        stride = self.config.stride

        img_size = image.shape[-2:]

        # Pad image if necessary for even patch extraction
        __, to_pad = self.config._get_compatible_img_size(img_size)

        if to_pad[0] > 0 or to_pad[1] > 0:
            image = F.pad(image, (0, to_pad[1], 0, to_pad[0]))

        # Extract patches using unfold
        patches = image.unfold(2, patch_size[0], stride[0]).unfold(
            3, patch_size[1], stride[1]
        )
        return patches.contiguous()

    def patches_to_image(
        self, patches: Tensor, img_size: tuple[int, int] | None = None
    ) -> Tensor:
        r"""
        Reconstruct an image from overlapping patches.

        This is the inverse operation of `image_to_patches`. Note that overlapping
        regions are summed, so proper normalization may be needed for correct reconstruction.

        :param torch.Tensor patches: Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
        :param img_size: Target output size (height, width). If provided, output is cropped.
        :return: Reconstructed image tensor of shape `(B, C, H, W)`.
        """
        stride = self.config.stride

        B, C, num_patches_h, num_patches_w, h, w = patches.size()

        output_size = (
            h + (num_patches_h - 1) * stride[0],
            w + (num_patches_w - 1) * stride[1],
        )

        # Reshape: (B, C, n_h, n_w, h, w) -> (B, n_h*n_w, C, h, w) -> (B, C*h*w, n_h*n_w)
        num_patches = num_patches_h * num_patches_w
        patches = (
            patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, num_patches, C, h, w)
        )
        patches = patches.view(B, num_patches, C * h * w).permute(0, 2, 1)

        # Fold patches back into image
        output = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(h, w),
            stride=stride,
        )

        # Crop to target size if specified
        if img_size is not None:
            img_size = _as_pair(img_size)
            output = output[:, :, : img_size[0], : img_size[1]]

        return output.contiguous()


# =============================================================================
# UNITY PARTITION FUNCTIONS
# =============================================================================
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
