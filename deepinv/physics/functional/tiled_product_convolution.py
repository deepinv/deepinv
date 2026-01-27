from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch import Tensor
from deepinv.physics.generator.blur import bump_function
import torch
import torch.nn.functional as F
import math
from einops import rearrange
from typing import Callable


# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================


@dataclass
class TiledPConv2dConfig:
    """Configuration for tiled patch-based operations.

    This dataclass centralizes all patch-related parameters and provides
    computed properties for derived values like stride and margin.

    :param patch_size: Size of each patch (height, width) or single int for square patches.
    :param overlap: Overlap between adjacent patches (height, width) or single int.
    :param psf_size: Size of the PSF/kernel (height, width) or single int. Optional.
    """

    patch_size: int | tuple[int, int]
    overlap: int | tuple[int, int]
    psf_size: int | tuple[int, int] | None = None

    def __post_init__(self):
        """Normalize all sizes to tuples after initialization."""
        self.patch_size = _as_pair(self.patch_size)
        self.overlap = _as_pair(self.overlap)
        if self.psf_size is not None:
            self.psf_size = _as_pair(self.psf_size)

    @property
    def stride(self) -> tuple[int, int]:
        """Compute stride between patches: patch_size - overlap."""
        return _add_tuple(self.patch_size, self.overlap, -1)

    @property
    def margin(self) -> tuple[int, int]:
        """Compute margin based on PSF size: psf_size - 1."""
        if self.psf_size is None:
            return (0, 0)
        return (self.psf_size[0] - 1, self.psf_size[1] - 1)

    def compute_num_patches(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """Compute the number of patches that fit in the image.

        :param img_size: Image size (height, width).
        :return: Number of patches (n_rows, n_cols).
        """
        img_size = _as_pair(img_size)
        stride = self.stride
        return (
            (img_size[0] - self.patch_size[0]) // stride[0] + 1,
            (img_size[1] - self.patch_size[1]) // stride[1] + 1,
        )

    def compute_max_size(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """Compute maximum image size that can be evenly split into patches.

        :param img_size: Image size (height, width).
        :return: Maximum size (height, width).
        """
        num_patches = self.compute_num_patches(img_size)
        stride = self.stride
        return (
            self.patch_size[0] + (num_patches[0] - 1) * stride[0],
            self.patch_size[1] + (num_patches[1] - 1) * stride[1],
        )

    def compute_padding(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """Compute padding needed to make image compatible with patch extraction.

        :param img_size: Image size (height, width).
        :return: Padding (pad_h, pad_w).
        """
        img_size = _as_pair(img_size)
        stride = self.stride
        n_h, n_w = self.compute_num_patches(img_size)
        pad_h = self.patch_size[0] + n_h * stride[0] - img_size[0]
        pad_w = self.patch_size[1] + n_w * stride[1] - img_size[1]
        return (pad_h, pad_w)

    def with_psf_expansion(self) -> "TiledPConv2dConfig":
        """Create a new config with patch_size and overlap expanded by PSF margin.

        Used for adjoint operations where we need larger patches.

        :return: New TiledPConv2dConfig with expanded sizes.
        """
        if self.psf_size is None:
            return self
        expansion = self.margin
        return TiledPConv2dConfig(
            patch_size=_add_tuple(self.patch_size, expansion),
            overlap=_add_tuple(self.overlap, expansion),
            psf_size=self.psf_size,
        )

    @classmethod
    def from_tensors(
        cls, w: Tensor, h: Tensor, overlap: int | tuple[int, int]
    ) -> "TiledPConv2dConfig":
        """Create config from weight and PSF tensors.

        :param w: Weight tensor of shape (b, c, K, patch_h, patch_w).
        :param h: PSF tensor of shape (b, c, K, psf_h, psf_w).
        :param overlap: Overlap between patches.
        :return: TiledPConv2dConfig instance.
        """
        return cls(
            patch_size=w.shape[-2:],
            overlap=overlap,
            psf_size=h.shape[-2:],
        )


# =============================================================================
# PATCH HANDLER CLASS
# =============================================================================


class TiledPConv2dHandler:
    """Handler for patch-based image operations.

    This class provides methods for extracting patches from images,
    reconstructing images from patches, and related utilities.

    :param config: TiledPConv2dConfig instance with patch parameters.
    """

    def __init__(self, config: TiledPConv2dConfig):
        self.config = config

    @property
    def patch_size(self) -> tuple[int, int]:
        return self.config.patch_size

    @property
    def overlap(self) -> tuple[int, int]:
        return self.config.overlap

    @property
    def stride(self) -> tuple[int, int]:
        return self.config.stride

    def extract_patches(self, image: Tensor) -> Tensor:
        """Extract patches from an image.

        :param image: Input image tensor of shape (B, C, H, W).
        :return: Patches tensor of shape (B, C, n_rows, n_cols, patch_h, patch_w).
        """
        return image_to_patches(image, self.patch_size, self.overlap)

    def reconstruct_image(
        self, patches: Tensor, img_size: tuple[int, int] | None = None
    ) -> Tensor:
        """Reconstruct image from patches.

        :param patches: Patches tensor of shape (B, C, n_rows, n_cols, patch_h, patch_w).
        :param img_size: Target image size (height, width). Optional.
        :return: Reconstructed image tensor of shape (B, C, H, W).
        """
        return patches_to_image(patches, self.overlap, img_size)

    def get_compatible_img_size(
        self, img_size: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Get compatible image size and required padding.

        :param img_size: Original image size (height, width).
        :return: Tuple of (compatible_size, padding).
        """
        return to_compatible_img_size(img_size, self.patch_size, self.overlap)


# =============================================================================
# TILED PRODUCT CONVOLUTION OPERATIONS
# =============================================================================

from deepinv.physics.functional.convolution import _prepare_filter_for_grouped


def tiled_product_conv2d(
    conv2d_fn: Callable,
    x: Tensor,
    w: Tensor,
    h: Tensor,
    overlap: int | tuple[int, int] = (128, 128),
) -> Tensor:
    r"""Product-convolution operator in 2d.

    Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions.
    <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    The convolution is done by patches, using only 'valid' padding.
    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product,
    :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (b, c, K, patch_size, patch_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (b, c, K, psf_size, psf_size). b in {1, B} and c in {1, C}, h<=H and w<=W
        where `K` is the number of patches.

    :return: torch.Tensor the blurry image.
    """
    config = TiledPConv2dConfig.from_tensors(w, h, overlap)
    handler = TiledPConv2dHandler(config)

    # Extract patches: (B, C, K1, K2, P1, P2)
    patches = handler.extract_patches(x)

    n_rows, n_cols = patches.size(2), patches.size(3)
    assert n_rows * n_cols == h.size(2), (
        f"The number of patches must be equal to the number of PSFs, "
        f"got {n_rows * n_cols} and {h.size(2)}"
    )

    # Flatten K1 and K2 to: (B, C, K, P1, P2)
    patches = patches.flatten(2, 3)
    patches = patches * w

    # Pad for convolution
    margin = config.margin
    patches = F.pad(
        patches,
        pad=(margin[1], margin[1], margin[0], margin[0]),
        value=0,
        mode="constant",
    )

    # Apply convolution per patch
    B, C = patches.shape[:2]
    h = _prepare_filter_for_grouped(h, B=B, C=C)

    result = conv2d_fn(
        rearrange(patches, "b c k h w -> (b k) c h w").contiguous(),
        rearrange(h, "b c k h w -> (b k) c h w").contiguous(),
        padding="valid",
    )

    result = rearrange(
        result, "(b k) c h w -> b c k h w", b=patches.size(0), k=h.size(2)
    )

    # Reconstruct image using handler with expanded overlap
    B, C, K, H, W = result.size()
    expanded_config = config.with_psf_expansion()
    expanded_handler = TiledPConv2dHandler(expanded_config)
    target_size = _add_tuple(x.shape[-2:], margin)

    result = expanded_handler.reconstruct_image(
        result.view(B, C, n_rows, n_cols, H, W),
        img_size=target_size,
    )

    # Remove margin
    return result[..., margin[0] : -margin[0], margin[1] : -margin[1]]


def tiled_product_conv2d_adjoint(
    conv2d_adjoint_fn: Callable,
    y: Tensor,
    w: Tensor,
    h: Tensor,
    overlap: int | tuple[int, int] = (128, 128),
) -> Tensor:
    r"""Adjoint of the product-convolution operator in 2d.

    Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions.
    <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    The convolution is done by patches, using only 'valid' padding.
    This is the adjoint of the forward operator:

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product,
    :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor y: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (b, c, K, patch_size, patch_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (b, c, K, psf_size, psf_size). b in {1, B} and c in {1, C}, h<=H and w<=W

    :return: torch.Tensor x
    """
    config = TiledPConv2dConfig.from_tensors(w, h, overlap)
    expanded_config = config.with_psf_expansion()

    margin = config.margin
    original_img_size = _add_tuple(y.shape[-2:], margin)

    # Pad input
    y = F.pad(
        y,
        pad=(margin[1], margin[1], margin[0], margin[0]),
        value=0,
        mode="constant",
    )

    # Extract patches with expanded config
    handler = TiledPConv2dHandler(expanded_config)
    patches = handler.extract_patches(y)

    n_rows, n_cols = patches.size(2), patches.size(3)
    assert n_rows * n_cols == h.size(
        2
    ), "The number of patches must be equal to the number of PSFs"

    # Apply transpose convolution per patch
    patches = patches.flatten(2, 3)
    B, C = patches.shape[:2]
    h = _prepare_filter_for_grouped(h, B=B, C=C)

    result = conv2d_adjoint_fn(
        rearrange(patches, "b c k h w -> (b k) c h w").contiguous(),
        rearrange(h, "b c k h w -> (b k) c h w").contiguous(),
        padding="valid",
    )

    result = rearrange(result, "(b k) c h w -> b c k h w", b=B, k=h.size(2))

    # Remove margin and apply weights
    result = result[..., margin[0] : -margin[0], margin[1] : -margin[1]]
    result = result * w

    # Reconstruct image using original config's handler
    B, C, _, H, W = result.size()
    original_handler = TiledPConv2dHandler(config)
    return original_handler.reconstruct_image(
        result.view(B, C, n_rows, n_cols, H, W),
        img_size=original_img_size,
    )


def generate_tiled_multipliers(img_size, patch_size, overlap, kernel_size, device):
    masks = unity_partition_function_2d(img_size, patch_size, overlap)
    w, _ = crop_unity_partition_2d(masks, patch_size, overlap, kernel_size)
    return w.flatten(0, 1).unsqueeze(0).unsqueeze(0).to(device)


# =============================================================================
# PSF EXTRACTION
# =============================================================================


def get_psf_pconv2d_patch(
    h: Tensor,
    w: Tensor,
    position: tuple[int],
    overlap: tuple[int],
    num_patches: tuple[int],
):
    r"""Get the PSF at the given position for tiled product convolution.

    :param torch.Tensor h: PSF tensor of size (B, C, K, h, w). h<=H and w<=W
    :param torch.Tensor w: Weight tensor of size (b, C, K, H, W). b in {1, B} and c in {1, C}
    :param tuple[int] position: Position of the PSF patch (B, N, 2)
    :param tuple[int] overlap: Overlap between PSF patches
    :param tuple[int] num_patches: Number of PSF patches (n_rows, n_cols)

    :return: PSF at the given position (B, C, N, psf_size, psf_size)
    """
    patch_size = w.shape[-2:]
    B, N = position.shape[:2]

    # Collect indices and positions for all positions
    indices = _collect_position_indices(position, patch_size, overlap, num_patches)

    # Convert to tensors
    tensors = _indices_to_tensors(indices, h.device, B, N)

    # Select and combine PSFs
    return _compute_combined_psf(h, w, tensors, num_patches)


def _collect_position_indices(
    position: Tensor,
    patch_size: tuple[int, int],
    overlap: tuple[int, int],
    num_patches: tuple[int, int],
) -> dict:
    """Collect patch indices and positions for all query positions.

    :param position: Position tensor of shape (B, N, 2).
    :param patch_size: Size of each patch.
    :param overlap: Overlap between patches.
    :param num_patches: Number of patches (n_rows, n_cols).
    :return: Dictionary with collected indices and positions.
    """
    index_h, index_w = [], []
    patch_position_h, patch_position_w = [], []

    for p in position.tolist():
        for pos in p:
            ih, iw, ph, pw = get_index_and_position(
                pos, patch_size, overlap, num_patches
            )
            index_h.append(ih)
            index_w.append(iw)
            patch_position_h.append(ph)
            patch_position_w.append(pw)

    return {
        "index_h": index_h,
        "index_w": index_w,
        "patch_position_h": patch_position_h,
        "patch_position_w": patch_position_w,
    }


def _indices_to_tensors(indices: dict, device: torch.device, B: int, N: int) -> dict:
    """Convert collected indices to padded tensors.

    :param indices: Dictionary with index lists.
    :param device: Target device for tensors.
    :param B: Batch size.
    :param N: Number of positions.
    :return: Dictionary with tensor versions of indices.
    """
    index_h, weight_h = _pad_sublist(indices["index_h"])
    index_w, weight_w = _pad_sublist(indices["index_w"])
    patch_position_h, _ = _pad_sublist(indices["patch_position_h"])
    patch_position_w, _ = _pad_sublist(indices["patch_position_w"])

    def to_tensor(data):
        return torch.tensor(data, device=device, dtype=torch.long).view(B, N, -1)

    return {
        "index_h": to_tensor(index_h),
        "index_w": to_tensor(index_w),
        "weight_h": to_tensor(weight_h),
        "weight_w": to_tensor(weight_w),
        "patch_position_h": to_tensor(patch_position_h),
        "patch_position_w": to_tensor(patch_position_w),
    }


def _compute_combined_psf(
    h: Tensor, w: Tensor, tensors: dict, num_patches: tuple[int, int]
) -> Tensor:
    """Compute the combined PSF from selected patches.

    :param h: PSF tensor.
    :param w: Weight tensor.
    :param tensors: Dictionary with index tensors.
    :param num_patches: Number of patches (n_rows, n_cols).
    :return: Combined PSF tensor.
    """
    # Reshape PSF and weights to grid layout
    h = h.view(
        h.size(0), h.size(1), num_patches[0], num_patches[1], h.size(3), h.size(4)
    )
    w = w.view(
        w.size(0), w.size(1), num_patches[0], num_patches[1], w.size(3), w.size(4)
    )

    # Create batch indices
    batch_idx = torch.arange(h.size(0), device=h.device, dtype=torch.long)[
        :, None, None, None
    ]

    # Select relevant PSFs and weights
    h_selected = h[
        batch_idx,
        :,
        tensors["index_h"][..., None, :],
        tensors["index_w"][..., :, None],
        ...,
    ]
    w_selected = w[
        batch_idx,
        :,
        tensors["index_h"][..., None, :],
        tensors["index_w"][..., :, None],
        tensors["patch_position_h"][..., None, :],
        tensors["patch_position_w"][..., :, None],
    ]

    # Compute weighted combination
    weight = tensors["weight_h"][..., None, :] * tensors["weight_w"][..., None]
    psf = torch.sum(
        h_selected * w_selected[..., None, None] * weight[..., None, None, None],
        dim=(2, 3),
    )
    return psf.transpose(1, 2)


def get_index_and_position(
    position: tuple[int],
    patch_size: tuple[int],
    overlap: tuple[int],
    num_patches: tuple[int],
):
    r"""Get the patch indices and local positions for a given image position.

    Determines which patches contain the given position and computes the
    local coordinates within each patch.

    :param tuple[int] position: Position in image coordinates (row, column)
    :param tuple[int] patch_size: Size of each patch (height, width)
    :param tuple[int] overlap: Overlap between patches (height, width)
    :param tuple[int] num_patches: Number of patches (n_rows, n_cols)

    :return: Tuple of (index_h, index_w, patch_position_h, patch_position_w)
    :raises ValueError: If position is outside all patches.
    """
    overlap = _as_pair(overlap)
    patch_size = _as_pair(patch_size)
    stride = _add_tuple(patch_size, overlap, -1)

    if isinstance(position, torch.Tensor):
        position = position.tolist()

    max_size = (
        patch_size[0] + (num_patches[0] - 1) * stride[0],
        patch_size[1] + (num_patches[1] - 1) * stride[1],
    )

    # Compute search range for patches
    n_min = (
        max(math.floor((position[0] - patch_size[0]) / stride[0]), 0),
        max(math.floor((position[1] - patch_size[1]) / stride[1]), 0),
    )
    n_max = (
        math.floor(position[0] / stride[0]),
        math.floor(position[1] / stride[1]),
    )

    index_h, patch_position_h = _find_containing_patches_1d(
        position[0], n_min[0], n_max[0], stride[0], patch_size[0], max_size[0]
    )
    index_w, patch_position_w = _find_containing_patches_1d(
        position[1], n_min[1], n_max[1], stride[1], patch_size[1], max_size[1]
    )

    if len(index_h) == 0 or len(index_w) == 0:
        raise ValueError(f"The position center {position} is not valid.")

    return index_h, index_w, patch_position_h, patch_position_w


def _find_containing_patches_1d(
    pos: int, n_min: int, n_max: int, stride: int, patch_size: int, max_size: int
) -> tuple[list[int], list[int]]:
    """Find patches containing a position along one dimension.

    :param pos: Position coordinate.
    :param n_min: Minimum patch index to check.
    :param n_max: Maximum patch index to check.
    :param stride: Stride between patches.
    :param patch_size: Size of each patch.
    :param max_size: Maximum valid coordinate.
    :return: Tuple of (patch_indices, local_positions).
    """
    indices = []
    positions = []

    for i in range(n_min, n_max + 1):
        left = i * stride
        right = left + patch_size

        if left <= pos < right <= max_size:
            indices.append(i)
            positions.append(pos - i * stride)

    return indices, positions


# =============================================================================
# UNITY PARTITION FUNCTIONS
# =============================================================================


def unity_partition_function_1d(
    image_size: int,
    patch_size: int,
    overlap: int,
    mode: Literal["bump", "linear"] = "bump",
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
    t = torch.linspace(-max_size // 2, max_size // 2, max_size)

    if mode.lower() == "bump":
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
    masks = masks / masks.sum(dim=0, keepdims=True)
    return masks


def unity_partition_function_2d(
    image_size: tuple[int],
    patch_size: tuple[int],
    overlap: tuple[int],
    mode: Literal["bump", "linear"] = "bump",
) -> Tensor:
    r"""Create a 2D unity partition function for smooth patch blending.

    Creates 2D masks by combining 1D partition functions. The masks sum to 1
    across all patches, enabling smooth blending in overlap regions.

    :param tuple image_size: Image size (height, width).
    :param tuple patch_size: Patch size (height, width).
    :param tuple overlap: Overlap size (height, width).
    :param str mode: Blending mode - 'bump' (smooth) or 'linear'.
    :return: Tensor of shape (n_rows, n_cols, patch_h, patch_w) with partition masks.
    """
    image_size = _as_pair(image_size)
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)

    masks_x = unity_partition_function_1d(
        image_size[0], patch_size[0], overlap[0], mode
    )
    masks_y = unity_partition_function_1d(
        image_size[1], patch_size[1], overlap[1], mode
    )

    # Combine 1D masks into 2D via outer product
    masks = torch.tensordot(masks_x, masks_y, dims=0)
    masks = masks.permute(0, 2, 1, 3)

    # Normalize to sum to 1
    masks = masks / (masks.sum(dim=(0, 1), keepdims=True) + 1e-8)
    return masks


def crop_unity_partition_2d(
    masks: Tensor,
    patch_size: tuple[int],
    overlap: tuple[int],
    psf_size: tuple[int],
) -> tuple[Tensor, Tensor]:
    r"""Crop unity partition masks to patch regions.

    Extracts the relevant portion of each partition mask corresponding to
    its patch location.

    :param torch.Tensor masks: Full masks of shape (K_h, K_w, H, W).
    :param tuple patch_size: Patch size (height, width).
    :param tuple overlap: Overlap size (height, width).
    :param tuple psf_size: PSF size for computing center indices.
    :return: Tuple of:
        - cropped_masks: Tensor of shape (K_h, K_w, patch_h, patch_w)
        - index: Tensor of shape (K_h, K_w, 2) with mask center indices
    """
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)
    psf_size = _as_pair(psf_size)

    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])
    n_rows, n_cols = masks.size(0), masks.size(1)

    # Create grid indices for vectorized extraction
    row_idx = torch.arange(n_rows, device=masks.device)
    col_idx = torch.arange(n_cols, device=masks.device)
    h_starts = row_idx * stride[0]
    w_starts = col_idx * stride[1]

    # Build index tensor: shape (K_h, K_w, 2)
    index = torch.stack(
        torch.meshgrid(
            h_starts + psf_size[0] // 2, w_starts + psf_size[1] // 2, indexing="ij"
        ),
        dim=-1,
    ).float()

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

    return cropped_masks, index


# =============================================================================
# PATCH UTILITY FUNCTIONS
# =============================================================================


def to_compatible_img_size(
    img_size: tuple[int], patch_size: tuple[int], overlap: tuple[int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Compute image size compatible with patch extraction and required padding.

    :param tuple img_size: Original image size (height, width).
    :param tuple patch_size: Patch size (height, width).
    :param tuple overlap: Overlap between patches (height, width).
    :return: Tuple of (compatible_size, padding_needed).
    """
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)
    img_size = _as_pair(img_size)

    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])

    # Compute number of patches and required padding
    n_h = (img_size[0] - patch_size[0]) // stride[0] + 1
    n_w = (img_size[1] - patch_size[1]) // stride[1] + 1

    pad_h = patch_size[0] + n_h * stride[0] - img_size[0]
    pad_w = patch_size[1] + n_w * stride[1] - img_size[1]

    return (img_size[0] + pad_h, img_size[1] + pad_w), (pad_h, pad_w)


def image_to_patches(
    image: Tensor, patch_size: int | tuple[int, int], overlap: int | tuple[int, int]
) -> Tensor:
    """Split an image into overlapping patches.

    The image will be padded if necessary to ensure all patches have the same size.

    :param torch.Tensor image: Input image tensor of shape (B, C, H, W).
    :param patch_size: Size of each patch (height, width) or single int.
    :param overlap: Overlap between adjacent patches (height, width) or single int.
    :return: Patches tensor of shape (B, C, n_rows, n_cols, patch_h, patch_w).
    :raises TypeError: If image is not a torch.Tensor.
    :raises AssertionError: If patch_size <= overlap.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image should be a torch.Tensor")

    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)
    img_size = image.shape[-2:]
    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])

    assert stride[0] > 0 and stride[1] > 0, "Patch size must be greater than overlap"

    # Pad image if necessary for even patch extraction
    __, to_pad = to_compatible_img_size(img_size, patch_size, overlap)
    if to_pad[0] > 0 or to_pad[1] > 0:
        image = F.pad(image, (0, to_pad[1], 0, to_pad[0]))

    # Extract patches using unfold
    patches = image.unfold(2, patch_size[0], stride[0]).unfold(
        3, patch_size[1], stride[1]
    )
    return patches.contiguous()


def patches_to_image(
    patches: Tensor,
    overlap: int | tuple[int, int],
    img_size: int | tuple[int, int] = None,
) -> Tensor:
    """Reconstruct an image from overlapping patches.

    This is the inverse operation of `image_to_patches`. Note that overlapping
    regions are summed, so proper normalization (e.g., using unity partition
    functions) may be needed for correct reconstruction.

    :param torch.Tensor patches: Patches tensor of shape (B, C, n_rows, n_cols, patch_h, patch_w).
    :param overlap: Overlap between patches (height, width) or single int.
    :param img_size: Target output size (height, width). If provided, output is cropped.
    :return: Reconstructed image tensor of shape (B, C, H, W).
    :raises TypeError: If patches is not a torch.Tensor.
    """
    if not isinstance(patches, torch.Tensor):
        raise TypeError("Patches should be a torch.Tensor")

    overlap = _as_pair(overlap)
    B, C, num_patches_h, num_patches_w, h, w = patches.size()
    stride = (h - overlap[0], w - overlap[1])

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
# PRIVATE HELPER FUNCTIONS
# =============================================================================


def _add_tuple(a: tuple, b: tuple, constant: float = 1) -> tuple:
    """Add two tuples element-wise with optional scaling of second tuple.

    Computes: output[i] = a[i] + b[i] * constant

    :param tuple a: First tuple.
    :param tuple b: Second tuple (must have same length as a).
    :param float constant: Scalar multiplier for b. Default is 1.
    :return: Result tuple.
    :raises AssertionError: If tuples have different lengths.
    """
    assert len(a) == len(b), "Input tuples must have the same length"
    return tuple(a[i] + constant * b[i] for i in range(len(a)))


def _as_pair(value: int | tuple | list) -> tuple[int, int]:
    """Ensure value is a 2-tuple.

    :param value: Integer (duplicated) or tuple/list (last 2 elements used).
    :return: 2-tuple of integers.
    :raises ValueError: If tuple/list has fewer than 2 elements.
    :raises TypeError: If value is neither int nor tuple/list.
    """
    if isinstance(value, int):
        return (value, value)
    elif isinstance(value, (tuple, list)):
        if len(value) >= 2:
            return tuple(value[-2:])
        else:
            raise ValueError("Tuple/list must have at least 2 elements.")
    else:
        raise TypeError(f"Expected int or tuple/list, got {type(value).__name__}")


def _pad_sublist(input_list: list[list]) -> tuple[list[list], list[list[int]]]:
    """Pad sublists to equal length by repeating last elements."""
    max_length = max(len(sublist) for sublist in input_list)

    padded_weights = [
        [1] * len(sublist) + [0] * (max_length - len(sublist)) for sublist in input_list
    ]
    padded_lists = [
        sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in input_list
    ]

    return padded_lists, padded_weights
