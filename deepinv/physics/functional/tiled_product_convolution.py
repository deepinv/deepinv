from torch import Tensor

from deepinv.physics.functional import conv2d, conv_transpose2d
from deepinv.physics.generator.blur import bump_function
import torch
import torch.nn.functional as F
import math


# METHOD 2: PRODUCT CONVOLUTION USING PATCHES
def tiled_product_convolution2d(
    x: Tensor,
    w: Tensor,
    h: Tensor,
    overlap: int | tuple[int, int] = (128, 128),
) -> Tensor:
    r"""

    Product-convolution operator in 2d. Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions. <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    The convolution is done by patches, using only 'valid' padding.
    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor x: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (b, c, K, patch_size, patch_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (b, c, K, psf_size, psf_size). b in {1, B} and c in {1, C}, h<=H and w<=W
        where `K` is the number of patches.

    :return: torch.Tensor the blurry image.
    """
    patch_size = w.shape[-2:]
    overlap = _as_pair(overlap)
    psf_size = h.shape[-2:]

    patches = image_to_patches(
        x, patch_size=patch_size, overlap=overlap
    )  # (B, C, K1, K2, P1, P2)

    n_rows, n_cols = patches.size(2), patches.size(3)
    assert n_rows * n_cols == h.size(
        2
    ), f"The number of patches must be equal to the number of PSFs, got {n_rows*n_cols} and {h.size(2)}"

    # Flatten K1 and K2 to: (B, C, K, P1, P2)
    patches = patches.flatten(2, 3)
    patches = patches * w

    patches = F.pad(
        patches,
        pad=(psf_size[1] - 1, psf_size[1] - 1, psf_size[0] - 1, psf_size[0] - 1),
        value=0,
        mode="constant",
    )

    result = []
    for k in range(h.size(2)):
        result.append(
            conv2d(
                patches[:, :, k, ...],
                h[:, :, k, ...],
                padding="valid",
            )
        )
    # (B, C, K, H', W')
    result = torch.stack(result, dim=2)
    margin = (psf_size[0] - 1, psf_size[1] - 1)
    B, C, K, H, W = result.size()

    result = patches_to_image(
        result.view(B, C, n_rows, n_cols, H, W),
        _add_tuple(overlap, _add_tuple(psf_size, (-1,) * len(psf_size))),
        img_size=_add_tuple(x.shape[-2:], margin),
    )
    return result[..., margin[0] : -margin[0], margin[1] : -margin[1]]


def tiled_product_convolution2d_adjoint(
    y: Tensor,
    w: Tensor,
    h: Tensor,
    overlap: tuple[int] = (128, 128),
) -> Tensor:
    r"""

    Product-convolution operator in 2d. Details available in the following paper:

    Escande, P., & Weiss, P. (2017).
    `Approximation of integral operators using product-convolution expansions. <https://hal.science/hal-01301235/file/Approximation_Integral_Operators_Convolution-Product_Expansion_Escande_Weiss_2016.pdf>`_
    Journal of Mathematical Imaging and Vision, 58, 333-348.

    The convolution is done by patches, using only 'valid' padding.
    This forward operator performs

    .. math::

        y = \sum_{k=1}^K h_k \star (w_k \odot x)

    where :math:`\star` is a convolution, :math:`\odot` is a Hadamard product, :math:`w_k` are multipliers :math:`h_k` are filters.

    :param torch.Tensor y: Tensor of size (B, C, H, W)
    :param torch.Tensor w: Tensor of size (b, c, K, patch_size, patch_size). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (b, c, K, psf_size, psf_size). b in {1, B} and c in {1, C}, h<=H and w<=W

    :return: torch.Tensor x
    """
    patch_size = w.shape[-2:]
    overlap = _as_pair(overlap)
    psf_size = h.shape[-2:]
    original_img_size = _add_tuple(
        y.shape[-2:], _add_tuple(psf_size, (-1,) * len(psf_size))
    )

    margin = (psf_size[0] - 1, psf_size[1] - 1)
    y = F.pad(
        y,
        pad=(margin[1], margin[1], margin[0], margin[0]),
        value=0,
        mode="constant",
    )

    patches = image_to_patches(
        y,
        patch_size=_add_tuple(patch_size, _add_tuple(psf_size, (-1,) * len(psf_size))),
        overlap=_add_tuple(overlap, _add_tuple(psf_size, (-1,) * len(psf_size))),
    )
    # (B, C, K1, K2, P1, P2)

    n_rows, n_cols = patches.size(2), patches.size(3)
    assert n_rows * n_cols == h.size(
        2
    ), "The number of patches must be equal to the number of PSFs"

    patches = patches.flatten(2, 3)
    result = []
    for k in range(h.size(2)):
        result.append(
            conv_transpose2d(patches[:, :, k, ...], h[:, :, k, ...], padding="valid"),
        )
    # (B, C, K, H', W')
    result = torch.stack(result, dim=2)
    margin = (psf_size[0] - 1, psf_size[1] - 1)
    result = result[..., margin[0] : -margin[0], margin[1] : -margin[1]]
    result = result * w
    B, C, _, H, W = result.size()

    return patches_to_image(
        result.view(B, C, n_rows, n_cols, H, W), overlap, img_size=original_img_size
    )


def get_psf_pconv2d_patch(
    h: Tensor,
    w: Tensor,
    position: tuple[int],
    overlap: tuple[int],
    num_patches: tuple[int],
):
    r"""
    Get the PSF at the given position of the :meth:`deepinv.physics.functional.tiled_product_convolution2d` function.

    :param torch.Tensor w: Tensor of size (b, C, K, H, W). b in {1, B} and c in {1, C}
    :param torch.Tensor h: Tensor of size (B, C, K, h, w). h<=H and w<=W

    :param tuple[int] position: Position of the PSF patch (B, N, 2)
    :param tuple[int] overlap: Overlap between PSF patches
    :param tuple[int] num_patches: Number of PSF patches


    :return: PSF at the given position (B, C, N, psf_size, psf_size)
    """
    patch_size = w.shape[-2:]
    index_h = []
    index_w = []
    patch_position_h = []
    patch_position_w = []

    B, N = position.shape[:2]
    # Possible to use torch.vmap to parallelize these loops
    for p in position.tolist():
        for pos in p:
            ih, iw, ph, pw = get_index_and_position(
                pos, patch_size, overlap, num_patches
            )

            index_h.append(ih)
            index_w.append(iw)
            patch_position_h.append(ph)
            patch_position_w.append(pw)
    index_h, weight_h = pad_sublist(index_h)
    index_w, weight_w = pad_sublist(index_w)
    patch_position_h, _ = pad_sublist(patch_position_h)
    patch_position_w, _ = pad_sublist(patch_position_w)

    index_h = torch.tensor(index_h, device=h.device, dtype=torch.long).view(B, N, -1)
    index_w = torch.tensor(index_w, device=h.device, dtype=torch.long).view(B, N, -1)
    weight_h = torch.tensor(weight_h, device=h.device, dtype=torch.long).view(B, N, -1)
    weight_w = torch.tensor(weight_w, device=h.device, dtype=torch.long).view(B, N, -1)

    patch_position_h = torch.tensor(
        patch_position_h, device=h.device, dtype=torch.long
    ).view(B, N, -1)
    patch_position_w = torch.tensor(
        patch_position_w, device=h.device, dtype=torch.long
    ).view(B, N, -1)

    # Reshape the PSF and the multipliers
    h = h.view(
        h.size(0), h.size(1), num_patches[0], num_patches[1], h.size(3), h.size(4)
    )
    w = w.view(
        w.size(0), w.size(1), num_patches[0], num_patches[1], w.size(3), w.size(4)
    )

    h_selected = h[
        torch.arange(h.size(0), device=h.device, dtype=torch.long)[:, None, None, None],
        :,
        index_h[..., None, :],
        index_w[..., :, None],
        ...,
    ]
    w_selected = w[
        torch.arange(w.size(0), device=w.device, dtype=torch.long)[:, None, None, None],
        :,
        index_h[..., None, :],
        index_w[..., :, None],
        patch_position_h[..., None, :],
        patch_position_w[..., :, None],
    ]

    weight = weight_h[..., None, :] * weight_w[..., None]
    psf = torch.sum(
        h_selected * w_selected[..., None, None] * weight[..., None, None, None],
        dim=(2, 3),
    )
    return psf.transpose(1, 2)


# UTILITY FUNCTIONS
def get_index_and_position(
    position: tuple[int],
    patch_size: tuple[int],
    overlap: tuple[int],
    num_patches: tuple[int],
):
    r"""
    Get the PSF index at the given position of the :meth:`deepinv.physics.functional.tiled_product_convolution2d` function.

    :param tuple[int] position: Position of the point of which we want to infer the PSf (row, column)
    :param tuple[int] overlap: Overlap between PSF patches
    :param tuple[int] num_patches: Number of PSF patches


    :return: index_h, index_w, patch_position_h, patch_position_w
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

    n_min = (
        math.floor((position[0] - patch_size[0]) / stride[0]),
        math.floor((position[1] - patch_size[1]) / stride[1]),
    )
    n_min = (max(n_min[0], 0), max(n_min[1], 0))
    n_max = (math.floor(position[0] / stride[0]), math.floor(position[1] / stride[1]))

    index_h = []
    index_w = []
    patch_position_h = []
    patch_position_w = []
    # VERTICAL INDEX
    # i = n[0]
    for i in range(n_min[0], n_max[0] + 1):
        left = i * stride[0]
        right = left + patch_size[0]

        if (left <= position[0]) and (position[0] < right) and (right <= max_size[0]):
            index_h.append(i)
            patch_position_h.append(position[0] - i * stride[0])

    # HORIZONTAL INDEX
    for j in range(n_min[1], n_max[1] + 1):
        left = j * stride[1]
        right = left + patch_size[1]

        if (left <= position[1]) and (position[1] < right) and (right <= max_size[1]):
            index_w.append(j)
            patch_position_w.append(position[1] - j * stride[1])

    # print(index_h, index_w, patch_position_h, patch_position_w)
    if len(index_h) == 0 or len(index_w) == 0:
        raise ValueError(f"The position center {position} is not valid.")
    return index_h, index_w, patch_position_h, patch_position_w


def unity_partition_function_1d(
    image_size: int, patch_size: int, overlap: int, mode: str = "bump"
):
    r"""
    Define the partition function, which is 1 on [-a, a] and decrease to 0 on - (a + b) and (a + b)
    where a = patch_size / 2 - overlap and b = overlap.

    :param int image_size: the size the dimension of image
    :param int patch_size: the size of patch along a dimension
    :param int overlap: overlapping size of patch along a dimension
    :param str mode: 'linear' or 'bump' are supported. Defined how the function is decreased.
    """
    n_patch = (image_size - patch_size) // (patch_size - overlap) + 1
    max = patch_size + (n_patch - 1) * (patch_size - overlap)
    t = torch.linspace(-max // 2, max // 2, max)
    if mode.lower() == "bump":
        mask = bump_function(t, patch_size / 2 - overlap, overlap).roll(
            shifts=-max // 2 + patch_size // 2
        )
    elif mode.lower() == "linear":
        a = patch_size / 2 - overlap
        b = overlap

        def linear(x):
            return ((a + b) / b - x.abs().clip(0, a + b) / b).abs().clip(0, 1)

        mask = linear(t).roll(shifts=-max // 2 + patch_size // 2)

    masks = torch.stack(
        [mask.roll(shifts=(patch_size - overlap) * i) for i in range(n_patch)], dim=0
    )
    masks[0, :overlap] = 1.0
    masks[-1, -overlap:] = 1.0
    masks = masks / masks.sum(dim=0, keepdims=True)
    return masks


def unity_partition_function_2d(
    image_size: tuple[int],
    patch_size: tuple[int],
    overlap: tuple[int],
    mode: str = "bump",
):
    r"""
    Unity partition function in 2D. Similar to :meth:`deepinv.physics.functional.product_convolution.unity_partition_function_1d`
    This function will create the mask corresponding to the patches output from :meth:`deepinv.physics.functional.product_convolution.image_to_patches`.

    :param tuple image_size: size of the image in the format (height, width)
    :param tuple patch_size: size of the patch in the format (height, width)
    :param tuple overlap: overlap size of the patch in the format (height, width)
    :param str mode: 'linear' or 'bump' are supported. Defined how the function is decreased. Default to `bump`.
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    masks_x = unity_partition_function_1d(
        image_size[0], patch_size[0], overlap[0], mode
    )
    masks_y = unity_partition_function_1d(
        image_size[1], patch_size[1], overlap[1], mode
    )
    masks = torch.tensordot(masks_x, masks_y, dims=0)
    masks = masks.permute(0, 2, 1, 3)
    masks = masks / (masks.sum(dim=(0, 1), keepdims=True) + 1e-8)
    return masks


def crop_unity_partition_2d(
    masks, patch_size: tuple[int], overlap: tuple[int], psf_size: tuple[int]
):
    r"""
    Crop the mask generated by unity_partition_function_2d.

    :param torch.Tensor masks: mask of shape (K_h, K_w, image_size, image_size) representing the the mask corresponding to each patch.
    :return:
        :rtype Tensor masks: mask of shape (K_h, K_w, patch_size, patch_size) representing the the mask corresponding to each patch.
        :rtype Tensor index: index of shape (K_h, K_w, 2) representing the index of the mask centers.
    """

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(overlap, int):
        overlap = (overlap, overlap)

    if isinstance(psf_size, int):
        psf_size = (psf_size, psf_size)

    diff_h = patch_size[0] - overlap[0]
    diff_w = patch_size[1] - overlap[1]
    supp_h = patch_size[0]
    supp_w = patch_size[1]

    index = torch.zeros(
        masks.size(0), masks.size(1), supp_h, supp_w, device=masks.device
    )
    cropped_masks = torch.zeros(
        masks.size(0), masks.size(1), patch_size[0], patch_size[1], device=masks.device
    )
    for h in range(masks.size(0)):
        for w in range(masks.size(1)):
            cropped_masks[h, w] = masks[
                h, w, h * diff_h : h * diff_h + supp_h, w * diff_w : w * diff_w + supp_w
            ]
            index[h, w, 0] = h * diff_h + psf_size[0] // 2
            index[h, w, 1] = w * diff_w + psf_size[1] // 2

    return cropped_masks, index


# UTILITY FUNCTIONS FOR PATCHES
def to_compatible_img_size(
    img_size: tuple[int], patch_size: tuple[int], overlap: tuple[int]
):
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)
    img_size = _as_pair(img_size)

    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])

    # Zero-pad the image if necessary to make it divisible by the patch extraction
    n_h, n_w = (
        (img_size[0] - patch_size[0]) // stride[0] + 1,
        (img_size[1] - patch_size[1]) // stride[1] + 1,
    )

    pad_h = patch_size[0] + n_h * stride[0] - img_size[0]
    pad_w = patch_size[1] + n_w * stride[1] - img_size[1]

    return (img_size[0] + pad_h, img_size[1] + pad_w), (pad_h, pad_w)


def image_to_patches(
    image: Tensor, patch_size: int | tuple[int, int], overlap: int | tuple[int, int]
):
    """
    Splits an image into patches with specified patch size and overlap.
    The image will be cropped to the appropriate size so that all patches have the same size.

    Parameters:
        image (torch.Tensor): Input image tensor of shape (B, C, H, W).
        patch_size (int): Size of each patch (patch_size x patch_size).
        overlap (int): Overlapping size between patches.

    Returns:
        torch.Tensor: Batch of patches of shape (B, C, n_rows, n_cols, patch_size, patch_size).

    """
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)
    img_size = image.shape[-2:]

    # Ensure image is a tensor
    if not isinstance(image, torch.Tensor):
        raise TypeError("Image should be a torch.Tensor")

    stride = (patch_size[0] - overlap[0], patch_size[1] - overlap[1])

    # Ensure the patch size and overlap are valid
    assert (stride[0] > 0) * (stride[1] > 0), "Patch size must be greater than overlap"

    # Zero-pad the image if necessary to make it divisible by the patch extraction
    __, to_pad = to_compatible_img_size(img_size, patch_size, overlap)
    if to_pad[0] > 0 or to_pad[1] > 0:
        image = F.pad(image, (0, to_pad[1], 0, to_pad[0]))

    patches = image.unfold(2, patch_size[0], stride[0]).unfold(
        3, patch_size[1], stride[1]
    )
    return patches.contiguous()


def patches_to_image(
    patches: Tensor,
    overlap: int | tuple[int, int],
    img_size: int | tuple[int, int] = None,
):
    """
    Reconstruct a batch of images from patches.
    This function is the reverse of  `patches_to_image`

    Parameters:
        patches (torch.Tensor): Input image tensor of shape (B, C, n_rows, n_cols, patch_size, patch_size).
        patch_size (int): Size of each patch (patch_size x patch_size).
        overlap (int): Overlapping size between patches.

    Returns:
        torch.Tensor: Batch of image of shape (B, C, H, W).
    """

    if isinstance(overlap, int):
        overlap = (overlap, overlap)
    B, C, num_patches_h, num_patches_w, h, w = patches.size()

    output_size = (
        h + (num_patches_h - 1) * (h - overlap[0]),
        w + (num_patches_w - 1) * (w - overlap[1]),
    )
    if not isinstance(patches, torch.Tensor):
        raise TypeError("Patches should be a torch.Tensor")

    # Rearrange the patches to have the desired shape (B, num_patches_h * num_patches_w, C, h, w)
    patches = (
        patches.permute(0, 2, 3, 1, 4, 5)
        .contiguous()
        .view(B, num_patches_h * num_patches_w, C, h, w)
    )

    # Now reverse the process using torch.fold
    # Calculate the number of patches
    num_patches = num_patches_h * num_patches_w

    # Reshape the patches to (B, C*h*w, num_patches)
    patches = patches.view(B, num_patches, C * h * w).permute(0, 2, 1)
    stride = (h - overlap[0], w - overlap[1])
    # Fold the patches back into the image
    output = F.fold(
        patches,
        output_size=output_size,
        kernel_size=(h, w),
        stride=stride,
    )
    if img_size is not None:
        img_size = _as_pair(img_size)
        output = output[:, :, : img_size[0], : img_size[1]]

    return output.contiguous()


def _add_tuple(a: tuple, b: tuple, constant: float = 1) -> tuple:
    r"""
    Add 2 tuples element-wise, where the second tuple is multiplied by constant:

        `output[i] = a[i] + b[i] * constant`

    :param tuple a: First tuple
    :param tuple b: Second tuple
    :param float constant: The constant to multiply the second tuple

    :return: (tuple) the output tuple.
    """
    assert len(a) == len(b), "Input must have the same length"
    return tuple(a[i] + constant * b[i] for i in range(len(a)))


def _as_pair(input: tuple | int) -> tuple:
    r"""
    Make sure that the input is a pair.

    :param Union[tuple, int] input: Input to be made a pair.

    :return: (tuple) a pair of the input, if it is already a pair, or a pair with the input and itself.
    """
    if isinstance(input, int):
        return (input, input)
    elif isinstance(input, (tuple, list)):
        if len(input) >= 2:
            return input[-2:]
        else:
            raise ValueError(
                "Input must be an int or a tuple of length greater than 2."
            )
    else:
        raise TypeError("Input must be an int or a tuple of length greater than 2.")


def pad_sublist(input: list):
    """
    Pads each sublist in the given list of `input` to ensure all sublists
    have the same length as the longest sublist.

    The padding is done by repeating the last values of each sublist

    :param list input: List of lists where each sublist may have varying lengths.

    :return: A new list of lists where all sublists have the same length.
    :Example:

    >>> input = [[1, 2, 3], [4, 5], [6]]
    >>> pad_sublist(input, value=0)
    [[1, 2, 3], [4, 5, 0], [6, 0, 0]]
    """
    max_length = max(len(sublist) for sublist in input)
    padded_weight = [
        [1] * len(sublist) + [0] * (max_length - len(sublist)) for sublist in input
    ]
    padded_list = [
        sublist + [sublist[-1]] * (max_length - len(sublist)) for sublist in input
    ]

    return padded_list, padded_weight


def compute_patch_info(
    image_size: tuple[int], patch_size: tuple[int], overlap: tuple[int]
):
    r"""
    Compute all information about the patches generated from the given image of image_size.

    :param tuple[int] image_size: image size in (height, width)
    :param tuple[int] patch_size: patch size in (height, width)
    :param tuple[int] overlap: overlap size in (height, width)

    :returns: (Dictionary) a dictionary containing all information about the patches, containing:
        - stride
        - num_patches (height, width)
        - max_size (height, width): the size of image which can be split into patches.
            Pixels from max_size to image_size will be cropped.
    """

    image_size = _as_pair(image_size)
    patch_size = _as_pair(patch_size)
    overlap = _as_pair(overlap)

    stride = _add_tuple(patch_size, overlap, -1)
    num_patches = (
        (image_size[0] - patch_size[0]) // stride[0] + 1,
        (image_size[1] - patch_size[1]) // stride[1] + 1,
    )

    max_size = (
        patch_size[0] + (num_patches[0] - 1) * stride[0],
        patch_size[1] + (num_patches[1] - 1) * stride[1],
    )
    patch_info = {
        "stride": stride,
        "num_patches": num_patches,
        "max_size": max_size,
    }
    return patch_info
