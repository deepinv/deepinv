import torch
from torch import Tensor

from ._tiling import (
    _image_to_patches_impl,
    _patches_to_image_impl,
    _resolve_tiling_params,
)


def patch_extractor(
    imgs, n_patches, patch_size, duplicates=False, position_inds_linear=None
):
    r"""
    This function takes a B x C x N x M tensor as input and extracts n_patches random patches
    of size C x patch_size x patch_size from each C x N x M image (C=1 for gray value, C=3 for RGB).
    Hence, the output is of shape B x n_patches x C x patch_size x patch_size.

    :param torch.Tensor imgs: Images for cutting out patches. Shape batch size x channels x height x width
    :param int patch_size: size of the patches
    :param bool duplicates: determines if a patch can appear twice.
    :param torch.Tensor position_inds_linear: allows it to cut patches with specific indices (required for the EPLL reconstruction).
        dtype of the tensor should be torch.long.
    """

    B, C, N, M = imgs.shape
    total_patch_number = (N - patch_size + 1) * (M - patch_size + 1)
    n_patches = min(n_patches, total_patch_number)
    if n_patches == -1:
        n_patches = total_patch_number

    # create linear indices for one single patch
    patch = torch.zeros(
        (C, patch_size, patch_size), dtype=torch.long, device=imgs.device
    )
    patch = patch + torch.arange(patch_size, device=imgs.device)[None, None, :]
    patch = patch + M * torch.arange(patch_size, device=imgs.device)[None, :, None]
    patch = patch + (N * M) * torch.arange(C, device=imgs.device)[:, None, None]
    patch = patch.reshape(-1)

    # select patch positions
    if position_inds_linear is None:
        if duplicates:
            position_inds_linear = torch.randint(
                0, total_patch_number, (n_patches,), device=imgs.device
            )
        else:
            position_inds_linear = torch.randperm(
                total_patch_number, device=imgs.device
            )[:n_patches]
    position_inds_height = position_inds_linear // (M - patch_size + 1)
    position_inds_width = position_inds_linear % (M - patch_size + 1)

    # linear indices of the patches in the image
    linear_inds = patch[None, :].tile(n_patches, 1)
    linear_inds = linear_inds + position_inds_width[:, None]
    linear_inds = linear_inds + M * position_inds_height[:, None]
    linear_inds = linear_inds.reshape(-1)

    # cut linear indices from images and reshape the output correctly
    imgs = imgs.reshape(B, -1)
    patches = imgs.view(B, -1)[:, linear_inds]
    patches = patches.reshape(B, n_patches, C, patch_size, patch_size)

    return patches, linear_inds


def image_to_patches(
    image: Tensor,
    patch_size: int | tuple[int, int],
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] | None = None,
    pad_if_needed: bool = True,
) -> torch.Tensor:
    r"""Split a batch of images into overlapping 2D patches.

    The behavior mirrors :class:`deepinv.utils.TiledMixin2d` while exposing a
    functional API.

    :param torch.Tensor image: Input image tensor of shape ``(B, C, H, W)``.
    :param int | tuple[int, int] patch_size: Patch size ``(patch_h, patch_w)``.
        If an ``int`` is provided, the same value is used for both dimensions.
    :param int | tuple[int, int] stride: Stride between adjacent patches as
        ``(stride_h, stride_w)``. If ``None``, defaults to half the patch size.
    :param bool pad_if_needed: If ``True``, adds extra right/bottom padding so
        that all extracted patches have the requested ``patch_size``.
    :return: Patches of shape ``(B, C, n_rows, n_cols, patch_h, patch_w)``.

    |sep|

    :Examples:

    >>> import deepinv as dinv
    >>> from torchvision.utils import make_grid
    >>> x = dinv.utils.load_example('butterfly.png')
    >>> patches = dinv.utils.patchify(x, patch_size=64, stride=32)
    >>> print(f"Input shape: {x.shape}, patchified shape: {patches.shape}")
    Input shape: torch.Size([1, 3, 256, 256]), patchified shape: torch.Size([1, 3, 7, 7, 64, 64])
    >>> list_patch = [patches[0,:, i, j, ...] for i in range(patches.shape[2]) for j in range(patches.shape[3])]
    >>> dinv.utils.plot([x, make_grid(list_patch, nrow=patches.shape[2])], titles=["Original", "Overlapping patches"])  # doctest: +SKIP

    .. plot::

        import deepinv as dinv
        from torchvision.utils import make_grid
        x = dinv.utils.load_example('butterfly.png')
        patches = dinv.utils.patchify(x, patch_size=64, stride=32)
        list_patch = [patches[0, :, i, j, ...] for i in range(patches.shape[2]) for j in range(patches.shape[3])]
        dinv.utils.plot([x, make_grid(list_patch, nrow=patches.shape[2])], titles=["Original", "Overlapping patches"])

    """
    patch_size_2d, stride_2d, padding_2d = _resolve_tiling_params(
        patch_size=patch_size,
        stride=stride,
        padding=padding,
    )
    return _image_to_patches_impl(
        image=image,
        patch_size=patch_size_2d,
        stride=stride_2d,
        padding=padding_2d,
        pad_if_needed=pad_if_needed,
    )


def patches_to_images(
    patches: Tensor,
    stride: int | tuple[int, int],
    img_size: tuple[int, int] | None = None,
    reduce_overlap: str = "sum",
) -> torch.Tensor:
    r"""Reconstruct images from overlapping 2D patches.

    :param torch.Tensor patches: Patches tensor of shape
        ``(B, C, n_rows, n_cols, patch_h, patch_w)``.
    :param int | tuple[int, int] stride: Stride between adjacent patches as
        ``(stride_h, stride_w)``.
    :param tuple[int, int] img_size: Optional target output image size
        ``(height, width)``. If provided, output is cropped to this size.
    :param str reduce_overlap: How to reduce overlapping areas, ``"sum"`` or
        ``"mean"``.
    :return: Reconstructed images of shape ``(B, C, H, W)``.
    """
    _, stride_2d, _ = _resolve_tiling_params(
        patch_size=patches.shape[-2:], stride=stride, padding=0
    )
    return _patches_to_image_impl(
        patches=patches,
        stride=stride_2d,
        img_size=img_size,
        reduce_overlap=reduce_overlap,
    )


def patchify(
    image: Tensor,
    patch_size: int | tuple[int, int],
    padding: int | tuple[int, int] = 0,
    stride: int | tuple[int, int] | None = None,
    pad_if_needed: bool = True,
) -> torch.Tensor:
    r"""Backward-compatible alias for :func:`image_to_patches`."""
    return image_to_patches(
        image=image,
        patch_size=patch_size,
        padding=padding,
        stride=stride,
        pad_if_needed=pad_if_needed,
    )
