import torch


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
