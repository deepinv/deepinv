from deepinv.datasets.base import ImageDataset
from deepinv.utils.decorators import _deprecated_alias

from deepinv.utils.mixins import TiledMixin2d
from typing import Callable


class PatchDataset(TiledMixin2d, ImageDataset):
    r"""
    Builds the dataset of all patches from a tensor of images.

    :param torch.Tensor imgs: Tensor of images, size: batch size x channels x height x width
    :param int patch_size: size of patches
    :param int stride: stride between patches
    :param Callable transform: data augmentation. callable object, None for no augmentation.
    :param tuple shape: shape of the returned tensor. None returns C x patch_size x patch_size.
            The default shape is (-1,).
    """

    @_deprecated_alias(transforms="transform")
    @_deprecated_alias(shapes="shape")
    def __init__(
        self,
        imgs,
        patch_size: int | tuple[int, int] = 6,
        stride: int | tuple[int, int] = 1,
        transform: Callable = None,
        shape: tuple[int, ...] = (-1,),
    ):
        super().__init__(patch_size=patch_size, stride=stride, pad_if_needed=True)
        self.transform = transform
        self.shape = shape
        all_patches = self.image_to_patches(imgs)
        from einops import rearrange
        # Reshape to (B * num_pch, C, patch_size, patch_size)
        self.all_patches = rearrange(all_patches, "B C n_rows n_cols pH pW -> (B n_rows n_cols) C pH pW")

    def __len__(self):
        return self.all_patches.shape[0]

    def __getitem__(self, idx):
        patch = self.all_patches[idx]
        if self.transform:
            patch = self.transform(patch)
        return patch.reshape(self.shape) if self.shape else patch
