from deepinv.datasets.base import ImageDataset
from deepinv.utils.decorators import _deprecated_alias

import random
import os

import torch


class PatchDataset(ImageDataset):
    r"""
    Builds the dataset of all patches from a tensor of images.

    :param torch.Tensor imgs: Tensor of images, size: batch size x channels x height x width
    :param int patch_size: size of patches
    :param Callable transform: data augmentation. callable object, None for no augmentation.
    :param tuple shape: shape of the returned tensor. None returns C x patch_size x patch_size.
            The default shape is (-1,).
    """

    @_deprecated_alias(transforms="transform")
    @_deprecated_alias(shapes="shape")
    def __init__(self, imgs, patch_size=6, stride=1, transform=None, shape=(-1,)):
        self.imgs = imgs
        self.patch_size = patch_size
        self.stride = stride
        self.patches_per_image_x = (self.imgs.shape[2] - patch_size) // stride + 1
        self.patches_per_image_y = (self.imgs.shape[3] - patch_size) // stride + 1
        self.patches_per_image = self.patches_per_image_x * self.patches_per_image_y
        self.transform = transform
        self.shape = shape

    def __len__(self):
        return self.imgs.shape[0] * self.patches_per_image

    def __getitem__(self, idx):
        idx_img = idx // self.patches_per_image
        idx_in_img = idx % self.patches_per_image

        idx_x = (idx_in_img // self.patches_per_image_y) * self.stride
        idx_y = (idx_in_img % self.patches_per_image_y) * self.stride

        patch = self.imgs[
            idx_img, :, idx_x : idx_x + self.patch_size, idx_y : idx_y + self.patch_size
        ]

        if self.transform:
            patch = self.transform(patch)

        return patch.reshape(self.shape) if self.shape else patch


def get_loader(format: str):
    if format.endswith(".npy"):
        from deepinv.utils.io_utils import load_np

        return load_np
    elif format.endswith(".nii") or format.endswith(".nii.gz"):
        from deepinv.utils.io_utils import load_nifti

        return load_nifti
    elif format.endswith(".b2nd"):
        from deepinv.utils.io_utils import load_blosc2

        return load_blosc2
    else:
        raise NotImplementedError(
            "No loader function for 3D volumes with extension {format}"
        )


class PatchDataset3D(ImageDataset):
    def __init__(
        self,
        x_dir: str = None,
        y_dir: str = None,
        patch_size: int = 64,
        format: str = ".npy",
    ):  # allow patch_size to be list/tuple, not just int
        r"""
        Builds a dataset from folders of 3D images. Each epoch, a single patch is randomly sampled from each volume.

        Each image can have a different shape, but all images must have shape H, W, D. Other axis are not allowed (or will be squeezed).

        :param str x_dir: Path to folder of ground-truth images.
        :param str y_dir: Path to folder of measurements. Measurements must be images of same shape as ground-truth.
        :param int patch_size: Patch size to use, must be <= smallest shape in the dataset
        :param str format: Format to use. Other files will be ignored. Supported: .npy, .nii(.gz), .b2nd (blosc2)
        """
        assert x_dir or y_dir, "Both x_dir and y_dir cannot be None."

        self.x_dir, self.y_dir = x_dir, y_dir

        self.patch_size = patch_size
        self.load = get_loader(format)

        x_imgs, y_imgs = None, None

        if x_dir is not None:
            assert os.path.exists(x_dir), f"Measurement dir {x_dir} does not exist."
            x_imgs = [f for f in os.listdir(x_dir) if f.endswith(format)]
            assert (
                len(x_imgs) != 0
            ), f"Measurement dir is given but empty for file format {format}."

        if y_dir is not None:
            assert os.path.exists(y_dir), f"Ground-truth dir {y_dir} does not exist."
            y_imgs = [f for f in os.listdir(y_dir) if f.endswith(format)]
            assert (
                len(y_imgs) != 0
            ), f"Ground-truth dir is given but empty for file format {format}."

        self.imgs = (
            [f for f in x_imgs if f in y_imgs]
            if (x_imgs and y_imgs)
            else (x_imgs or y_imgs)
        )

        self.shapes = [
            self.load(os.path.join(y_dir if y_dir else x_dir, im), as_memmap=True).shape for im in self.imgs
        ]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        shape = self.shapes[idx]
        # We use random here: need to ensure deterministic behaviour based on seed --> seed worker function, see torch reproducibility page
        start_coords = [
            random.randint(self.patch_size, shape[0] - self.patch_size),
            random.randint(self.patch_size, shape[1] - self.patch_size),
            random.randint(self.patch_size, shape[2] - self.patch_size),
        ]

        fname = self.imgs[idx]

        x = (
            self.load(
                os.path.join(self.x_dir, fname),
                start_coords=start_coords,
                patch_size=self.patch_size,
            ).unsqueeze(0)
            if self.x_dir
            else torch.nan
        )
        if self.y_dir is not None:
            y = self.load(
                os.path.join(self.y_dir, fname),
                start_coords=start_coords,
                patch_size=self.patch_size,
            ).unsqueeze(0)
            return (x, y)
        else:
            return x
