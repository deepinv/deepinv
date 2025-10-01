from deepinv.datasets.base import ImageDataset
from deepinv.utils.decorators import _deprecated_alias
from deepinv.utils.io_utils import load_np

import random
import os

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

class PatchDataset3D(ImageDataset):
    def __init__(self, im_dir, patch_size=64, stride=32):
        self.patch_size = patch_size
        self.stride = stride
        self.im_dir = im_dir

        self.imgs = os.listdir(im_dir) # add a check here to ensure only nifti and/or npy files are included?
        # self.shapes = [load_np(os.path.join(im_dir, im), as_memmap=True).shape for im in self.imgs]
        # to keep things as simple and close to the PatchDataset example, assume all files have same shape
        D, H, W = load_np(os.path.join(im_dir, self.imgs[0]), as_memmap=True).shape

        self.patches_per_image_d = (D - patch_size) // stride + 1
        self.patches_per_image_h = (H - patch_size) // stride + 1
        self.patches_per_image_w = (W - patch_size) // stride + 1

        self.patches_per_image = (
            self.patches_per_image_d
            * self.patches_per_image_h
            * self.patches_per_image_w
        )

    def __len__(self):
        return len(self.imgs) * self.patches_per_image

    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_image
        idx_in_vol = idx % self.patches_per_image
        fpath = os.path.join(self.im_dir, self.imgs[vol_idx])

        per_h_w = self.patches_per_image_h * self.patches_per_image_w
        id = idx_in_vol // per_h_w
        rem = idx_in_vol %  per_h_w
        ih  = rem // self.patches_per_image_w
        iw  = rem %  self.patches_per_image_w

        return load_np(fpath, start_coords=[id * self.stride, ih * self.stride, iw * self.stride], patch_size=self.patch_size).unsqueeze(0)
    

class AlternativePatchDataset3D(ImageDataset):
    def __init__(self, im_dir, patch_size=64):
        self.patch_size = patch_size
        self.im_dir = im_dir

        self.imgs = os.listdir(im_dir) # add a check here to ensure only nifti and/or npy files are included?
        self.shapes = [load_np(os.path.join(im_dir, im), as_memmap=True).shape for im in self.imgs]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        fpath = os.path.join(self.im_dir, self.imgs[idx])
        shape = self.shapes[idx]

        # We use random here: need to make a fix for deterministic behaviour based on seed --> seed worker function, see torch reproducibility page
        return load_np(fpath, start_coords=[random.randint(self.patch_size, shape[0] - self.patch_size), random.randint(self.patch_size, shape[1] - self.patch_size), random.randint(self.patch_size, shape[2] - self.patch_size)], patch_size=self.patch_size).unsqueeze(0)