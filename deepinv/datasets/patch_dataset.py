from deepinv.datasets.base import ImageDataset
from deepinv.utils.decorators import _deprecated_alias


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
