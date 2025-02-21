from torch.utils import data


class PatchDataset(data.Dataset):
    r"""
    Builds the dataset of all patches from a tensor of images.

    :param torch.Tensor imgs: Tensor of images, size: batch size x channels x height x width
    :param int patch_size: size of patches
    :param Callable transforms: data augmentation. callable object, None for no augmentation.
    :param tuple shape: shape of the returned tensor. None returns C x patch_size x patch_size.
            The default shape is (-1,).
    """

    def __init__(self, imgs, patch_size=6, stride=1, transforms=None, shapes=(-1,)):
        self.imgs = imgs
        self.patch_size = patch_size
        self.stride = stride
        self.patches_per_image_x = (self.imgs.shape[2] - patch_size) // stride + 1
        self.patches_per_image_y = (self.imgs.shape[3] - patch_size) // stride + 1
        self.patches_per_image = self.patches_per_image_x * self.patches_per_image_y
        self.transforms = transforms
        self.shapes = shapes

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
        if self.transforms:
            patch = self.transforms(patch)
        return patch.reshape(self.shapes) if self.shapes else patch, idx
