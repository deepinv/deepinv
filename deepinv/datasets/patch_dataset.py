from torch.utils import data


class PatchDataset(data.Dataset):
    r"""
    Builds the dataset of all patches from a tensor of images.

    :param torch.Tensor imgs: Tensor of images, size: batch size x channels x height x width
    :param int patch_size: size of patches
    :param callable: data augmentation. callable object, None for no augmentation.
    """

    def __init__(self, imgs, patch_size=6, transforms=None):
        self.imgs = imgs
        self.patch_size = patch_size
        self.patches_per_image = (self.imgs.shape[2] - patch_size + 1) * (
            self.imgs.shape[3] - patch_size + 1
        )
        self.transforms = transforms

    def __len__(self):
        return self.imgs.shape[0] * self.patches_per_image

    def __getitem__(self, idx):
        idx_img = idx // self.patches_per_image
        idx_in_img = idx % self.patches_per_image
        idx_x = idx_in_img // (self.imgs.shape[3] - self.patch_size + 1)
        idx_y = idx_in_img % (self.imgs.shape[3] - self.patch_size + 1)
        patch = self.imgs[
            idx_img, :, idx_x : idx_x + self.patch_size, idx_y : idx_y + self.patch_size
        ]
        if self.transforms and False:
            patch = self.transforms(patch)
        return patch.reshape(-1), idx
