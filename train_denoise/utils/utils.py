import deepinv as dinv
from deepinv.models import ArtifactRemoval
import torch.nn

# from models.unext_v2 import UNeXt
from models.unext_wip import UNeXt

from torchvision import transforms
from models.drunet_multi_conditional import DRUNetConditional


def get_wandb_setup(
    wandb_logs_path, WANDB_PROJ_NAME, mode="offline", wandb_resume_id=None
):
    """
    Set up the wandb configuration.

    :param str wandb_logs_path: path to the wandb logs
    :param wandb_resume_id: id of the wandb run to resume
    :return: dictionary containing the wandb setup
    """

    if wandb_resume_id is not None:
        wandb_setup = {
            "dir": wandb_logs_path,
            "mode": mode,
            "project": WANDB_PROJ_NAME,
            "id": wandb_resume_id,
            "resume": "must",
        }
    else:
        wandb_setup = {"dir": wandb_logs_path, "mode": mode, "project": WANDB_PROJ_NAME}

    return wandb_setup


def rescale_img(img, rescale_mode="min_max"):
    if rescale_mode == "min_max":
        print("Max : ", img.max())
        print("Min : ", img.min())
        if img.max() != img.min():
            img = img - img.min()
            img = img / img.max()
    elif rescale_mode == "clip":
        img = img.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")
    return img


def get_transforms(train_patch_size=128, grayscale=False):
    """
    Get the transforms to be applied to the training and validation datasets.

    :param int train_patch_size: size of the training patches
    :param bool grayscale: if True, the images are converted to grayscale
    :return: train_transform, val_transform, int in_channels, int out_channels
    """
    if grayscale:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(train_patch_size, pad_if_needed=True),
                transforms.functional.rgb_to_grayscale,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        in_channels, out_channels = 1, 1
        val_transform = transforms.Compose(
            [transforms.functional.rgb_to_grayscale, transforms.ToTensor()]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(train_patch_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        in_channels, out_channels = 3, 3
        val_transform = transforms.ToTensor()

    return train_transform, val_transform, in_channels, out_channels
