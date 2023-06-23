import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import wandb
import math


def torch2cpu(img):
    if img.shape[1] == 2:  # for complex images (e.g. in MRI)
        img = img.pow(2).sum(dim=1, keepdim=True).sqrt()

    return (
        img[0, :, :, :]
        .clamp(min=0.0, max=1.0)
        .detach()
        .permute(1, 2, 0)
        .squeeze()
        .cpu()
        .numpy()
    )


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def numpy2uint(img):
    img = img.clip(0, 1)
    return np.uint8((img * 255.0).round())


def im_save(save_img_path, img):
    img = numpy2uint(img)
    plt.imsave(save_img_path, img)


def plot(img_list, titles=None, save_dir=None, tight=True, max_imgs=4):
    r"""
    Plots a list of images.

    The images should be of shape [B,C,H,W], where B is the batch size, C is the number of channels,
    H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
    and the number of columns is the length of the list. If the list is longer than max_imgs, only the first
    max_imgs are plotted.


    Example usage:

    ::

        import torch
        import deepinv.utils.plot as plot
        img = torch.rand(4, 3, 256, 256)
        plot([img, img, img], titles=["img1", "img2", "img3"], save_dir="test.png")

    :param list[torch.Tensor], torch.Tensor img_list: list of images to plot or single image.
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param str save_dir: path to save the plot.
    :param bool tight: use a tight layout.
    :param int max_imgs: maximum number of images to plot.

    """
    if save_dir:
        if not os.path.exists(save_dir.split("/")[0]):
            print("Creating ", save_dir.split("/")[0], " folder...")
            os.makedirs(save_dir.split("/")[0])

    if isinstance(img_list, torch.Tensor):
        img_list = [img_list]

    if isinstance(titles, str):
        titles = [titles]

    imgs = []
    for im in img_list:
        col_imgs = []
        for i in range(min(im.shape[0], max_imgs)):
            if im.shape[1] == 2:  # for complex images
                pimg = im[i, :, :, :].pow(2).sum(dim=0).sqrt().unsqueeze(0)
            else:
                pimg = im[i, :, :, :]

            col_imgs.append(
                pimg.clamp(min=0.0, max=1.0)
                .detach()
                .permute(1, 2, 0)
                .squeeze()
                .cpu()
                .numpy()
            )

        imgs.append(col_imgs)

    plt.figure(figsize=(len(imgs), len(imgs[0]) * 1.3))

    for i, row_imgs in enumerate(imgs):
        for r, img in enumerate(row_imgs):
            plt.subplot(len(imgs[0]), len(imgs), r * len(imgs) + i + 1)

            plt.imshow(img, cmap="gray")
            if titles and r == 0:
                plt.title(titles[i], size=8)
            plt.axis("off")

    if tight:
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

    if save_dir:
        plt.savefig(save_dir, dpi=1200)

    plt.show()


def wandb_imgs(imgs, captions, n_plot):
    wandb_imgs = []
    for i in range(len(imgs)):
        wandb_imgs.append(
            wandb.Image(
                make_grid(imgs[i][:n_plot], nrow=int(math.sqrt(n_plot)) + 1),
                caption=captions[i],
            )
        )
    return wandb_imgs
