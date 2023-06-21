import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
matplotlib.rcParams.update({'font.size': 17})
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.style.use('seaborn-darkgrid')
from matplotlib.ticker import MaxNLocator
use_tex = matplotlib.checkdep_usetex(True)
if use_tex:
    plt.rcParams['text.usetex'] = True


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


def plot(img_list, titles=None, save_dir=None, tight=True, max_imgs=4, clip=False, show = True):
    r"""
    Plots a list of images.

    The images should be of shape [B,C,H,W], where B is the batch size, C is the number of channels,
    H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
    and the number of columns is the length of the list. If the B is bigger than max_imgs, only the first
    batches are plotted.

    Example usage:

    ::
        import torch
        from deepinv.utils import plot
        img = torch.rand(4, 3, 256, 256)
        plot([img, img, img], titles=["img1", "img2", "img3"], save_dir="test.png")

    :param list[torch.tensor] img_list: list of images to plot
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param str save_dir: path to save the plot
    :param bool tight: whether to use tight layout
    :param int max_imgs: maximum number of images to plot
    :param bool clip: whether to clip or not the image between 0 and 1 before plotting. If not, it will be automatically linearly rescaled in 0 and 1 using its minimum and maximum values.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    imgs = []
    for im in img_list:
        col_imgs = []
        for i in range(min(im.shape[0], max_imgs)):
            if im.shape[1] == 2:  # for complex images
                pimg = im[i, :, :, :].pow(2).sum(dim=0).sqrt().unsqueeze(0)
            else:
                pimg = im[i, :, :, :]
            if clip :
                pimg = pimg.clamp(min=0.0, max=1.0)

            col_imgs.append(
                pimg
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
        plt.savefig(save_dir / "images.png", dpi=1200)
        for i, row_imgs in enumerate(imgs):
            for r, img in enumerate(row_imgs):
                plt.imsave(
                    save_dir / (titles[i] + "_" + str(r) + ".png"),
                    img,
                    cmap="gray"
                )
    if show:
        plt.show()


def plot_curves(metrics, save_dir=None, show = True):
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, len(metrics.keys()), figsize=(6*len(metrics.keys()),5))
    for i, metric_name in enumerate(metrics.keys()):
        metric_val = metrics[metric_name]
        if len(metric_val) > 0:
            batch_size, n_iter = len(metric_val), len(metric_val[0])
            axs[i].spines['right'].set_visible(False)
            axs[i].spines['top'].set_visible(False)
            for b in range(batch_size):
                axs[i].plot(metric_val[b], 'o', label = f"batch {i+1}")
            axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[i].set_xlabel("iterations")
            if metric_name == 'residual' :
                label = r'Residual $\frac{||x_{k+1} - x_k||}{||x_k||}$'
            elif metric_name == 'psnr' :
                label = r'$PSNR(x_k)$'
            elif metric_name == 'cost' :
                label = r'$F(x_k)$'
            else :
                label = metric_name
            axs[i].set_ylabel(label)
    if save_dir:
        plt.savefig(save_dir / "curves.png")
    if show:
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


if __name__ == "__main__":
    import torch
    from deepinv.utils import plot
    img = torch.rand(4, 3, 256, 256)
    plot([img, img, img], titles=["img1", "img2", "img3"], max_imgs=2)