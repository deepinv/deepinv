import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
import math
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.rcParams.update({"font.size": 17})
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.style.use("seaborn-darkgrid")
from matplotlib.ticker import MaxNLocator

use_tex = matplotlib.checkdep_usetex(True)
if use_tex:
    plt.rcParams["text.usetex"] = True
import torch


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


def plot(
    img_list,
    titles=None,
    save_dir=None,
    tight=True,
    max_imgs=4,
    rescale_mode="min_max",
    show=True,
):
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

    :param list[torch.Tensor], torch.Tensor img_list: list of images to plot or single image.
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param str save_dir: path to save the plot.
    :param bool tight: use tight layout.
    :param int max_imgs: maximum number of images to plot.
    :param str rescale_mode: rescale mode, either 'min_max' (images are linearly rescaled between 0 and 1 using their min and max values) or 'clip' (images are clipped between 0 and 1).
    :param bool show: show the image plot.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(img_list, torch.Tensor):
        img_list = [img_list]

    if isinstance(titles, str):
        titles = [titles]

    imgs = []
    for im in img_list:
        col_imgs = []
        for i in range(min(im.shape[0], max_imgs)):
            if im.shape[1] == 2:  # for complex images
                pimg = (
                    im[i, :, :, :]
                    .pow(2)
                    .sum(dim=0)
                    .sqrt()
                    .unsqueeze(0)
                    .type(torch.float32)
                )
            else:
                pimg = im[i, :, :, :].type(torch.float32)
            if rescale_mode == "min_max":
                pimg = (pimg - pimg.min()) / (pimg.max() - pimg.min())
            elif rescale_mode == "clip":
                pimg = pimg.clamp(min=0.0, max=1.0)
            else:
                raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")

            col_imgs.append(pimg.detach().permute(1, 2, 0).squeeze().cpu().numpy())
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
                    save_dir / (titles[i] + "_" + str(r) + ".png"), img, cmap="gray"
                )
    if show:
        plt.show()


def plot_curves(metrics, save_dir=None, show=True):
    r"""
    Plots the metrics of a Plug-and-Play algorithm.

    :param dict metrics: dictionary of metrics to plot.
    :param str save_dir: path to save the plot.
    :param bool show: show the image plot.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(
        1, len(metrics.keys()), figsize=(6 * len(metrics.keys()), 4)
    )
    for i, metric_name in enumerate(metrics.keys()):
        metric_val = metrics[metric_name]
        if len(metric_val) > 0:
            batch_size, n_iter = len(metric_val), len(metric_val[0])
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["top"].set_visible(False)
            if metric_name == "residual":
                label = r"Residual $\frac{||x_{k+1} - x_k||}{||x_k||}$"
                log_scale = True
            elif metric_name == "psnr":
                label = r"$PSNR(x_k)$"
                log_scale = False
            elif metric_name == "cost":
                label = r"$F(x_k)$"
                log_scale = True
            else:
                label = metric_name
                log_scale = False
            for b in range(batch_size):
                if not log_scale:
                    axs[i].plot(metric_val[b], "-o", label=f"batch {b+1}")
                else:
                    axs[i].semilogy(metric_val[b], "-o", label=f"batch {b+1}")
            axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            # axs[i].set_xlabel("iterations")
            axs[i].set_title(label)
            axs[i].legend()
    plt.subplots_adjust(hspace=0.1)
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


def wandb_plot_curves(metrics, batch_idx=0, step=0):
    for metric_name, metric_val in zip(metrics.keys(), metrics.values()):
        if len(metric_val) > 0:
            batch_size, n_iter = len(metric_val), len(metric_val[0])
            wandb.log(
                {
                    f"{metric_name} batch {i}": wandb.plot.line_series(
                        xs=range(n_iter),
                        ys=metric_val,
                        keys=[f"image {j}" for j in range(batch_size)],
                        title=f"{metric_name} batch {batch_idx}",
                        xname="iteration",
                    )
                },
                step=step,
            )


def plot_parameters(model, init_params=None, save_dir=None, show=True):
    r"""
    Plot the parameters of the model before and after training.
    This can be used after training Unfolded optimization models.

    :param torch.nn.Module model: the model whose parameters are plotted. The parameters are contained in the dictionary
        ``params_algo`` attribute of the model.
    :param dict init_params: the initial parameters of the model, before training. Defaults to ``None``.
    :param str, Path save_dir: the directory where to save the plot. Defaults to ``None``.
    :param show bool: whether to show the plot. Defaults to ``True``.
    """

    color = ["b", "g", "r", "c", "m", "y", "k", "w"]

    fig, ax = plt.subplots(figsize=(7, 7))

    def get_param(param):
        if torch.is_tensor(param):
            if len(param.shape) > 0:
                return param[0].mean().item()
            else:
                return param.item()
        else:
            return param

    for i, name_param in enumerate(model.params_algo):
        value = [
            get_param(model.params_algo[name_param][k])
            for k in range(len(model.params_algo[name_param]))
        ]
        if init_params is not None and name_param in init_params:
            value_init = [
                get_param(init_params[name_param][k])
                for k in range(len(init_params[name_param]))
            ]
            ax.plot(value_init, "--o", label="init. " + name_param, color=color[i])
            ax.plot(value, "-o", label="learned " + name_param, color=color[i])

    # Set labels and title
    ax.set_facecolor("white")
    ax.set_xticks(np.arange(len(value), step=5))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="-", alpha=0.5, color="lightgray")
    ax.tick_params(color="lightgray")
    ax.legend()

    if show:
        plt.show()
    if save_dir:
        plt.savefig(Path(save_dir) / "parameters.png")
