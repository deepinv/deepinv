import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
import math
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Iterable
import matplotlib
import shutil
import torchvision.transforms as T
import torchvision.transforms.functional as F

matplotlib.rcParams.update({"font.size": 17})
matplotlib.rcParams["lines.linewidth"] = 2
from matplotlib.ticker import MaxNLocator

plt.rcParams["text.usetex"] = True if shutil.which("latex") else False


def resize_pad_square_tensor(tensor, size):
    r"""
    Resize a tensor BxCxWxH to a square tensor BxCxsizexsize with the same aspect ratio thanks to zero-padding.

    :param torch.Tensor tensor: the tensor to resize.
    :param int size: the new size.
    :return torch.Tensor: the resized tensor.
    """

    class SquarePad:
        def __call__(self, image):
            W, H = image.size
            print(W, H)
            max_wh = np.max([W, H])
            hp = int((max_wh - W) / 2)
            vp = int((max_wh - H) / 2)
            padding = (hp, vp, hp, vp)
            return F.pad(image, padding, fill=0, padding_mode="constant")

    transform = T.Compose([T.ToPILImage(), SquarePad(), T.Resize(size), T.ToTensor()])
    return torch.stack([transform(el) for el in tensor])


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


def rescale_img(img, rescale_mode="min_max"):
    if rescale_mode == "min_max":
        img = (img - img.min()) / (img.max() - img.min())
    elif rescale_mode == "clip":
        img = img.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")
    return img


def plot(
    img_list,
    titles=None,
    save_dir=None,
    tight=True,
    max_imgs=4,
    rescale_mode="min_max",
    show=True,
    return_fig=False,
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
    :param bool return_fig: return the figure object.
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
            pimg = rescale_img(pimg, rescale_mode=rescale_mode)
            col_imgs.append(pimg.detach().permute(1, 2, 0).squeeze().cpu().numpy())
        imgs.append(col_imgs)

    fig, axs = plt.subplots(
        len(imgs[0]),
        len(imgs),
        figsize=(len(imgs) * 2, len(imgs[0]) * 2),
        squeeze=False,
    )

    # plt.figure(figsize=(len(imgs) * 2, len(imgs[0]) * 2))
    for i, row_imgs in enumerate(imgs):
        for r, img in enumerate(row_imgs):
            axs[r, i].imshow(img, cmap="gray")
            if titles and r == 0:
                axs[r, i].set_title(titles[i], size=9)
            axs[r, i].axis("off")
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

    if return_fig:
        return fig


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
                label = (
                    r"Residual $\frac{||x_{k+1} - x_k||}{||x_k||}$"
                    if plt.rcParams["text.usetex"]
                    else "residual"
                )
                log_scale = True
            elif metric_name == "psnr":
                label = r"$PSNR(x_k)$" if plt.rcParams["text.usetex"] else "PSNR"
                log_scale = False
            elif metric_name == "cost":
                label = r"$F(x_k)$" if plt.rcParams["text.usetex"] else "F"
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
                    f"{metric_name} batch {batch_idx}": wandb.plot.line_series(
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

    for key, value in zip(init_params.keys(), init_params.values()):
        if not isinstance(value, Iterable):
            init_params[key] = [value]

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
