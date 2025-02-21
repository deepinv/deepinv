import math
import shutil
from pathlib import Path
from collections.abc import Iterable
from typing import List, Tuple, Union
from itertools import zip_longest
from functools import partial
from warnings import warn

import wandb
import torch
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms as T
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


def config_matplotlib(fontsize=17):
    """Config matplotlib for nice plots in the examples."""
    plt.rcParams.update({"font.size": fontsize})
    plt.rcParams["lines.linewidth"] = 2
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


def prepare_images(x, y, x_net, x_nl=None, rescale_mode="min_max"):
    r"""
    Prepare the images for plotting.

    It prepares the images for plotting by rescaling them and concatenating them in a grid.

    :param torch.Tensor x: Ground truth.
    :param torch.Tensor y: Measurement.
    :param torch.Tensor x_net: Reconstruction network output.
    :param torch.Tensor x_nl: No-learning reconstruction.
    :returns: The images, the titles, the grid image, and the caption.
    """
    with torch.no_grad():
        imgs = [x]
        titles = ["Ground truth"]
        caption = "From left to right: Ground truth, "
        if y.shape == x.shape:
            imgs.append(y)
            titles.append("Measurement")
            caption += "Measurement, "

        if x_nl is not None:
            imgs.append(x_nl)
            titles.append("No learning")
            caption += "No learning, "

        imgs.append(x_net)
        titles.append("Reconstruction")
        caption += "Reconstruction"

        vis_array = torch.cat(imgs, dim=0)
        for i in range(len(vis_array)):
            vis_array[i] = rescale_img(vis_array[i], rescale_mode=rescale_mode)
        grid_image = make_grid(vis_array, nrow=y.shape[0])

    for k in range(len(imgs)):
        imgs[k] = preprocess_img(imgs[k], rescale_mode=rescale_mode)

    return imgs, titles, grid_image, caption


def preprocess_img(im, rescale_mode="min_max"):
    r"""
    Preprocesses an image tensor for plotting.

    :param torch.Tensor im: the image to preprocess.
    :param str rescale_mode: the rescale mode, either 'min_max' or 'clip'.
    :return: the preprocessed image.
    """
    with torch.no_grad():
        if im.shape[1] == 2:  # for complex images
            pimg = im.pow(2).sum(dim=1, keepdim=True).sqrt().type(torch.float32)
        elif im.shape[1] > 3:
            pimg = im.type(torch.float32)
        else:
            if torch.is_complex(im):
                pimg = im.abs().type(torch.float32)
            else:
                pimg = im.type(torch.float32)

        pimg = rescale_img(pimg, rescale_mode=rescale_mode)
    return pimg


def tensor2uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def numpy2uint(img):
    img = img.clip(0, 1)
    return np.uint8((img * 255.0).round())


def rescale_img(im, rescale_mode="min_max"):
    r"""
    Rescale an image tensor.

    :param torch.Tensor im: the image to rescale.
    :param str rescale_mode: the rescale mode, either 'min_max' or 'clip'.
    :return: the rescaled image.
    """
    img = im.clone()
    if rescale_mode == "min_max":
        shape = img.shape
        img = img.reshape(shape[0], -1)
        mini = img.min(1)[0]
        maxi = img.max(1)[0]
        idx = mini < maxi
        mini = mini[idx].unsqueeze(1)
        maxi = maxi[idx].unsqueeze(1)
        img[idx, :] = (img[idx, :] - mini) / (maxi - mini)
        img = img.reshape(shape)
    elif rescale_mode == "clip":
        img = img.clamp(min=0.0, max=1.0)
    else:
        raise ValueError("rescale_mode has to be either 'min_max' or 'clip'.")
    return img


def plot(
    img_list,
    titles=None,
    save_fn=None,
    save_dir=None,
    tight=True,
    max_imgs=4,
    rescale_mode="min_max",
    show=True,
    figsize=None,
    suptitle=None,
    cmap="gray",
    fontsize=17,
    interpolation="none",
    cbar=False,
    dpi=1200,
    fig=None,
    axs=None,
    return_fig=False,
    return_axs=False,
):
    r"""
    Plots a list of images.

    The images should be of shape [B,C,H,W] or [C, H, W], where B is the batch size, C is the number of channels,
    H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
    and the number of columns is the length of the list. If the B is bigger than max_imgs, only the first
    batches are plotted.

    .. warning::

        If the number of channels is 2, the magnitude of the complex images is plotted.
        If the number of channels is bigger than 3, only the first 3 channels are plotted.

    We provide flexibility to save plots either side-by-side using ``save_fn`` or as individual images using ``save_dir``.

    Example usage:

    .. doctest::

        import torch
        from deepinv.utils import plot
        img = torch.rand(4, 3, 256, 256)
        plot([img, img, img], titles=["img1", "img2", "img3"], save_dir="test.png")

    .. note::

        Using ``show=True`` calls ``plt.show()`` with blocking (outside notebook environments).
        If this is undesired simply use ``fig = plot(..., show=False, return_fig=True)``
        and plot at your desired location using ``fig.show()``.

    :param list[torch.Tensor], dict[str,torch.Tensor], torch.Tensor img_list: list of images, single image,
        or dict of titles: images to plot.
    :param list[str], str, None titles: list of titles for each image, has to be same length as img_list.
    :param None, str, pathlib.Path save_fn: path to save the plot as a single image (i.e. side-by-side).
    :param None, str, pathlib.Path save_dir: path to save the plots as individual images.
    :param bool tight: use tight layout.
    :param int max_imgs: maximum number of images to plot.
    :param str rescale_mode: rescale mode, either ``'min_max'`` (images are linearly rescaled between 0 and 1 using
        their min and max values) or ``'clip'`` (images are clipped between 0 and 1).
    :param bool show: show the image plot.
    :param tuple[int] figsize: size of the figure. If ``None``, calculated from the size of ``img_list``.
    :param str suptitle: title of the figure.
    :param str cmap: colormap to use for the images. Default: gray
    :param str interpolation: interpolation to use for the images.
        See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more details.
        Default: none
    :param int dpi: DPI to save images.
    :param None, Figure: matplotlib Figure object to plot on. If None, create new Figure. Defaults to None.
    :param None, Axes: matplotlib Axes object to plot on. If None, create new Axes. Defaults to None.
    :param bool return_fig: return the figure object.
    :param bool return_axs: return the axs object.
    """
    # Use the matplotlib config from deepinv
    config_matplotlib(fontsize=fontsize)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(img_list, torch.Tensor):
        img_list = [img_list]
    elif isinstance(img_list, dict):
        assert titles is None, "titles should be None when img_list is a dictionary"
        titles, img_list = list(img_list.keys()), list(img_list.values())

    for i, img in enumerate(img_list):
        if len(img.shape) == 3:
            img_list[i] = img.unsqueeze(0)

    if isinstance(titles, str):
        titles = [titles]

    imgs = []
    for im in img_list:
        col_imgs = []
        im = preprocess_img(im, rescale_mode=rescale_mode)
        for i in range(min(im.shape[0], max_imgs)):
            col_imgs.append(
                im[i, ...].detach().permute(1, 2, 0).squeeze().cpu().numpy()
            )
        imgs.append(col_imgs)

    if figsize is None:
        figsize = (len(imgs) * 2, len(imgs[0]) * 2)

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            len(imgs[0]),
            len(imgs),
            figsize=figsize,
            squeeze=False,
        )

    if suptitle:
        plt.suptitle(suptitle, size=12)
        fig.subplots_adjust(top=0.75)

    for i, row_imgs in enumerate(imgs):
        for r, img in enumerate(row_imgs):
            im = axs[r, i].imshow(img, cmap=cmap, interpolation=interpolation)
            if cbar:
                divider = make_axes_locatable(axs[r, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                colbar = fig.colorbar(im, cax=cax, orientation="vertical")
                colbar.ax.tick_params(labelsize=8)
            if titles and r == 0:
                axs[r, i].set_title(titles[i], size=9)
            axs[r, i].axis("off")

    if tight:
        if cbar:
            plt.subplots_adjust(hspace=0.2, wspace=0.2)
        else:
            plt.subplots_adjust(hspace=0.01, wspace=0.05)

    if save_fn:
        plt.savefig(save_fn, dpi=dpi)

    if save_dir:
        plt.savefig(save_dir / "images.svg", dpi=dpi)
        save_dir_i = Path(save_dir) / Path(titles[i])
        save_dir_i.mkdir(parents=True, exist_ok=True)
        for i, row_imgs in enumerate(imgs):
            for r, img in enumerate(row_imgs):
                plt.imsave(save_dir_i / (str(r) + ".png"), img, cmap=cmap)
    if show:
        plt.show()

    if return_fig and return_axs:
        return fig, axs
    elif return_fig:
        return fig
    elif return_axs:
        return axs


def scatter_plot(
    xy_list,
    titles=None,
    save_dir=None,
    tight=True,
    show=True,
    return_fig=False,
    figsize=None,
    suptitle=None,
    cmap="gray",
    fontsize=17,
    s=0.1,
    linewidths=1.5,
    color="b",
):
    r"""
    Plots a list of scatter plots.

    Example usage:

    .. doctest::

        import torch
        from deepinv.utils import scatter_plot
        xy = torch.randn(10, 2)
        scatter_plot([xy, xy], titles=["scatter1", "scatter2"], save_dir="test.png")

    :param list[torch.Tensor], torch.Tensor xy_list: list of scatter plots data, or single scatter plot data.
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param None, str, pathlib.Path save_dir: path to save the plot.
    :param bool tight: use tight layout.
    :param bool show: show the image plot.
    :param bool return_fig: return the figure object.
    :param tuple[int] figsize: size of the figure.
    :param str suptitle: title of the figure.
    :param str cmap: colormap to use for the images. Default: gray
    :param int fontsize: fontsize for the plot. Default: 17
    :param float s: size of the scattered points. Default: 0.1
    :param float linewidths: width of the lines. Default: 1.5
    :param str color: color of the points. Default: blue
    """
    # Use the matplotlib config from deepinv
    config_matplotlib(fontsize=fontsize)

    if isinstance(xy_list, torch.Tensor):
        xy_list = [xy_list]

    if isinstance(titles, str):
        titles = [titles]

    scatters = []
    for xy in xy_list:
        scatters.append([xy.detach().cpu().numpy()])

    if figsize is None:
        figsize = (len(scatters) * 2, len(scatters[0]) * 2)

    fig, axs = plt.subplots(
        len(scatters[0]),
        len(scatters),
        figsize=figsize,
        squeeze=False,
    )

    if suptitle:
        plt.suptitle(suptitle, size=12)
        fig.subplots_adjust(top=0.75, wspace=0.15)

    for i, row_scatter in enumerate(scatters):
        for r, xy in enumerate(row_scatter):
            axs[r, i].scatter(
                xy[:, 0], xy[:, 1], s=s, linewidths=linewidths, c=color, cmap=cmap
            )
            if titles and r == 0:
                axs[r, i].set_title(titles[i], size=9)
            axs[r, i].axis("off")
    if tight:
        plt.subplots_adjust(hspace=0.01, wspace=0.05)

    if save_dir:
        plt.savefig(save_dir / "images.png", dpi=1200)
        for i, row_scatter in enumerate(scatters):
            save_dir_i = Path(save_dir) / Path(titles[i])
            save_dir_i.mkdir(parents=True, exist_ok=True)
            for r, img in enumerate(row_scatter):
                plt.imsave(save_dir_i / Path(str(r) + ".png"), img, cmap=cmap)
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
    # Use the matplotlib config from deepinv
    config_matplotlib()

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
                label = r"$\text{PSNR}(x_k)$" if plt.rcParams["text.usetex"] else "PSNR"
                log_scale = False
            elif metric_name == "cost":
                label = r"$F(x_k)$" if plt.rcParams["text.usetex"] else "F"
                log_scale = False
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
    :param str, pathlib.Path save_dir: the directory where to save the plot. Defaults to ``None``.
    :param bool show: whether to show the plot. Defaults to ``True``.
    """

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

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


def plot_inset(
    img_list: List[torch.Tensor],
    titles: List[str] = None,
    labels: List[str] = [],
    label_loc: Union[Tuple, List] = (0.03, 0.03),
    extract_loc: Union[Tuple, List] = (0.0, 0.0),
    extract_size: float = 0.2,
    inset_loc: Union[Tuple, List] = (0.0, 0.5),
    inset_size: float = 0.4,
    figsize: Tuple[int] = None,
    save_fn: str = None,
    dpi: int = 1200,
    show: bool = True,
    return_fig: bool = False,
    cmap: str = "gray",
):
    r"""Plots a list of images with zoomed-in insets extracted from the images.

    The inset taken from extract_loc and shown at inset_loc. The coordinates extract_loc, inset_loc, and label_loc correspond to their top left corners taken at (horizontal, vertical) from the image's top left.

    Each loc can either be a tuple (float, float) which uses the same loc for all images across the batch dimension, or a list of these whose length must equal the batch dimension.

    Coordinates are fractions from 0-1, (0, 0) is the top left corner and (1, 1) is the bottom right corner.

    :param list[torch.Tensor], torch.Tensor img_list: list of images to plot or single image.
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param list[str] labels: list of overlaid labels for each image, has to be same length as img_list.
    :param list, tuple label_loc: location or locations for label to be plotted on image, defaults to (.03, .03)
    :param list, tuple extract_loc: image location or locations for extract to be taken from, defaults to (0., 0.)
    :param float extract_size: size of extract to be taken from image, defaults to 0.2
    :param list, tuple inset_loc: location or locations for inset to be plotted on image, defaults to (0., 0.5)
    :param float inset_size: size of inset to be plotted on image, defaults to 0.4
    :param tuple[int] figsize: size of the figure.
    :param str save_fn: filename for plot to be saved, if None, don't save, defaults to None
    :param int dpi: DPI to save images.
    :param bool show: show the image plot.
    :param bool return_fig: return the figure object.
    """

    fig = plot(
        img_list, titles, show=False, return_fig=True, cmap=cmap, figsize=figsize
    )
    axs = fig.axes
    batch_size = img_list[0].shape[0]

    # Expand the locs over img_list and batch dimensions
    def expand_locs(locs, n):
        if not isinstance(locs[0], (tuple, list)):
            locs = (locs,)
        n //= len(locs)
        temp = [(loc,) * n for loc in locs]
        return [a for b in temp for a in b]

    extract_locs = expand_locs(extract_loc, len(img_list) * batch_size)
    inset_locs = expand_locs(inset_loc, len(img_list) * batch_size)
    label_locs = expand_locs(label_loc, len(img_list) * batch_size)

    for img, ax, label, extract_loc, inset_loc, label_loc in zip_longest(
        [vol[[i]] for i in range(batch_size) for vol in img_list],
        axs,
        labels,
        extract_locs,
        inset_locs,
        label_locs,
    ):
        _, _, h, w = img.shape

        # Plot inset
        axins = ax.inset_axes(
            (inset_loc[0], 1 - inset_loc[1] - inset_size, inset_size, inset_size)
        )
        axins.imshow(
            rescale_img(img)
            .type(torch.float32)
            .squeeze(0)
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy(),
            cmap=cmap,
        )

        # Set inset image according to extract
        axins.set_xlim(extract_loc[0] * w, (extract_loc[0] + extract_size) * w)
        axins.set_ylim((extract_loc[1] + extract_size) * h, extract_loc[1] * h)

        # Inset borders
        for spine in ["bottom", "top", "left", "right"]:
            axins.spines[spine].set_color("lime")

        axins.grid(False)
        axins.set_xticks([])
        axins.set_yticks([])

        # Extract borders
        ax.indicate_inset(
            [
                extract_loc[0] * w,
                extract_loc[1] * h,
                extract_size * w,
                extract_size * h,
            ],
            edgecolor="red",
        )

        if label is not None:
            ax.text(
                label_loc[0],
                1 - label_loc[1],
                str(label),
                fontsize="medium",
                color="red",
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square,pad=0", fc="white", ec="none"),
            )

    if save_fn:
        plt.savefig(save_fn, dpi=dpi)

    if show:
        plt.show()

    if return_fig:
        return fig


def plot_videos(
    vid_list: Union[torch.Tensor, List[torch.Tensor]],
    titles: Union[str, List[str]] = None,
    time_dim: int = 2,
    rescale_mode: str = "min_max",
    display: bool = False,
    figsize: Tuple[int] = None,
    dpi: int = None,
    save_fn: str = None,
    return_anim: bool = False,
    anim_writer: str = None,
    anim_kwargs: dict = {},
    **plot_kwargs,
):
    r"""Plots and animates a list of image sequences.

    Plots videos as sequence of side-by-side frames, and saves animation (e.g. GIF) or displays as interactive HTML in notebook.
    This is useful for e.g. time-varying inverse problems. Individual frames are plotted with :func:`deepinv.utils.plot`

    vid_list can either be a video or a list of them. A video is defined as images of shape [B,C,H,W] augmented with a time dimension specified by ``time_dim``, e.g. of shape [B,C,T,H,W] and ``time_dim=2``. All videos must be same time-length.

    Per frame of the videos, this function calls :func:`deepinv.utils.plot`, see its params to see how the frames are plotted.

    To display an interactive HTML video in an IPython notebook, use ``display=True``. Note IPython must be installed for this.

    |sep|

    :Examples:

        Display list of image sequences live in a notebook:

        >>> from deepinv.utils import plot_videos
        >>> x = torch.rand((1, 3, 5, 8, 8)) # B,C,T,H,W image sequence
        >>> y = torch.rand((1, 3, 5, 16, 16))
        >>> plot_videos([x, y], display=True) # Display interactive view in notebook (requires IPython)
        >>> plot_videos([x, y], save_fn="vid.gif") # Save video as GIF


    :param Union[torch.Tensor, List[torch.Tensor]] vid_list: video or list of videos as defined above.
    :param Union[str, List[str]] titles: titles of images in frame, defaults to None.
    :param int time_dim: time dimension of the videos. All videos should have same length in this dimension, or length 1.
        After indexing this dimension, the resulting images should be of shape (B,C,H,W). Defaults to 2.
    :param str rescale_mode: rescaling mode for :func:`deepinv.utils.plot`, defaults to "min_max"
    :param bool display: display an interactive HTML video in an IPython notebook, defaults to False
    :param tuple[int], None figsize: size of the figure. If None, calculated from size of img list.
    :param str save_fn: if not None, save the animation to this filename.
        File extension must be provided, note ``anim_writer`` might have to be specified. Defaults to None
    :param str anim_writer: animation writer, see https://matplotlib.org/stable/users/explain/animations/animations.html#animation-writers, defaults to None
    :param bool return_anim: return matplotlib animation object, defaults to False
    :param int dpi: DPI of saved videos.
    :param dict anim_kwargs: keyword args for matplotlib FuncAnimation init
    :param plot_kwargs: kwargs to pass to :func:`deepinv.utils.plot`
    """
    if isinstance(vid_list, torch.Tensor):
        vid_list = [vid_list]

    def animate(i, fig=None, axs=None):
        return plot(
            [
                vid.select(time_dim, i if vid.shape[time_dim] > 1 else 0)
                for vid in vid_list
            ],
            titles=titles,
            show=False,
            rescale_mode=rescale_mode,
            return_fig=True,
            return_axs=True,
            fig=fig,
            axs=axs,
            figsize=figsize,
            **plot_kwargs,
        )

    fig, axs = animate(0)
    anim = FuncAnimation(
        fig,
        partial(animate, fig=fig, axs=axs),
        frames=vid_list[0].shape[time_dim],
        **anim_kwargs,
    )

    if save_fn:
        save_fn = Path(save_fn)
        anim.save(
            save_fn.with_suffix(".gif") if save_fn.suffix == "" else save_fn,
            writer=anim_writer,
            dpi=dpi,
        )

    if return_anim:
        return anim

    if display:
        try:
            from IPython.display import HTML

            return HTML(anim.to_jshtml())
        except ImportError:
            warn("IPython can't be found. Install it to use display=True. Skipping...")


def plot_ortho3D(
    img_list,
    titles=None,
    save_dir=None,
    tight=True,
    max_imgs=4,
    rescale_mode="min_max",
    show=True,
    return_fig=False,
    figsize=None,
    suptitle=None,
    cmap="gray",
    fontsize=17,
    interpolation="nearest",
):
    r"""
    Plots an orthogonal view of 3D images.

    The images should be of shape [B, C, D, H, W] or [C, D, H, W], where B is the batch size, C is the number of channels,
    D is the depth, H is the height and W is the width. The images are plotted in a grid, where the number of rows is B
    and the number of columns is the length of the list. If the B is bigger than max_imgs, only the first
    batches are plotted.

    .. warning::

        If the number of channels is 2, the magnitude of the complex images is plotted.
        If the number of channels is bigger than 3, only the first 3 channels are plotted.

    Example usage:

    .. doctest::

        import torch
        from deepinv.utils import plot_ortho3D
        img = torch.rand(2, 3, 8, 16, 16)
        plot_ortho3D(img)

    :param list[torch.Tensor], torch.Tensor img_list: list of images to plot or single image.
    :param list[str] titles: list of titles for each image, has to be same length as img_list.
    :param None, str, pathlib.Path save_dir: path to save the plot.
    :param bool tight: use tight layout.
    :param int max_imgs: maximum number of images to plot.
    :param str rescale_mode: rescale mode, either 'min_max' (images are linearly rescaled between 0 and 1 using their min and max values) or 'clip' (images are clipped between 0 and 1).
    :param bool show: show the image plot.
    :param bool return_fig: return the figure object.
    :param tuple[int] figsize: size of the figure.
    :param str suptitle: title of the figure.
    :param str cmap: colormap to use for the images. Default: gray
    :param int fontsize: fontsize for the titles. Default: 17
    :param str interpolation: interpolation to use for the images. See https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html for more details. Default: none
    """
    # Use the matplotlib config from deepinv
    config_matplotlib(fontsize=fontsize)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(img_list, torch.Tensor):
        img_list = [img_list]

    for i, img in enumerate(img_list):
        if len(img.shape) == 3:
            img_list[i] = img.unsqueeze(0)

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
            elif im.shape[1] > 3:
                pimg = im[i, 0:3, :, :, :].type(torch.float32)
            else:
                if torch.is_complex(im):
                    pimg = im[i, :, :, :, :].abs().type(torch.float32)
                else:
                    pimg = im[i, :, :, :, :].type(torch.float32)
            pimg = rescale_img(pimg, rescale_mode=rescale_mode)
            col_imgs.append(pimg.detach().permute(1, 2, 3, 0).cpu().numpy())
        imgs.append(col_imgs)

    if figsize is None:
        figsize = (3 * len(imgs), 3 * len(imgs[0]))

    split_ratios = np.zeros((len(imgs), len(imgs[0])))
    for icol in range(len(imgs)):
        for jrow in range(len(imgs[0])):
            split_ratios[icol, jrow] = np.max(
                [
                    imgs[icol][jrow].shape[0] / imgs[icol][jrow].shape[1],
                    imgs[icol][jrow].shape[0] / imgs[icol][jrow].shape[2],
                ]
            )

    fig, axs = plt.subplots(
        len(imgs[0]),
        len(imgs),
        figsize=figsize,
        squeeze=False,
    )

    if suptitle:
        plt.suptitle(suptitle)
        fig.subplots_adjust(top=0.75)

    for i, row_imgs in enumerate(imgs):
        for r, img in enumerate(row_imgs):
            img = img**0.5

            ax_XY = axs[r, i]
            ax_XY.imshow(
                img[img.shape[0] // 2] ** 0.5, cmap=cmap, interpolation=interpolation
            )
            # ax_XY.set_aspect(1.)
            divider = make_axes_locatable(ax_XY)
            ax_XZ = divider.append_axes(
                "bottom", 3 * 0.5 * split_ratios[i, r], sharex=ax_XY
            )  # pad=1.0*split_ratios[i, r], sharex=ax_XY)
            ax_XZ.imshow(
                img[:, img.shape[1] // 2, :] ** 0.5,
                cmap=cmap,
                interpolation=interpolation,
            )
            ax_ZY = divider.append_axes(
                "right", 3 * 0.5 * split_ratios[i, r], sharey=ax_XY
            )  # pad=1.0*split_ratios[i, r]
            ax_ZY.imshow(
                np.moveaxis(img[:, :, img.shape[2] // 2] ** 0.5, (0, 1, 2), (1, 0, 2)),
                cmap=cmap,
                interpolation=interpolation,
            )

            if titles and r == 0:
                axs[r, i].set_title(titles[i])
            ax_XY.axis("off")
            ax_XZ.axis("off")
            ax_ZY.axis("off")

    if tight:
        plt.subplots_adjust(hspace=0.05, wspace=0.05)
    if save_dir:
        plt.savefig(save_dir / "images.png", dpi=600)
        for i, row_imgs in enumerate(imgs):
            for r, img in enumerate(row_imgs):
                plt.imsave(
                    save_dir / (titles[i] + "_" + str(r) + ".png"), img, cmap=cmap
                )
    if show:
        plt.show()

    if return_fig:
        return fig
