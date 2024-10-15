import os

from dotmap import DotMap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import yaml

from deepinv.utils.demo import load_url_image, get_image_url

# Load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return DotMap(config_dict)

def generate_signal(img_size, mode, config, dtype, device):
    if mode == "shepp-logan":
        url = get_image_url("SheppLogan.png")
        img = load_url_image(
        url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
        )
    elif mode == "random":
        # random phase signal
        img = torch.rand((1, 1, img_size, img_size), device=device)
    elif mode == "mix":
        url = get_image_url("SheppLogan.png")
        img = load_url_image(
        url=url, img_size=img_size, grayscale=True, resize_mode="resize", device=device
        )
        img = img * (1-config.noise_ratio) + torch.rand_like(img) * config.noise_ratio
    elif mode == "delta":
        img = torch.zeros((1, 1, img_size, img_size), device=device)
        img[0, 0, img_size // 2, img_size // 2] = 1.0
    elif mode == "constant":
        img == 0.3 * torch.ones((1, 1, img_size, img_size), device=device)
    else:
        raise ValueError("Invalid image mode.")
    if config.reverse is True:
        img = 1 - img
    # generate phase signal
    # The phase is computed as 2*pi*x - pi, where x is the original image.
    x = torch.exp(1j * img * torch.pi - 0.5j * torch.pi).to(device)
    # Every element of the signal should have unit norm.
    assert torch.allclose(x.real**2 + x.imag**2, torch.tensor(1.0))
    if config.varying_norm is True:
        scale = config.max_scale*torch.rand_like(x, dtype=torch.float)
        x = x * scale
    return x

def compare(a:int,b:int):
    if a > b:
        return ">"
    elif a < b:
        return "<"
    else:
        return "="

def merge_order(a:str,b:str):
    if a == ">" and b == "<":
        return "!"
    elif a == "<" and b == ">":
        return "!"
    elif a == ">" or b == ">":
        return ">"
    elif a == "<" or b == "<":
        return "<"
    else:
        return "="

def default_preprocessing(y, physics):
    return torch.max(1 - 1 / y, torch.tensor(-5.0))


def correct_global_phase(
    x_recon: torch.Tensor, x: torch.Tensor, threshold=1e-5
) -> torch.Tensor:
    r"""
    Corrects the global phase of the reconstructed image.

    Do not mix the order of the reconstructed and original images since this function modifies x_recon in place.

    :param torch.Tensor x_recon: Reconstructed image.
    :param torch.Tensor x: Original image.

    :return: The corrected image.
    """
    assert x_recon.shape == x.shape, "The shapes of the images should be the same."
    assert (
        len(x_recon.shape) == 4
    ), "The images should be input with shape (N, C, H, W) "

    n_imgs = x_recon.shape[0]
    n_channels = x_recon.shape[1]

    for i in range(n_imgs):
        for j in range(n_channels):
            e_minus_phi = (x_recon[i, j].conj() * x[i, j]) / (x[i, j].abs() ** 2)
            if e_minus_phi.var() < threshold:
                print(f"Image {i}, channel {j} has a constant global phase shift.")
            else:
                print(f"Image {i}, channel {j} does not have a global phase shift.")
            e_minus_phi = e_minus_phi.mean()
            x_recon[i, j] = x_recon[i, j] * e_minus_phi

    return x_recon


def cosine_similarity(a: torch.Tensor, b: torch.Tensor):
    r"""
    Compute the cosine similarity between two images.

    The cosine similarity is computed as:

    .. math::
        \text{cosine\_similarity} = \frac{a \cdot b}{\|a\| \cdot \|b\|}.

    The value range is [0,1], higher values indicate higher similarity.
    If one image is a scaled version of the other, i.e., :math:`a = c * b` where :math:`c` is a nonzero complex number, then the cosine similarity will be 1.

    :param torch.Tensor a: First image.
    :param torch.Tensor b: Second image.
    :return: The cosine similarity between the two images."""
    assert a.shape == b.shape
    a = a.flatten()
    b = b.flatten()
    norm_a = torch.sqrt(torch.dot(a.conj(), a).real)
    norm_b = torch.sqrt(torch.dot(b.conj(), b).real)
    return torch.abs(torch.dot(a.conj(), b)) / (norm_a * norm_b)


def spectral_methods(
    y: torch.Tensor,
    physics,
    x=None,
    n_iter=50,
    preprocessing=default_preprocessing,
    lamb=10.0,
    x_true=None,
    log: bool = False,
    log_metric=cosine_similarity,
    early_stop: bool = True,
    rtol: float = 1e-5,
):
    r"""
    Utility function for spectral methods.

    :param torch.Tensor y: Measurements.
    :param deepinv.physics physics: Instance of the physics modeling the forward matrix.
    :param torch.Tensor x: Initial guess for the signals :math:`x_0`.
    :param int n_iter: Number of iterations.
    :param function preprocessing: Function to preprocess the measurements. Default is :math:`\max(1 - 1/x, -5)`.
    :param float lamb: Regularization parameter. Default is 10.

    :return: The estimated signals :math:`x`.
    """
    if x is None:
        # always use randn for initial guess, never use rand!
        x = torch.randn(
            (y.shape[0],) + physics.input_shape,
            dtype=physics.dtype,
            device=physics.device,
        )

    if log is True:
        metrics = []
    
    #! estimate the norm of x using y
    #! for the i.i.d. case, we have norm(x) = sqrt(sum(y)/A_squared_mean)
    #! for the structured case, when the mean of the squared diagonal elements is 1, we have norm(x) = sqrt(sum(y)), otherwise y gets scaled by the mean to the power of number of layers
    norm_x = torch.sqrt(y.sum())

    x = x.to(torch.complex64)
    # y should have mean 1
    y = y / torch.mean(y)
    diag_T = preprocessing(y, physics)
    diag_T = diag_T.to(torch.complex64)
    for i in range(n_iter):
        x_new = physics.B(x)
        x_new = diag_T * x_new
        x_new = physics.B_adjoint(x_new)
        x_new = x_new + lamb * x
        x_new = x_new / torch.linalg.norm(x_new)
        if log:
            metrics.append(log_metric(x_new, x_true))
        if early_stop:
            if torch.linalg.norm(x_new - x) / torch.linalg.norm(x) < rtol:
                print(f"Power iteration early stopped at iteration {i}.")
                break
        x = x_new
    #! change the norm of x so that it matches the norm of true x
    x = x * norm_x
    if log:
        return x, metrics
    else:
        return x


def spectral_methods_wrapper(y, physics, n_iter=5000, **kwargs):
    x = spectral_methods(y, physics, n_iter=n_iter, **kwargs)
    z = x.detach().clone()
    return {"est": (x, z)}


def plot_error_bars(
    oversamplings,
    datasets,
    labels,
    xlim=None,
    xticks=None,
    ylim=None,
    yticks=None,
    axis=1,
    title:str=None,
    xlabel="Oversampling Ratio",
    ylabel="Cosine Similarity",
    xscale="linear",
    yscale="linear",
    save_dir:str=None,
    figsize=(10, 6),
    marker=".",
    markersize=10,
    capsize=5,
    font="Times New Roman",
    fontsize=14,
    labelsize=16,
    ticksize=16,
    error_bar='quantile',
    quantiles=[0.10,0.50,0.90],
    error_bar_linestyle='--',
    structured_color='red',
    iid_color='blue',
    plot='other',
    legend_loc='upper left',
    transparent=True,
    show=True,
    bbox_inches = 'tight',
):

    # Generate a color palette
    palette = sns.color_palette(n_colors=len(datasets))

    plt.rcParams['font.family'] = font
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.labelsize'] = labelsize
    plt.figure(figsize=figsize)

    for i, (oversampling, data, label) in enumerate(
        zip(oversamplings, datasets, labels)
    ):
        print(label)

        if plot == 'reconstruction':
            if 'structured' in label:
                color = structured_color
            elif 'iid' in label:
                color = iid_color
        elif plot == 'layer':
            if '1 layer' in label:
                color = palette[0]
            elif '1.5 layers' in label:
                color = palette[1]
            elif '2 layers' in label:
                color = palette[2]
            elif '3 layers' in label:
                color = palette[3]
            elif 'haar' in label:
                color = palette[4]
        elif plot == 'time':
            color = palette[i]
        else:
            color = palette[i]
        
        if 'gd rand' in label:
            linestyle = ':'
        elif 'gd spec' in label:
            linestyle = '-'
        elif 'spec' in label:
            linestyle = '--'
        else:
            linestyle = '-'

        print(color,label)
        # Calculate statistics
        if type(data) == torch.Tensor:
            std_vals = data.std(dim=1).numpy()
            avg_vals = data.mean(dim=1).numpy()
            min_vals = avg_vals - std_vals
            max_vals = avg_vals + std_vals
        elif type(data) == pd.DataFrame:
            if plot == 'reconstruction' or plot == 'layer':
                for column in data.columns:
                    if "repeat" not in column:
                        data.drop(columns=column, inplace=True)
            if error_bar == 'quantile':
                min_vals = data.quantile(quantiles[0], axis=axis).values
                avg_vals = data.quantile(quantiles[1], axis=axis).values
                max_vals = data.quantile(quantiles[2], axis=axis).values
            elif error_bar == 'std':
                avg_vals = data.mean(axis=axis).values
                std_vals = data.std(axis=axis).values
                min_vals = avg_vals - std_vals
                max_vals = avg_vals + std_vals

        # Calculate error bars
        yerr_lower = avg_vals - min_vals
        yerr_upper = max_vals - avg_vals

        # Prepare data for plotting
        df = pd.DataFrame(
            {
                "x": oversampling,
                "mid": avg_vals,
                "yerr_lower": yerr_lower,
                "yerr_upper": yerr_upper,
            }
        )

        # Plotting
        ax = sns.lineplot(data=df, x="x", y="mid", marker=marker, label=label, color=color, markersize=markersize, linestyle=linestyle, zorder=2)
        if plot != 'time':
            # Adding error bars
            eb = ax.errorbar(
                df["x"],
                df["mid"],
                yerr=[df["yerr_lower"], df["yerr_upper"]],
                fmt=marker,
                capsize=capsize,
                color=color,
                zorder=2,
            )
            eb[-1][0].set_linestyle(error_bar_linestyle)
    
    if plot == 'reconstruction':
        legend_contents = [
            (Patch(visible=False), r'$\bf{Model}$'),
            (plt.Line2D([], [], linestyle='-', color=structured_color), 'structured random'),
            (plt.Line2D([], [], linestyle='-', color=iid_color), 'i.i.d. random'),
            #(Patch(visible=False), ''),  # spacer
            (Patch(visible=False), r'$\bf{Algorithm}$'),
            (plt.Line2D([], [], linestyle='-', marker='.',color='black'), 'GD + SM'),
            (plt.Line2D([], [], linestyle='--', marker='.',color='black'), 'SM'),
            (plt.Line2D([], [], linestyle=':', marker='.',color='black'), 'GD'),
        ]
        legend = ax.legend(*zip(*legend_contents),loc=legend_loc)
    elif plot == 'layer':
        legend_contents = [
            (Patch(visible=False), '$\\bf{Structure}$'),
            (plt.Line2D([], [], linestyle='-', color=palette[0]), 'FD'),
            (plt.Line2D([], [], linestyle='-', color=palette[1]), 'FDF'),
            (plt.Line2D([], [], linestyle='-', color=palette[2]), 'FDFD'),
            (plt.Line2D([], [], linestyle='-', color=palette[3]), 'FDFDFD'),
            (plt.Line2D([], [], linestyle='-', color=palette[4]), 'Random Unitary'),
            #(Patch(visible=False), ''),  # spacer
            (Patch(visible=False), '$\\bf{Algorithm}$'),
            (plt.Line2D([], [], linestyle='-', marker='.',color='black'), 'GD + SM'),
            (plt.Line2D([], [], linestyle='--', marker='.',color='black'), 'SM'),
        ]
        legend = ax.legend(*zip(*legend_contents),loc=legend_loc)
    elif plot == 'time':
        legend = ax.legend(loc=legend_loc)
    else:
        legend = ax.legend(loc=legend_loc)
    # set legend on the bottom layer
    legend.set_zorder(1)


    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    if xlim:
        ax.set_xlim(xlim,auto=True)
    if xticks:
        ax.set_xticks(xticks)
    if ylim:
        ax.set_ylim(ylim,auto=True)
    if yticks:
        ax.set_yticks(yticks)
    if title:
        ax.set_title(title)

    # Set the tick size
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.tick_params(axis='both', which='minor', labelsize=ticksize)


    if save_dir is not None:
        plt.savefig(save_dir,transparent=transparent,bbox_inches=bbox_inches)
        print(f"Figure saved to {save_dir}")

    # Show plot
    if show:
        plt.show()
