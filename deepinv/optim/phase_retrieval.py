import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


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
    early_stop: bool = False,
    rtol: float = 1e-8,
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
            (y.shape[0],) + physics.img_shape,
            dtype=physics.dtype,
            device=physics.device,
        )

    if log == True:
        metrics = []

    x = x.to(torch.cfloat)
    x = x / torch.linalg.norm(x)
    # y should have mean 1
    y = y / torch.mean(y)
    diag_T = preprocessing(y, physics)
    diag_T = diag_T.to(torch.cfloat)
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
                print(f"Power iteration early stopping at iteration {i}.")
                print(x_new - x)
                print(x)
                break
        x = x_new
    x = x * torch.sqrt(y.sum())
    if log:
        return x, metrics
    else:
        return x


def spectral_methods_wrapper(y, physics, n_iter=5000, **kwargs):
    x = spectral_methods(y, physics, n_iter=n_iter, **kwargs)
    z = spectral_methods(y, physics, n_iter=n_iter, **kwargs)
    return {"est": (x, z)}


def plot_error_bars(oversamplings,
                    datasets,
                    labels,
                    axis=1,
                    title="Performance",
                    xlabel="Oversampling Ratio",
                    ylabel="Consine Similarity",
                    xscale="linear",
                    yscale="linear",
                    ):

    # Generate a color palette
    palette = sns.color_palette(n_colors=len(datasets))

    plt.figure(figsize=(10, 6))

    for i, (oversampling, data, label) in enumerate(
        zip(oversamplings, datasets, labels)
    ):
        print(label)
        # Calculate statistics
        if type(data) == torch.Tensor:
            min_vals = data.min(dim=1).values.numpy()
            max_vals = data.max(dim=1).values.numpy()
            avg_vals = data.mean(dim=1).numpy()
        elif type(data) == pd.DataFrame:
            min_vals = data.min(axis=axis).values
            max_vals = data.max(axis=axis).values
            avg_vals = data.mean(axis=axis).values

        # Calculate error bars
        yerr_lower = avg_vals - min_vals
        yerr_upper = max_vals - avg_vals

        # Prepare data for plotting
        df = pd.DataFrame(
            {
                "x": oversampling,
                "avg": avg_vals,
                "yerr_lower": yerr_lower,
                "yerr_upper": yerr_upper,
            }
        )

        # Plotting
        color = palette[i]
        ax = sns.lineplot(data=df, x="x", y="avg", marker="o", label=label, color=color)
        # Adding error bars
        ax.errorbar(
            df["x"],
            df["avg"],
            yerr=[df["yerr_lower"], df["yerr_upper"]],
            fmt="o",
            capsize=5,
            color=color,
        )

    # Adding labels and title
    ax.set_xlabel(xlabel)
    ax.set_xscale(xscale)
    ax.set_ylabel(ylabel)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.legend()

    # Show plot
    plt.show()
