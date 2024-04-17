import torch

import deepinv as dinv


def spectral_methods(
    y: torch.Tensor,
    physics,
    x=None,
    n_iter=50,
    preprocessing=lambda y: torch.max(1 - 1 / y, torch.tensor(-5.0)),
    lamb=10.0,
):
    r"""
    Utility function for spectral methods.

    :param int n_iter: Number of iterations.
    :param torch.Tensor x: Initial guess for the signals :math:`x_0`.
    :param torch.Tensor y: Measurements.
    :param deepinv.physics physics: Instance of the physics modeling the forward matrix.
    :param function preprocessing: Function to preprocess the measurements. Default is :math:`\max(1 - 1/x, -5)`.
    :param float lamb: Regularization parameter. Default is 10.

    :return: The estimated signals :math:`x`.
    """
    if x is None:
        x = torch.rand((y.shape[0],) + physics.img_shape, dtype=physics.dtype)
    x = x.to(torch.cfloat)
    x = x / torch.linalg.norm(x)

    diag_T = preprocessing(y)
    diag_T = diag_T.to(torch.cfloat)
    for _ in range(n_iter):
        res = physics.B(x)
        res = diag_T * res
        res = physics.B_adjoint(res)
        x = res + lamb * x
        x = x / torch.linalg.norm(x)
    return x


def correct_global_phase(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""
    Corrects the global phase of the reconstructed image.

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
            if e_minus_phi.var() < 1e-3:
                print(f"Image {i}, channel {j} has a constant global phase shift.")
            else:
                print(f"Image {i}, channel {j} does not have a global phase shift.")
            e_minus_phi = e_minus_phi.mean()
            x_recon[i, j] = x_recon[i, j] * e_minus_phi

    return x_recon
