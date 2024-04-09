import torch

import deepinv as dinv


def spectral_methods(
    n_iter,
    x: torch.Tensor,
    y: torch.Tensor,
    physics,
    preprocessing=lambda x: torch.max(1 - 1 / x, torch.tensor(-5.0)),
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
    x = x.to(torch.cfloat)
    diag_T = preprocessing(y)
    diag_T = diag_T.to(torch.cfloat)
    for _ in range(n_iter):
        res = physics.B(x)
        res = diag_T * res
        res = physics.B_adjoint(res)
        x = res + lamb * x
        x = x / torch.linalg.norm(x)
    return x
