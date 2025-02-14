import torch.nn as nn
import torch
import numpy as np
import time as time

from deepinv.optim import ScorePrior
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator


class SKRockIterator(SamplingIterator):
    r"""
    Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.

    Expected cur_params dict:
    :param float step_size: Step size of the algorithm
    :param float alpha: Regularization parameter
    :param int inner_iter: Number of internal SK-ROCK iterations
    :param float eta: Damping parameter
    :param float sigma: Noise level for the prior
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        y,
        physics,
        cur_data_fidelity,
        cur_prior: ScorePrior,
        cur_params,
        *args,
        **kwargs,
    ):
        # Extract parameters from cur_params
        step_size = cur_params["step_size"]
        alpha = cur_params["alpha"]
        inner_iter = cur_params["inner_iter"]
        eta = cur_params["eta"]
        sigma = cur_params["sigma"]

        # Define posterior gradient
        posterior = lambda u: -cur_data_fidelity.grad(u, y, physics) + alpha * (
            -cur_prior.grad(u, sigma)
        )

        # First kind Chebyshev functions
        T_s = lambda s, u: np.cosh(s * np.arccosh(u))
        T_prime_s = lambda s, u: s * np.sinh(s * np.arccosh(u)) / np.sqrt(u**2 - 1)

        # Compute SK-ROCK parameters
        w0 = 1 + eta / (inner_iter**2)  # parameter \omega_0
        w1 = T_s(inner_iter, w0) / T_prime_s(inner_iter, w0)  # parameter \omega_1
        mu1 = w1 / w0  # parameter \mu_1
        nu1 = inner_iter * w1 / 2  # parameter \nu_1
        kappa1 = inner_iter * (w1 / w0)  # parameter \kappa_1

        # Sample noise
        noise = torch.randn_like(x) * np.sqrt(2 * step_size)

        # First internal iteration (s=1)
        xts_2 = x.clone()
        xts = x.clone() - mu1 * step_size * posterior(x + nu1 * noise) + kappa1 * noise

        # Remaining internal iterations
        for js in range(2, inner_iter + 1):
            xts_1 = xts.clone()
            mu = 2 * w1 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \mu_js
            nu = 2 * w0 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \nu_js
            kappa = 1 - nu  # parameter \kappa_js
            xts = -mu * step_size * posterior(xts) + nu * xts + kappa * xts_2
            xts_2 = xts_1

        return xts
