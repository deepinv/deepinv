import torch.nn as nn
import torch
import numpy as np
import time as time

from deepinv.optim import ScorePrior
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator


class SKRockIterator(SamplingIterator):
    r"""
    Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.

    Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
    approximation to accelerate the standard Unadjusted Langevin Algorithm.

    The algorithm was introduced in "Accelerating proximal Markov chain Monte Carlo by using an explicit stabilised method"
    by L. Vargas, M. Pereyra and K. Zygalakis (https://arxiv.org/abs/1908.08845)

    - SKROCK assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, SKROCK requires that ``step_size`` smaller than :math:`\frac{1}{L+\|A\|_2^2}`
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
        r"""
        Performs a single SK-ROCK sampling step.

        :param torch.Tensor x: Current state :math:`X_t` of the Markov chain
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator 
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param ScorePrior cur_prior: Prior
        :param dict cur_params: Dictionary containing the algorithm parameters (see table below)

        .. list-table::
           :widths: 15 10 75
           :header-rows: 1

           * - Parameter
             - Type
             - Description
           * - step_size
             - float
             - Step size of the algorithm (default: 1.0). Tip: use physics.lipschitz to compute the Lipschitz constant
           * - alpha
             - float
             - Regularization parameter :math:`\alpha` (default: 1.0)
           * - inner_iter
             - int
             - Number of internal iterations (default: 10)
           * - eta
             - float
             - Damping parameter :math:`\eta` (default: 0.05)
           * - sigma
             - float
             - Noise level for the score prior denoiser (default: 0.05). A larger value of sigma will result in a more regularized reconstruction
        :return: Next state :math:`X_{t+1}` in the Markov chain
        :rtype: torch.Tensor
        """
        # Extract parameters from cur_params with defaults
        step_size = cur_params.get("step_size", 1.0)
        alpha = cur_params.get("alpha", 1.0)  # using common default for consistency
        inner_iter = cur_params.get("inner_iter", 10)
        eta = cur_params.get("eta", 0.05)
        sigma = cur_params.get("sigma", 0.05)

        # Define posterior gradient
        posterior = lambda u: cur_data_fidelity.grad(u, y, physics) + alpha * (
            cur_prior.grad(u, sigma)
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
