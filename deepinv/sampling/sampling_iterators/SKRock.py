from deepinv.sampling.utils import projbox
import torch
import numpy as np
import time as time
from typing import Optional

from deepinv.optim import ScorePrior
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.physics import Physics

# First kind Chebyshev functions
T_s = lambda s, u: np.cosh(s * np.arccosh(u))
T_prime_s = lambda s, u: s * np.sinh(s * np.arccosh(u)) / np.sqrt(u**2 - 1)


class SKRockIterator(SamplingIterator):
    r"""
    Single iteration of the SK-ROCK (Stabilized Runge-Kutta-Chebyshev) Algorithm.

    Obtains samples of the posterior distribution using an orthogonal Runge-Kutta-Chebyshev stochastic
    approximation to accelerate the standard Unadjusted Langevin Algorithm.

    The algorithm was introduced in :footcite:t:`pereyra2020accelerating`.

    - SKROCK assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, SKROCK requires that ``step_size`` smaller than :math:`\frac{1}{L+\|A\|_2^2}`

    :param tuple(int,int) clip: Tuple of (min, max) values to clip/project the samples into a bounded range during sampling.
        Useful for images where pixel values should stay within a specific range (e.g., (0,1) or (0,255)). Default: ``None``
    :param dict algo_params: Dictionary containing the algorithm parameters (see table below)

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
    """

    def __init__(
        self, algo_params: dict[str, float], clip: Optional[tuple[float, float]] = None
    ):
        super().__init__(algo_params)
        # Check for required parameters and raise error if any are missing
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "inner_iter" not in algo_params:
            missing_params.append("inner_iter")
        if "eta" not in algo_params:
            missing_params.append("eta")
        if "sigma" not in algo_params:
            missing_params.append("sigma")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for SKRock: {', '.join(missing_params)}"
            )
        self.clip = clip

    def forward(
        self,
        X: dict[str, torch.Tensor],
        y: torch.Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: ScorePrior,
        iteration: int,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        r"""
        Performs a single SK-ROCK sampling step.

        :param Dict X: Dictionary containing the current state :math:`x_t`.
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param ScorePrior cur_prior: Prior

        :return: Dictionary `{"x": x}` containing the next state :math:`x_{t+1}` in the Markov chain.
        :rtype: Dict
        """
        x = X["x"]
        # Define posterior gradient
        posterior = lambda u: cur_data_fidelity.grad(u, y, physics) + self.algo_params[
            "alpha"
        ] * (cur_prior.grad(u, self.algo_params["sigma"]))

        # Compute SK-ROCK parameters
        w0 = 1 + self.algo_params["eta"] / (
            self.algo_params["inner_iter"] ** 2
        )  # parameter \omega_0
        w1 = T_s(self.algo_params["inner_iter"], w0) / T_prime_s(
            self.algo_params["inner_iter"], w0
        )  # parameter \omega_1
        mu1 = w1 / w0  # parameter \mu_1
        nu1 = self.algo_params["inner_iter"] * w1 / 2  # parameter \nu_1
        kappa1 = self.algo_params["inner_iter"] * (w1 / w0)  # parameter \kappa_1

        # Sample noise
        noise = torch.randn_like(x) * np.sqrt(2 * self.algo_params["step_size"])

        # First internal iteration (s=1)
        xts_2 = x.clone()
        xts = (
            x.clone()
            - mu1 * self.algo_params["step_size"] * posterior(x + nu1 * noise)
            + kappa1 * noise
        )

        # Remaining internal iterations
        for js in range(2, self.algo_params["inner_iter"] + 1):
            xts_1 = xts.clone()
            mu = 2 * w1 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \mu_js
            nu = 2 * w0 * T_s(js - 1, w0) / T_s(js, w0)  # parameter \nu_js
            kappa = 1 - nu  # parameter \kappa_js
            xts = (
                -mu * self.algo_params["step_size"] * posterior(xts)
                + nu * xts
                + kappa * xts_2
            )
            xts_2 = xts_1

        if self.clip:
            xts = projbox(xts, self.clip[0], self.clip[1])

        return {"x": xts}


# Alias for SKRockIterator
SKROCKIterator = SKRockIterator
