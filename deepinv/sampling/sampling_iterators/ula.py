from deepinv.sampling.utils import projbox
import torch
from torch import Tensor
import time as time
import numpy as np
from deepinv.physics import Physics
from deepinv.optim.prior import ScorePrior
from deepinv.sampling.sampling_iterators.sampling_iterator import SamplingIterator
from deepinv.optim.data_fidelity import DataFidelity


class ULAIterator(SamplingIterator):
    r"""
    Projected Plug-and-Play Unadjusted Langevin Algorithm.

    The algorithm, introduced by :cite:t:`laumont2022bayesian` runs the following markov chain iteration
    (Algorithm 2 from :cite:p:`laumont2022bayesian`):

    .. math::

        x_{k+1} = \Pi_{[a,b]} \left(x_{k} + \eta \nabla \log p(y|A,x_k) +
        \eta \alpha \nabla \log p(x_{k}) + \sqrt{2\eta}z_{k+1} \right).

    where :math:`x_{k}` is the :math:`k` th sample of the Markov chain,
    :math:`\log p(y|x)` is the log-likelihood function, :math:`\log p(x)` is the log-prior,
    :math:`\eta>0` is the step size, :math:`\alpha>0` controls the amount of regularization,
    :math:`\Pi_{[a,b]}(x)` projects the entries of :math:`x` to the interval :math:`[a,b]` and
    :math:`z\sim \mathcal{N}(0,I)` is a standard Gaussian vector.


    - Projected PnP-ULA assumes that the denoiser is :math:`L`-Lipschitz differentiable
    - For convergence, ULA requires that ``step_size`` is smaller than :math:`\frac{1}{L+\|A\|_2^2}`

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
         - Step size :math:`\eta` (default: 1.0)
       * - alpha
         - float
         - Regularization parameter :math:`\alpha` (default: 1.0)
       * - sigma
         - float
         - Noise level for the score model (default: 0.05)

    :return: Next state :math:`X_{t+1}` in the Markov chain
    :rtype: torch.Tensor

    """

    def __init__(self, algo_params: dict[str, float], clip=None):
        super().__init__(algo_params)

        # Raise an error if these are not supplied
        missing_params = []
        if "step_size" not in algo_params:
            missing_params.append("step_size")
        if "alpha" not in algo_params:
            missing_params.append("alpha")
        if "sigma" not in algo_params:
            missing_params.append("sigma")

        if missing_params:
            raise ValueError(
                f"Missing required parameters for ULA: {', '.join(missing_params)}"
            )

        self.clip = clip

    def forward(
        self,
        X: dict[str, Tensor],
        y: Tensor,
        physics: Physics,
        cur_data_fidelity: DataFidelity,
        cur_prior: ScorePrior,
        iteration: int,
        *args,
        **kwargs,
    ) -> dict[str, Tensor]:
        r"""
        Performs a single ULA sampling step using the Unadjusted Langevin Algorithm.

        Computes the next state in the Markov chain using the formula:

        .. math::

            x_{t+1} = x_t + \eta \nabla \log p(y|A,x_t) + \eta \alpha \nabla \log p(x_t) + \sqrt{2\eta}z_{t+1}

        where :math:`z_{t+1} \sim \mathcal{N}(0,I)` is a standard Gaussian noise vector.

        :param Dict X: Dictionary containing the current state :math:`x_t`.
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator :math:`A` that models the measurement process
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param ScorePrior cur_prior: Score-based prior model for :math:`\nabla \log p(x)`
        :param int iteration: Current iteration number in the sampling process (zero-indexed)

        :return: Dictionary `{"x": x}` containing the next state :math:`x_{t+1}` in the Markov chain.
        :rtype: Dict
        """
        x = X["x"]
        noise = torch.randn_like(x) * np.sqrt(2 * self.algo_params["step_size"])
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior = (
            -cur_prior.grad(x, self.algo_params["sigma"]) * self.algo_params["alpha"]
        )
        x_t = x + self.algo_params["step_size"] * (lhood + lprior) + noise
        if self.clip:
            x_t = projbox(x_t, self.clip[0], self.clip[1])
        return {"x": x_t}
