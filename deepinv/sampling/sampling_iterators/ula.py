import torch.nn as nn
import torch
import numpy as np
import time as time
from deepinv.physics import LinearPhysics
from deepinv.optim import PnP
from deepinv.optim.prior import ScorePrior
from deepinv.sampling.sampling_iterators.sample_iterator import SamplingIterator
from deepinv.loss.regularisers import JacobianSpectralNorm


class ULAIterator(SamplingIterator):
    r"""
    Projected Plug-and-Play Unadjusted Langevin Algorithm.

    The algorithm runs the following markov chain iteration
    (Algorithm 2 from https://arxiv.org/abs/2103.04715):

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
        Performs a single ULA sampling step using the Unadjusted Langevin Algorithm.

        Computes the next state in the Markov chain using the formula:

        .. math::

            x_{t+1} = x_t + \eta \nabla \log p(y|A,x_t) + \eta \alpha \nabla \log p(x_t) + \sqrt{2\eta}z_{t+1}

        where :math:`z_{t+1} \sim \mathcal{N}(0,I)` is a standard Gaussian noise vector.

        :param torch.Tensor x: Current state :math:`x_t` of the Markov chain
        :param torch.Tensor y: Observed measurements/data tensor
        :param Physics physics: Forward operator :math:`A` that models the measurement process
        :param DataFidelity cur_data_fidelity: Negative log-likelihood function
        :param ScorePrior cur_prior: Score-based prior model for :math:`\nabla \log p(x)`
        :param dict cur_params: Dictionary containing the algorithm parameters (see table below)

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
        :return: Next state :math:`x_{t+1}` in the Markov chain
        :rtype: torch.Tensor
        """
        # Get parameters with defaults
        # TODO: raise error if we don't have these (no defaults)
        # TODO: check if this is reasonable 
        step_size = cur_params.get("step_size", 1) #self.compute_step_size(x, y, physics, cur_prior))
        alpha = cur_params.get("alpha", 1.0)
        sigma = cur_params.get("sigma", 0.05)
        
        noise = torch.randn_like(x) * np.sqrt(2 * step_size)
        lhood = -cur_data_fidelity.grad(x, y, physics)
        lprior = -cur_prior.grad(x, sigma) * alpha
        return x + step_size * (lhood + lprior) + noise
    
    # BUG: broken atm
    def compute_step_size(self, x, y, physics: LinearPhysics, prior):
        if not isinstance(physics, LinearPhysics):
            # TODO: raise warning here
            return 0.01
        physicsnorm = physics.compute_norm(x)
        # NOTE: eval wrong here?
        reg_l2 = JacobianSpectralNorm(max_iter=10, tol=1e-3, eval_mode=True, verbose=False)
        jacy = prior(y)
        priornorm = reg_l2(jacy, y)
        return 1/(priornorm + physicsnorm)

