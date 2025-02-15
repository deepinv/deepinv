import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Tuple, List, Any
import numpy as np
from .sde_solver import BaseSDESolver, SDEOutput
from deepinv.models.base import Reconstructor
from deepinv.optim.prior import Zero

class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):

    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion.

    It defines the common interface for drift and diffusion functions.

    :param callable drift: a time-dependent drift function :math:`f(x, t)`
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)`
    :param torch.dtype dtype: the data type of the computations.
    :param str device: the device for the computations.
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.dtype = dtype
        self.device = device

    def sample(
        self,
        x_init: Tensor = None,
        solver: BaseSDESolver = None,
        seed: int = None,
        *args,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the SDE with the given timesteps.

        :param torch.Tensor x_init: initial value.
        :param str method: method for solving the SDE. One of the methods available in :func:`deepinv.sampling.sde_solver`.

        :return SDEOutput: a namespaced container of the output.
        """
        solver.rng_manual_seed(seed)
        solution = solver.sample(self, x_init, *args, **kwargs)
        return solution

    def discretize(
        self, x: Tensor, t: Union[Tensor, float], *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Discretize the SDE at the given time step.

        :param torch.Tensor x: current state.
        :param float t: discretized time step.
        :param args: additional arguments for the drift.
        :param kwargs: additional keyword arguments for the drift.

        :return Tuple[Tensor, Tensor]: discretized drift and diffusion.
        """
        return self.drift(x, t, *args, **kwargs), self.diffusion(t)


class DiffusionSDE(BaseSDE):
    r"""
    Reverse-time Diffusion Stochastic Differential Equation defined by 
    
    .. math::
        d\, x_{t} = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_t(x_t) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}.

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param callable alpha: a scalar weighting the diffusion term. :math:`\alpha = 0` corresponds to the ODE sampling and :math:`\alpha > 0` corresponds to the SDE sampling.
    :param callable score_module: a module built to provide an approximation of the score at time :math:`t` :math:`\nabla \log p_t`, typically a denoiser.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    """

    def __init__(
        self,
        forward_drift: Callable,
        forward_diffusion: Callable,
        alpha: float = 1.,
        score_module: nn.Module = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        
        self.alpha = alpha
        self.score_module = score_module
        
        def backward_drift(x, t, *args, **kwargs):
            return -forward_drift(x, t) + ((1+alpha)/2)*forward_diffusion(t) ** 2 * self.score_module(x, t, *args, **kwargs)
        
        def backward_diffusion(t):
            return np.sqrt(alpha)*forward_diffusion(t)
        
        super().__init__(
            drift = backward_drift,
            diffusion = backward_diffusion,
            dtype = dtype, 
            device = device, 
            *args, 
            **kwargs)


class VarianceExplodingSDE(DiffusionSDE):
    r"""
    `Variance-Exploding Stochastic Differential Equation (VE-SDE) <https://arxiv.org/abs/2011.13456>`_

    The forward-time SDE is defined as follows:

    .. math::
        d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t

    This class is the reverse-time SDE of the VE-SDE, serving as the generation process.
    """

    def __init__(
        self,
        score_module: nn.Module = None,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        def forward_drift(x, t, *args, **kwargs):
            return 0.0

        def forward_diffusion(t):
            return self.sigma_t(t) * np.sqrt(
                2 * (np.log(sigma_max) - np.log(sigma_min))
            )

        super().__init__(
            drift=forward_drift,
            diffusion=forward_diffusion,
            score_module=score_module,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )

    def prior_sample(self, shape, rng) -> Tensor:
        return (
            torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            * self.sigma_max
        )
    
    def _handle_time_step(self, t) -> Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def sigma_t(self, t):
        t = self._handle_time_step(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    

class VEDiffusionReconstructor(Reconstructor):
    r"""
    """

    def __init__(
        self,
        data_fidelity : Zero,
        denoiser: nn.Module = None,   
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        *args,
        **kwargs,
    ):
        SDE = VarianceExplodingSDE(score_module=score_module, sigma_min=sigma_min, sigma_max=sigma_max, *args, *kwargs)