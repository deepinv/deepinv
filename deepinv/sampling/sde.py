import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Tuple, List, Any
import numpy as np
from .sde_solver import BaseSDESolver, SDEOutput


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


class DiffusionSDE(nn.Module):
    r"""
    Forward-time and Reverse-time Diffusion Stochastic Differential Equation with parameterized drift term.

    The forward SDE is defined by:

    .. math::
        d\, x_{t} = f(x_t, t) d\,t + g(t) d\, w_{t}.

    This forward SDE can be reversed by the following SDE running backward in time:

    .. math::
        d\, x_{t} = \left( f(x_t, t) - g(t)^2 \nabla \log p_t(x_t) \right) d\,t + g(t) d\, w_{t}.

    There also exists a deterministic probability flow ODE whose trajectories share the same marginal distribution as the SDEs:

    .. math::
        d\, x_{t} = \left( f(x_t, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x_t) \right) d\,t.


    The score function can be computed using Tweedie's formula, given a MMSE denoiser :math:`\denoisername`:

    .. math::
        \nabla \log p_{t}(x) = \left( \denoiser{x}{\sigma_t} - x \right) / \sigma_t^2

    where :math:`sigma_t` is the noise level at time :math:`t`, which can be accessed through the attribute ``sigma_t``

    Default parameters correspond to the `Ornstein-Uhlenbeck process <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_ SDE, defined in the time interval :math:`[0,1]`.

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param callable denoiser: a pre-trained MMSE denoiser which will be used to approximate the score function by Tweedie's formula.
    :param bool rescale: a boolean indicating whether to rescale the input and output of the denoiser to match the scale of the drift.
                        Should be set to `True` if the denoiser was trained on :math:`[0,1]`. Default to `False`.

    :param bool use_backward_ode: a boolean indicating whether to use the deterministic probability flow ODE for the backward process.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    """

    def __init__(
        self,
        drift: Callable = lambda x, t: -x,  # Default to Ornstein-Uhlenbeck process
        diffusion: Callable = lambda t: math.sqrt(2.0),
        use_backward_ode=False,
        denoiser: nn.Module = None,
        rescale: bool = False,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        if not rescale:
            self.denoiser = denoiser
        else:
            self.denoiser = (
                lambda x, sigma, *args, **kwargs: denoiser(
                    (x + 1) * 0.5, sigma / 2, *args, **kwargs
                )
                * 2.0
                - 1.0
            )
        self.device = device
        self.dtype = dtype
        self.use_backward_ode = use_backward_ode

        def forward_drift(x, t, *args, **kwargs):
            return drift(x, t, *args, **kwargs)

        def forward_diff(t):
            return diffusion(t)

        self.forward_sde = BaseSDE(
            drift=forward_drift,
            diffusion=forward_diff,
            dtype=dtype,
            device=device,
        )

        if self.use_backward_ode:

            def backward_drift(x, t, *args, **kwargs):
                return -drift(x, t) + 0.5 * diffusion(t) ** 2 * self.score(
                    x, t, *args, **kwargs
                )

            def backward_diff(t):
                return 0.0

        else:

            def backward_drift(x, t, *args, **kwargs):
                return -drift(x, t) + diffusion(t) ** 2 * self.score(
                    x, t, *args, **kwargs
                )

            def backward_diff(t):
                return diffusion(t)

        self.backward_sde = BaseSDE(
            drift=backward_drift,
            diffusion=backward_diff,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def forward(
        self,
        x_init: Union[Tensor, Tuple],
        solver: BaseSDESolver,
        seed: int = None,
        *args,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Sample the backward-SDE.

        :param Union[Tensor, Tuple] x_init: Initial condition of the backward-SDE.
            If it is a :meth:`torch.Tensor`, ``x_init`` should follow the distribution of :math:`p_T`, which is usually :math:`\mathcal{N}(0, \sigma_{\mathrm{max}}^2 \mathrm{Id})`.
            If it is a tuple, it should be of the form ``(B, C, H, W)``. A sample from the distribution :math:`p_T` will be generated automatically.

        :param BaseSDESolver: The solver to solve the backward-SDE, can be one of the methods available in :meth:`deepinv.sampling.sde_solver`.
        :param int seed: The seed for the random number generator, will be used in the solver and for the initial point if not given.
        :param args: additional arguments for the backward drift (passed to the `denoiser`).
        :param kwargs: additional keyword arguments for the backward drift (passed to the `denoiser`), e.g., `class_labels` for class-conditional models.

        :rtype: SDEOutput.

        """
        solver.rng_manual_seed(seed)

        if isinstance(x_init, (Tuple, List, torch.Size)):
            x_init = self.prior_sample(x_init, rng=solver.rng)
        return solver.sample(
            self.backward_sde,
            x_init,
            *args,
            **kwargs,
        )

    def prior_sample(
        self, shape: Union[List, Tuple, torch.Size], rng: torch.Generator = None
    ) -> Tensor:
        r"""
        Sample from the end-point distribution :math:`p_T` of the forward-SDE.

        :param shape: The shape of the the sample, of the form `(B, C, H, W)`.
        """
        return torch.randn(
            shape, generator=rng, device=self.device, dtype=self.dtype
        ) * self.sigma_t(1.0).view(-1, 1, 1, 1)

    def score(self, x: Tensor, t: Any, *args, **kwargs) -> Tensor:
        r"""
        Approximating the score function :math:`\nabla \log p_t` by the denoiser.

        :param torch.Tensor x: current state
        :param Any t: current time step
        :param args: additional arguments for the `denoiser`.
        :param kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.

        :return: the score function :math:`\nabla \log p_t(x)`.
        :rtype: torch.Tensor
        """
        t = self._handle_time_step(t)
        exp_minus_t = torch.exp(-t) if isinstance(t, Tensor) else np.exp(-t)
        return self.sigma_t(t).view(-1, 1, 1, 1) ** (-2) * (
            exp_minus_t
            * self.denoiser(
                x.to(torch.float32), self.sigma_t(t).to(torch.float32), *args, **kwargs
            ).to(self.dtype)
            - x
        )

    def _handle_time_step(self, t) -> Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def sigma_t(self, t: Any) -> Tensor:
        r"""
        The std of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(..., \sigma_t^2 \mathrm{Id})`.

        :param Any t: time step.

        :return torch.Tensor: the noise level at time step :attr:`t`.
        """
        t = self._handle_time_step(t)
        return torch.sqrt(1.0 - torch.exp(-2.0 * t))


class VESDE(DiffusionSDE):
    r"""
    `Variance-Exploding Stochastic Differential Equation (VE-SDE) <https://arxiv.org/abs/2011.13456>`_

    The forward-time SDE is defined as follows:

    .. math::
        d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t

    """

    def __init__(
        self,
        denoiser: nn.Module,
        rescale: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        use_backward_ode: bool = False,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        def forward_drift(x, t, *args, **kwargs):
            return 0.0

        def forward_diff(t):
            return self.sigma_t(t) * np.sqrt(
                2 * (np.log(sigma_max) - np.log(sigma_min))
            )

        super().__init__(
            drift=forward_drift,
            diffusion=forward_diff,
            use_backward_ode=use_backward_ode,
            denoiser=denoiser,
            rescale=rescale,
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

    def sigma_t(self, t):
        t = self._handle_time_step(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def score(self, x, t, *args, **kwargs):
        return self.sigma_t(t).view(-1, 1, 1, 1) ** (-2) * (
            self.denoiser(
                x.to(torch.float32), self.sigma_t(t).to(torch.float32), *args, **kwargs
            ).to(self.dtype)
            - x
        )
