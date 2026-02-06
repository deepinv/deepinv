from __future__ import annotations
import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
from deepinv.physics import Physics
from deepinv.models.base import Reconstructor, Denoiser
from deepinv.optim.data_fidelity import ZeroFidelity
from deepinv.sampling.sde_solver import BaseSDESolver, SDEOutput
from deepinv.sampling.noisy_datafidelity import NoisyDataFidelity, DPSDataFidelity
from deepinv.sampling.utils import trapz_torch
from deepinv.models.wrapper import MinusOneOneDenoiserWrapper


class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):

    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion.
    It defines the common interface for drift and diffusion functions.

    :param Callable drift: a time-dependent drift function :math:`f(x, t)`
    :param Callable diffusion: a time-dependent diffusion function :math:`g(t)`
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: the data type of the computations.
    :param torch.device device: the device for the computations.
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.solver = solver
        self.dtype = dtype
        self.device = device

    def sample(
        self,
        x_init: Tensor = None,
        seed: int = None,
        get_trajectory: bool = False,
        *args,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the SDE with the given timesteps.

        :param torch.Tensor x_init: initial value.
        :param int seed: the seed for the pseudo-random number generator used in the solver.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to False
        :param \*args: additional arguments for the solver.
        :param \*\*kwargs: additional keyword arguments for the solver.

        :return : the generated sample (:class:`torch.Tensor` of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns (:class:`torch.Tensor`, :class:`torch.Tensor`) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.
        """
        self.solver.rng_manual_seed(seed)
        if isinstance(x_init, (tuple, list, torch.Size)):
            x_init = self.sample_init(x_init, rng=self.solver.rng)
        solution = self.solver.sample(
            self, x_init, *args, **kwargs, get_trajectory=get_trajectory
        )
        if get_trajectory:
            return solution.sample, solution.trajectory
        else:
            return solution.sample

    def discretize(
        self, x: Tensor, t: Tensor | float, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        r"""
        Discretize the SDE at the given time step.

        :param torch.Tensor x: current state.
        :param float t: discretized time step.
        :param \*args: additional arguments for the drift.
        :param \*\*kwargs: additional keyword arguments for the drift.

        :return tuple[Tensor, Tensor]: discretized drift and diffusion.
        """
        return self.drift(x, t, *args, **kwargs), self.diffusion(t)

    def sample_init(
        self, shape: list | tuple | torch.Size, rng: torch.Generator = None
    ) -> Tensor:
        r"""
        Sample from the end-time distribution of the forward diffusion.

        :param shape: The shape of the the sample, of the form `(B, C, H, W)`.
        """
        raise NotImplementedError

    def forward(
        self,
        x_init: Tensor = None,
        seed: int = None,
        get_trajectory: bool = False,
        *args,
        **kwargs,
    ) -> SDEOutput:
        r"""
        The forward function corresponds to SDE sampling.

        :param torch.Tensor x_init: initial value.
        :param int seed: the seed for the pseudo-random number generator used in the solver.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to False
        :param \*args: additional arguments for the solver.
        :param \*\*kwargs: additional keyword arguments for the solver.

        :return: the generated sample (:class:`torch.Tensor` of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns (:class:`torch.Tensor`, :class:`torch.Tensor`) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.
        """
        return self.sample(x_init, seed, get_trajectory, *args, **kwargs)


class DiffusionSDE(BaseSDE):
    r"""
    Define the Reverse-time Diffusion Stochastic Differential Equation.

    Given a forward-time SDE of the form:

    .. math::
        d x_t = f(x_t, t) dt + g(t)d w_t

    This class define the following reverse-time SDE:

    .. math::
        d x_{t} = \left( f(x_t, t) - \frac{1 + \alpha(t)}{2} g(t)^2 \nabla \log p_t(x_t) \right) dt + g(t) \sqrt{\alpha(t)} d w_{t}.

    :param Callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param Callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param Callable, float alpha: a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function :math:`\alpha(t) = 0` corresponds to ODE sampling and :math:`\alpha(t) > 0` corresponds to SDE sampling.
    :param deepinv.models.Denoiser: a denoiser used to provide an approximation of the score at time :math:`t` :math:`\nabla \log p_t`.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param bool minus_one_one: If `True`, wrap the denoiser so that SDE states `x` in [-1, 1] are converted to [0, 1] before denoising and mapped back afterward.
        Set `True` for denoisers trained on [0, 1] (all denoisers in :class:`deepinv.models.Denoiser`);
        set `False` only if the denoiser natively expects [-1, 1].
        This affects only the denoiser interface and usually improves quality when matched to the denoiser's training range.
        Default: `True`.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    :param \*args: additional arguments for the :class:`deepinv.sampling.BaseSDE`.
    :param \*\*kwargs: additional keyword arguments for the :class:`deepinv.sampling.BaseSDE`.
    """

    def __init__(
        self,
        forward_drift: Callable,
        forward_diffusion: Callable,
        alpha: Callable | float = 1.0,
        denoiser: nn.Module = None,
        solver: BaseSDESolver = None,
        minus_one_one: bool = True,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        if not isinstance(alpha, Callable):
            alpha_value = alpha

            def alpha(t: Tensor | float) -> float:
                return alpha_value

        def backward_drift(x, t, *args, **kwargs):
            return -forward_drift(x, t) + ((1 + alpha(t)) / 2) * forward_diffusion(
                t
            ) ** 2 * self.score(x, t, *args, **kwargs)

        def backward_diffusion(t):
            return (alpha(t) ** 0.5) * forward_diffusion(t)

        super().__init__(
            drift=backward_drift,
            diffusion=backward_diffusion,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

        self.alpha = alpha
        self.forward_drift = forward_drift
        self.forward_diffusion = forward_diffusion
        self.solver = solver
        self.denoiser = (
            denoiser if not minus_one_one else MinusOneOneDenoiserWrapper(denoiser)
        )
        self.minus_one_one = minus_one_one

    def score(self, x: Tensor, t: Tensor | float, *args, **kwargs) -> Tensor:
        r"""
        Approximating the score function :math:`\nabla \log p_t` by the denoiser.

        :param torch.Tensor x: current state
        :param torch.Tensor, float t: current time step
        :param \*args: additional arguments for the `denoiser`.
        :param \*\*kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.

        :return: the score function :math:`\nabla \log p_t(x)`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def _handle_time_step(self, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def sigma_t(
        self,
        t: Tensor | float,
    ) -> Tensor:
        r"""
        The :math:`\sigma(t)` of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})`.

        :param torch.Tensor, float t: current time step

        :return: the noise level at time step `t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def scale_t(self, t: Tensor | float) -> Tensor:
        r"""
        The scale :math:`s(t)` of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})`.

        :param torch.Tensor, float t: current time step

        :return: the mean of the condition distribution at time step `t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class EDMDiffusionSDE(DiffusionSDE):
    r"""
    Generative diffusion Stochastic Differential Equation.

    This class implements the diffusion generative SDE based on the formulation from :footcite:t:`karras2022elucidating` (with :math:`\beta(t) = \alpha(t) s(t)^2 \sigma(t) \sigma'(t)`):

    .. math::
        d x_t = \left(\frac{s'(t)}{s(t)} x_t - (1 + \alpha(t)) s(t)^2 \sigma(t) \sigma'(t) \nabla \log p_t(x_t) \right) dt + s(t) \sqrt{2 \alpha(t) \sigma(t) \sigma'(t)} d w_t

    where :math:`s(t)` is a time-dependent scale, :math:`\sigma(t)` is a time-dependent noise level, and :math:`\alpha(t)` is weighting the diffusion term.
    It corresponds to the reverse-time SDE of the following forward-time SDE:

    .. math::
        d x_t = \frac{s'(t)}{s(t)} x_t dt + s(t) \sqrt{2 \sigma(t) \sigma'(t)} d w_t

    The scale :math:`s(t)` and noise :math:`\sigma(t)` schedulers must satisfy :math:`s(0) = 1`, :math:`\sigma(0) = 0` and :math:`\lim_{t \to \infty} \sigma(t) = +\infty`.


    Common choices include the variance-preserving formulation :math:`s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}` and the variance-exploding formulation :math:`s(t) = 1`.

        - For choosing variance-preserving formulation, set `variance_preserving=True` and do not provide `scale_t` and `scale_prime_t`.
        - For choosing variance-exploding formulation, set `variance_exploding=True` and do not provide `scale_t` and `scale_prime_t`.

    .. note::

        This SDE must be solved by going reverse in time i.e. from :math:`t=T` to :math:`t=0`.

    :param Callable sigma_t: a time-dependent noise level schedule.
        It takes a time step `t` (either a Python ``float`` or a ``torch.Tensor``) as input  and returns the noise level at time `t` (either a Python ``float`` or a ``torch.Tensor``).
        Note that this is a required argument.
    :param Callable scale_t: a time-dependent scale schedule.
        It takes a time step `t` (either a Python ``float`` or a ``torch.Tensor``) as input  and returns the noise level at time `t` (either a Python ``float`` or a ``torch.Tensor``).
        If not provided, it will be set to :math:`s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}` if `variance_preserving=True`, or :math:`s(t) = 1` if `variance_exploding=True`.
        If both `variance_preserving` and `variance_exploding` are `False`, `scale_t` must be provided. Default to `None`.
    :param Callable sigma_prime_t: the derivative of `sigma_t`.
        It takes a time step `t` (either a Python ``float`` or a ``torch.Tensor``) as input and returns the noise level at time `t` (either a Python ``float`` or a ``torch.Tensor``).
        If not provided, it will be computed using autograd. Default to `None`.
    :param Callable scale_prime_t: the derivative of `scale_t`.
        It takes a time step `t` (either a Python ``float`` or a ``torch.Tensor``) as input and returns the noise level at time `t` (either a Python ``float`` or a ``torch.Tensor``).
        If not provided, it will be computed using autograd. Default to `None`.
    :param bool variance_preserving: whether to use a variance-preserving diffusion schedule, which imposes :math:`s(t) = \left(1 + \sigma(t)^2\right)^{-1/2}`. Default to `False`.
    :param bool variance_exploding: whether to use a variance-exploding diffusion schedule, which imposes :math:`s(t) = 1`. Default to `False`.
    :param Callable, float alpha: a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function :math:`\alpha(t) = 0` corresponds to ODE sampling and :math:`\alpha(t) > 0` corresponds to SDE sampling.
    :param float T: the end time of the forward SDE. Default to `1.0`.
    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t`: :math:`\nabla \log p_t`. Default to `None`.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE. Default to `None`.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed. Default to CPU.
    :param \*args: additional arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    :param \*\*kwargs: additional keyword arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    """

    def __init__(
        self,
        sigma_t: Callable,
        scale_t: Callable = None,
        sigma_prime_t: Callable = None,
        scale_prime_t: Callable = None,
        variance_preserving: bool = False,
        variance_exploding: bool = False,
        alpha: Callable | float = 1.0,
        T: float = 1.0,
        denoiser: nn.Module = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        self.T = T

        _sigma_t = sigma_t

        def sigma_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return _sigma_t(t)

        self.sigma_t = sigma_t

        assert not (
            variance_preserving and variance_exploding
        ), "Cannot set both variance_preserving and variance_exploding to True."

        if scale_t is None:
            if variance_preserving:

                def scale_t(t: Tensor | float) -> Tensor:
                    t = self._handle_time_step(t)
                    return (1 / (1 + self.sigma_t(t) ** 2)) ** 0.5

                def scale_prime_t(t: Tensor | float) -> Tensor:
                    self._handle_time_step(t)
                    return (
                        -self.sigma_t(t)
                        * self.sigma_prime_t(t)
                        * (1 / (1 + self.sigma_t(t) ** 2)) ** 1.5
                    )

            elif variance_exploding:

                def scale_t(t: Tensor | float) -> Tensor:
                    t = self._handle_time_step(t)
                    return torch.ones_like(t)

                def scale_prime_t(t: Tensor | float) -> Tensor:
                    t = self._handle_time_step(t)
                    return torch.zeros_like(t)

            else:
                raise ValueError(
                    "'scale_t' must be provided if 'variance_preserving' and 'variance_exploding' is False"
                )
        else:
            _scale_t = scale_t

            def scale_t(t: Tensor | float) -> Tensor:
                t = self._handle_time_step(t)
                return _scale_t(t)

        self.scale_t = scale_t

        if sigma_prime_t is None:

            def sigma_prime_t(t: Tensor | float) -> Tensor:
                with torch.enable_grad():
                    t = self._handle_time_step(t).requires_grad_(True)
                    sigma = self.sigma_t(t)
                    grad = torch.autograd.grad(
                        sigma.sum(), t, create_graph=False, retain_graph=False
                    )[0]
                    return grad

        else:
            _sigma_prime_t = sigma_prime_t

            def sigma_prime_t(t: Tensor | float) -> Tensor:
                t = self._handle_time_step(t)
                return _sigma_prime_t(t)

        self.sigma_prime_t = sigma_prime_t

        if scale_prime_t is None:

            def scale_prime_t(t: Tensor | float) -> Tensor:
                with torch.enable_grad():
                    t = self._handle_time_step(t).requires_grad_(True)
                    scale = self.scale_t(t)
                    grad = torch.autograd.grad(
                        scale.sum(), t, create_graph=False, retain_graph=False
                    )[0]
                    return grad

        else:
            _scale_prime_t = scale_prime_t

            def scale_prime_t(t: Tensor | float) -> Tensor:
                t = self._handle_time_step(t)
                return _scale_prime_t(t)

        self.scale_prime_t = scale_prime_t

        def forward_drift(x, t, *args, **kwargs):
            return (scale_prime_t(t) / scale_t(t)) * x

        def forward_diffusion(t):
            return scale_t(t) * torch.sqrt(2 * sigma_t(t) * sigma_prime_t(t))

        super().__init__(
            forward_drift=forward_drift,
            forward_diffusion=forward_diffusion,
            alpha=alpha,
            denoiser=denoiser,
            solver=solver,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )

    def score(self, x: Tensor, t: Tensor | float, *args, **kwargs) -> Tensor:
        r"""
        Approximating the score function :math:`\nabla \log p_t` by the denoiser.

        :param torch.Tensor x: current state
        :param torch.Tensor, float t: current time step
        :param \*args: additional arguments for the `denoiser`.
        :param \*\*kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.

        :return: the score function :math:`\nabla \log p_t(x)`.

        """
        sigma = self.sigma_t(t)
        scale = self.scale_t(t)
        x_in = x / scale
        model_output = self.denoiser(
            x_in.to(torch.float32),
            sigma.to(torch.float32),
            *args,
            **kwargs,
        ).to(self.dtype)
        return self._score_from_model_output(x, model_output, sigma, scale)

    def _score_from_model_output(
        self, x: Tensor, model_output: Tensor, sigma: Tensor, scale: Tensor
    ) -> Tensor:
        denoised = scale * model_output
        score = (denoised - x.to(self.dtype)) / (scale * sigma).pow(2)
        return score

    def sample_init(self, shape, rng: torch.Generator) -> Tensor:
        r"""
        Sample from the initial distribution of the reverse-time diffusion SDE, which is a Gaussian with zero mean and covariance matrix :math:` s(T)^2 \sigma(T)^2 \operatorname{Id}`.

        :param tuple shape: The shape of the sample to generate
        :param torch.Generator rng: Random number generator for reproducibility
        :return: A sample from the prior distribution
        :rtype: torch.Tensor
        """
        init = (
            torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            * self.sigma_t(self.T)
            * self.scale_t(self.T)
        )
        return init


class SongDiffusionSDE(EDMDiffusionSDE):
    r"""
    Generative diffusion Stochastic Differential Equation.

    This class implements the diffusion generative SDE based the formulation from :footcite:t:`song2020score`:

    .. math::
        d x_t = -\left(\frac{1}{2} \beta(t) x_t + \frac{1 + \alpha(t)}{2} g(t) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha(t) g(t)} d w_t

    where :math:`\beta(t)` is a time-dependent linear drift, :math:`g(t)` is a time-dependent linear diffusion, and
    :math:`\alpha(t)` is weighting the diffusion term.

    It corresponds to the reverse-time SDE of the following forward-time SDE:

    .. math::
        d x_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{g(t)} d w_t

    Compared to the EDM formulation in :class:`deepinv.sampling.EDMDiffusionSDE`, the scale :math:`s(t)` and noise :math:`\sigma(t)` schedulers are defined with respect to :math:`\beta(t)` and :math:`g(t)` as follows:

    .. math::
        s(t) = \exp\left(-\int_0^t \beta(s) ds\right), \quad \sigma(t) = \sqrt{2 \int_0^t \frac{g(s)}{s(s)^2} ds}.

    Common choices include the variance-preserving formulation :math:`\beta(t) = g(t)` and the variance-exploding formulation :math:`\beta(t) = 0`.

        - For choosing variance-preserving formulation, set `variance_preserving=True` and `beta_t` and `xi_t` will be automatically set to be the same function.
        - For choosing variance-exploding formulation, set `variance_exploding=True` and `beta_t` will be automatically set to `0`.

    .. note::

        This SDE must be solved going reverse in time i.e. from :math:`t=T` to :math:`t=0`.

    :param Callable beta_t: a time-dependent linear drift of the forward-time SDE.
    :param Callable B_t: time integral of beta_t between 0 and t. If None, it is calculated by numerical integration.
    :param Callable xi_t: a time-dependent linear diffusion of the forward-time SDE.
    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t`: :math:`\nabla \log p_t`.
    :param Callable, float alpha: a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function :math:`\alpha(t) = 0` corresponds to ODE sampling and :math:`\alpha(t) > 0` corresponds to SDE sampling.
    :param float T: the end time of the forward SDE.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    :param \*args: additional arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    :param \*\*kwargs: additional keyword arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    """

    def __init__(
        self,
        beta_t: Callable = None,
        B_t: Callable = None,
        xi_t: Callable = None,
        variance_preserving: bool = False,
        variance_exploding: bool = False,
        alpha: Callable | float = 0.,
        T: float = 1.0,
        denoiser: nn.Module = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        if variance_preserving:
            if beta_t is not None:
                xi_t = beta_t
            elif xi_t is not None:
                beta_t = xi_t
            else:
                raise ValueError(
                    "Either beta_t or xi_t must be provided if variance_preserving is True"
                )
        elif variance_exploding:
            beta_t = lambda t: 0 * self._handle_time_step(t)
            B_t = lambda t: 0 * self._handle_time_step(t)

        if B_t is None:

            def B_t(t: Tensor | float, n_steps: int = 100) -> Tensor:
                t = self._handle_time_step(t)
                return trapz_torch(
                    beta_t, torch.tensor(0.0, device=t.device), t, n_steps
                )

        def scale_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return torch.exp(-B_t(t))

        def scale_prime_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return -beta_t(t) * scale_t(t)

        def sigma_t(t: Tensor | float, n_steps: int = 100) -> Tensor:
            t = self._handle_time_step(t)
            if variance_preserving:
                return (1 / scale_t(t) ** 2 - 1) ** 0.5
            else:

                def integrand(s: torch.Tensor) -> torch.Tensor:
                    return xi_t(s) / (scale_t(s) ** 2)

                integral = trapz_torch(
                    integrand, torch.tensor(0.0, device=t.device), t, n_steps
                )
                return (2 * integral).sqrt()

        def sigma_prime_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return (xi_t(t) / (scale_t(t) ** 2)) * (1 / sigma_t(t))

        super().__init__(
            sigma_t=sigma_t,
            sigma_prime_t=sigma_prime_t,
            scale_t=scale_t,
            scale_prime_t=scale_prime_t,
            variance_preserving=variance_preserving,
            variance_exploding=variance_exploding,
            alpha=alpha,
            T=T,
            denoiser=denoiser,
            solver=solver,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )


class FlowMatching(EDMDiffusionSDE):
    r"""
    Generative Flow Matching process.

    It corresponds to the reverse-time SDE of the following forward-time noising process, which corresponds to a linear interpolation between data and Gaussian noise:

    .. math::
        x_t = a_t x_0 + b_t z \quad \mbox{ where } x_0 \sim p_{data}  \mbox{ and } z \sim \mathcal{N}(0, I)

    The schedulers :math:`a(t)` and :math:`b(t)` must satisfy :math:`a(0) = 1`, :math:`b(0) = 0`, :math:`a(1) = 0`, and :math:`b(1) = 1`.

    Compared to the EDM formulation in :class:`deepinv.sampling.EDMDiffusionSDE`, the scale :math:`s(t)` and noise :math:`\sigma(t)` schedulers are defined with respect to :math:`a(t)` and :math:`b(t)` as follows:

    .. math::
        s(t) = a(t), \quad \sigma(t) = \frac{b(t)}{a(t)} .

    .. note::

        This SDE must be solved going reverse in time i.e. from :math:`t=1` to :math:`t=0`.
        Note that in order to unify flow matching and diffusion models, we set the starting time of the generating process (noise distribution) to be 1, and the ending time of the generating process (data distribution) to be 0, which is different from the convention in the flow matching literature.


    :param Callable a_t: time-dependent parameter :math:`a(t)` of flow-matching. Default to `lambda t: 1-t`.
    :param Callable a_prime_t: time derivative :math:`a'(t)` of :math:`a(t)`. Default to `lambda t: -1`.
    :param Callable b_t: time-dependent parameter :math:`b(t)` of flow-matching.Default to `lambda t: t`.
    :param Callable b_prime_t: time derivative :math:`b'(t)` of :math:`b(t)`. Default to `lambda t: 1`.
    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t`: :math:`\nabla \log p_t`.
    :param Callable, float alpha: a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function :math:`\alpha(t) = 0` corresponds to ODE sampling and :math:`\alpha(t) > 0` corresponds to SDE sampling.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    :param \*args: additional arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    :param \*\*kwargs: additional keyword arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    """

    def __init__(
        self,
        a_t: Callable = lambda t: 1 - t,
        a_prime_t: Callable = lambda t: -1.0,
        b_t: Callable = lambda t: t,
        b_prime_t: Callable = lambda t: 1.0,
        T: float = 0.99,
        alpha: Callable | float = 0.0,
        denoiser: nn.Module = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):

        def scale_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return a_t(t)

        def scale_prime_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return a_prime_t(t)

        def sigma_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return b_t(t) / a_t(t)

        def sigma_prime_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return (b_prime_t(t) * a_t(t) - b_t(t) * a_prime_t(t)) / (a_t(t) ** 2)

        super().__init__(
            scale_t=scale_t,
            scale_prime_t=scale_prime_t,
            sigma_t=sigma_t,
            sigma_prime_t=sigma_prime_t,
            alpha=alpha,
            T=T,
            denoiser=denoiser,
            solver=solver,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    def velocity(self, x: Tensor, t: Tensor | float, *args, **kwargs) -> Tensor:
        r"""
        Computes the velocity field of the flow matching process, which is defined as the drift of the backward SDE.

        :param torch.Tensor x: current state
        :param torch.Tensor, float t: current timestep
        :param \*args: additional arguments for the `denoiser`.
        :param \*\*kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.

        :return: the velocity field at state `x` and time `t`.
        :rtype: torch.Tensor
        """
        return self.drift(x, t, *args, **kwargs)


class VarianceExplodingDiffusion(EDMDiffusionSDE):

    def __init__(
        self,
        denoiser: nn.Module = None,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        alpha: Callable | float = 0.25,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):

        def sigma_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return sigma_min * (sigma_max / sigma_min) ** t

        def sigma_prime_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            return self.sigma_t(t) * (np.log(sigma_max) - np.log(sigma_min))

        super().__init__(
            sigma_t=sigma_t,
            sigma_prime_t=sigma_prime_t,
            variance_exploding=True,
            T=1,
            alpha=alpha,
            denoiser=denoiser,
            solver=solver,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )


class VariancePreservingDiffusion(SongDiffusionSDE):
    r"""
    Variance-Preserving Stochastic Differential Equation (VP-SDE).

    This class implements the reverse-time SDE of the Variance-Preserving SDE (VP-SDE) :footcite:t:`song2020score`.

    The forward-time SDE is defined as follows:

    .. math::
        d x_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} d w_t \quad \mbox{ where } \beta(t) = \beta_{\mathrm{min}}  + t \left( \beta_{\mathrm{max}} - \beta_{\mathrm{min}} \right)

    The reverse-time SDE is defined as follows:

    .. math::
        d x_t = -\left(\frac{1}{2} \beta(t) x_t + \frac{1 + \alpha(t)}{2} \beta(t) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha(t) \beta(t)} d w_t

    where :math:`\alpha(t)` is weighting the diffusion term.

    This class is the reverse-time SDE of the VP-SDE, serving as the generation process.

    .. note::

        This SDE must be solved going reverse in time i.e. from :math:`t=T` to :math:`t=0`.

    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t`: :math:`\nabla \log p_t`.
    :param float beta_min: the minimum noise level.
    :param float beta_max: the maximum noise level.
    :param Callable, float alpha: a (possibly time-dependent) positive scalar weighting the diffusion term. A  constant function :math:`\alpha(t) = 0` corresponds to ODE sampling and :math:`\alpha(t) > 0` corresponds to SDE sampling.
    :param bool scaled_linear: whether to use the scaled linear beta schedule. If `False`, uses the more standard linear schedule. Default to `False`.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    :param \*args: additional arguments for the :class:`deepinv.sampling.DiffusionSDE`.
    :param \*\*kwargs: additional keyword arguments for the :class:`deepinv.sampling.DiffusionSDE`.

    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        beta_min: float = 0.1,
        beta_max: float = 20,
        alpha: Callable | float = 0.,
        scaled_linear: bool = False,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):

        def beta_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            if not scaled_linear:
                return beta_min + t * (beta_max - beta_min)
            else:
                beta_min_sqrt = np.sqrt(beta_min)
                beta_max_sqrt = np.sqrt(beta_max)
                return (beta_min_sqrt + t * (beta_max_sqrt - beta_min_sqrt)) ** 2

        def B_t(t: Tensor | float) -> Tensor:
            t = self._handle_time_step(t)
            if not scaled_linear:
                return beta_min * t + 0.5 * t**2 * (beta_max - beta_min)
            else:
                beta_min_sqrt = np.sqrt(beta_min)
                beta_max_sqrt = np.sqrt(beta_max)
                a = beta_min_sqrt
                c = beta_max_sqrt - beta_min_sqrt
                return (a**2) * t + a * c * t**2 + (c**2 / 3.0) * t**3

        super().__init__(
            beta_t=beta_t,
            B_t=B_t,
            variance_preserving=True,
            alpha=alpha,
            T=1,
            denoiser=denoiser,
            solver=solver,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )


class PosteriorDiffusion(Reconstructor):
    r"""
    Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).

    Consider the acquisition model:

    .. math::
        y = \noise{\forw{x}}.

    This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:

    .. math::
        d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha(t)}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha(t)} d\, w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion. The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `sde`. The (conditional) score function :math:`\nabla_{x_t} \log p_t(x_t | y)` can be decomposed using the Bayes' rule:

    .. math::
        \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

    The first term is the score function of the unconditional SDE, which is typically approximated by a MMSE denoiser using the well-known Tweedie's formula, while the second term is approximated by the (noisy) data-fidelity term. We implement various data-fidelity terms in :class:`deepinv.sampling.NoisyDataFidelity`.

    :param deepinv.sampling.NoisyDataFidelity data_fidelity: the noisy data-fidelity term, used to approximate the score :math:`\nabla_{x_t} \log p_t(y \vert x_t)`. Default to :class:`deepinv.optim.ZeroFidelity`, which corresponds to the zero data-fidelity term and the sampling process boils down to the unconditional SDE sampling.
    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the (unconditional) score at time :math:`t` :math:`\nabla \log p_t`.
    :param deepinv.sampling.DiffusionSDE sde: the forward-time SDE, which defines the drift and diffusion terms of the reverse-time SDE.
    :param deepinv.sampling.BaseSDESolver solver: the solver for the SDE. If not specified, the solver from the `sde` will be used.
    :param torch.dtype dtype: the data type of the sampling solver, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: the device for the computations.
    :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to `False`.
    :param bool minus_one_one: If `True`, wrap the denoiser so that SDE states `x` in `[-1, 1]` are converted to `[0, 1]` before denoising and mapped back afterward.

        - Set `True` for denoisers trained on `[0, 1]` data range (all denoisers in :class:`deepinv.models.Denoiser`).
        - Set `False` only if the denoiser natively expects `[-1, 1]` data range.

        This affects only the denoiser interface and usually improves quality when matched to the denoiser's training range.
        Default: `True`.

    """

    def __init__(
        self,
        data_fidelity: NoisyDataFidelity | None = None,
        denoiser: Denoiser = None,
        sde: DiffusionSDE = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        verbose: bool = False,
        minus_one_one: bool = True,
        *args,
        **kwargs,
    ):
        if data_fidelity is None:
            data_fidelity = ZeroFidelity()
        super().__init__(device=device)
        self.data_fidelity = data_fidelity
        self.sde = sde
        self.minus_one_one = minus_one_one
        assert (
            denoiser is not None or sde.denoiser is not None
        ), "A denoiser must be specified."
        if denoiser is None:
            denoiser = sde.denoiser

        self.sde.denoiser = denoiser
        if hasattr(self.data_fidelity, "denoiser"):
            self.data_fidelity.denoiser = denoiser

        assert (
            solver is not None or sde.solver is not None
        ), "A SDE solver must be specified."
        if solver is not None:
            self.solver = solver
        else:
            self.solver = sde.solver
        self.dtype = dtype
        self.device = device
        self.verbose = verbose

        def backward_drift(x, t, y, physics, *args, **kwargs):
            return -self.sde.forward_drift(x, t) + (
                (1 + self.sde.alpha(t)) / 2
            ) * self.sde.forward_diffusion(t) ** 2 * self.score(
                y, physics, x, t, *args, **kwargs
            )

        def backward_diffusion(t):
            return self.sde.diffusion(t)

        self.posterior = BaseSDE(
            drift=backward_drift,
            diffusion=backward_diffusion,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        x_init: Tensor | None = None,
        seed: int = None,
        timesteps: Tensor = None,
        get_trajectory: bool = False,
        *args,
        **kwargs,
    ):
        r"""
        Sample the posterior distribution :math:`p(x|y)` given the data measurement :math:`y`.

        :param torch.Tensor y: the data measurement.
        :param deepinv.physics.Physics physics: the forward operator.
        :param torch.Tensor, tuple x_init: the initial value for the sampling, can be a :class:`torch.Tensor` or a tuple `(B, C, H, W)`, indicating the shape of the initial point, matching the shape of `physics` and `y`. In this case, the initial value is taken randomly following the end-point distribution of the `sde`.
        :param int seed: the random seed for reproducibility, the same samples will be generated for the same seed. Default to `None`.
        :param torch.Tensor timesteps: the time steps for the solver. If `None`, the default time steps in the solver will be used. Default to `None`.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to `False`.
        :param \*args: the additional arguments for the solver.
        :param \*\*kwargs: the additional keyword arguments for the solver.

        :return: the generated sample (:class:`torch.Tensor` of shape `(B, C, H, W)`) if `get_trajectory` is `False`. Otherwise, returns a tuple (:class:`torch.Tensor`, :class:`torch.Tensor`) of shape `(B, C, H, W)` and `(N, B, C, H, W)` where `N` is the number of steps.
        """
        self.solver.rng_manual_seed(seed)
        if isinstance(x_init, (tuple, list, torch.Size)):
            x_init = self.sde.sample_init(x_init, rng=self.solver.rng)
        elif x_init is None:
            if physics is not None:
                x_init = self.sde.sample_init(
                    physics.A_dagger(y).shape, rng=self.solver.rng
                )
            elif y is not None:
                x_init = self.sde.sample_init(y.shape, rng=self.solver.rng)
            else:
                raise ValueError("Either `x_init` or `physics` must be specified.")
        solution = self.solver.sample(
            self.posterior,
            x_init,
            y=y,
            physics=physics,
            timesteps=timesteps,
            get_trajectory=get_trajectory,
            verbose=self.verbose,
            *args,
            **kwargs,
        )
        # Scale the output back to [0, 1]
        sample = solution.sample
        if self.minus_one_one:
            sample = (sample.clamp(-1, 1) + 1) / 2
        if get_trajectory:
            return sample, solution.trajectory
        else:
            return sample

    def score(
        self,
        y: Tensor,
        physics: Physics,
        x: Tensor,
        t: Tensor | float,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""
        Approximating the conditional score :math:`\nabla_{x_t} \log p_t(x_t \vert y)`.

        :param torch.Tensor y: the data measurement.
        :param deepinv.physics.Physics physics: the forward operator.
        :param torch.Tensor x: the current state.
        :param torch.Tensor, float t: the current time step.
        :param \*args: additional arguments for the score function of the unconditional SDE.
        :param \*\*kwargs: additional keyword arguments for the score function of the unconditional SDE.

        :return: the score function :math:`\nabla_{x_t} \log p_t(x_t \vert y)`.
        :rtype: torch.Tensor
        """

        if isinstance(self.data_fidelity, ZeroFidelity):
            return self.sde.score(x, t, *args, **kwargs).to(self.dtype)
        else:
            sigma = self.sde.sigma_t(t)
            scale = self.sde.scale_t(t)

            if isinstance(self.sde, EDMDiffusionSDE) and isinstance(
                self.data_fidelity, DPSDataFidelity
            ):
                # For EDM, we can compute the score from model output directly, avoid redundant computation
                data_fid_grad, model_output = self.data_fidelity.grad(
                    (x / scale), y, physics=physics, sigma=sigma, get_model_outputs=True
                )
                score = self.sde._score_from_model_output(
                    x, model_output, sigma, scale
                ) - data_fid_grad / scale.to(self.dtype)
            else:
                score = (
                    self.sde.score(x, t, *args, **kwargs).to(self.dtype)
                    - self.data_fidelity.grad(
                        (x / scale),
                        y,
                        physics=physics,
                        sigma=sigma,
                    ).to(self.dtype)
                    / scale
                )
            return score
