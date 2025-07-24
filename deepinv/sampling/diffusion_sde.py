import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional
import numpy as np
from deepinv.physics import Physics
from deepinv.models.base import Reconstructor, Denoiser
from deepinv.optim.data_fidelity import ZeroFidelity
from deepinv.sampling.sde_solver import BaseSDESolver, SDEOutput
from deepinv.sampling.noisy_datafidelity import NoisyDataFidelity
from copy import deepcopy


class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):min_num_steps

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
        self, x: Tensor, t: Union[Tensor, float], *args, **kwargs
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
        self, shape: Union[list, tuple, torch.Size], rng: torch.Generator = None
    ) -> Tensor:
        r"""
        Sample from the end-time distribution of the forward diffusion.

        :param shape: The shape of the the sample, of the form `(B, C, H, W)`.
        """
        raise NotImplementedError


class DiffusionSDE(BaseSDE):
    r"""
    Define the Reverse-time Diffusion Stochastic Differential Equation.

    Given a forward-time SDE of the form:

    .. math::
        d x_t = f(x_t, t) dt + g(t)d w_t

    This class define the following reverse-time SDE:

    .. math::
        d x_{t} = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_t(x_t) \right) dt + g(t) \sqrt{\alpha} d w_{t}.

    :param Callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param Callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param Callable alpha: a scalar weighting the diffusion term. :math:`\alpha = 0` corresponds to the ODE sampling and :math:`\alpha > 0` corresponds to the SDE sampling.
    :param deepinv.models.Denoiser: a denoiser used to provide an approximation of the score at time :math:`t` :math:`\nabla \log p_t`.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.
    """

    def __init__(
        self,
        forward_drift: Callable,
        forward_diffusion: Callable,
        alpha: float = 1.0,
        denoiser: nn.Module = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        def backward_drift(x, t, *args, **kwargs):
            return -forward_drift(x, t) + ((1 + alpha) / 2) * forward_diffusion(
                t
            ) ** 2 * self.score(x, t, *args, **kwargs)

        def backward_diffusion(t):
            return (alpha**0.5) * forward_diffusion(t)

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
        self.denoiser = deepcopy(denoiser)

    def score(self, x: Tensor, t: Union[Tensor, float], *args, **kwargs) -> Tensor:
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

    def _handle_time_step(self, t: Union[Tensor, float]) -> Tensor:
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def sigma_t(
        self,
        t: Union[Tensor, float],
    ) -> Tensor:
        r"""
        The :math:`\sigma(t)` of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})`.

        :param torch.Tensor, float t: current time step

        :return: the noise level at time step `t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def scale_t(self, t: Union[Tensor, float]) -> Tensor:
        r"""
        The scale :math:`s(t)` of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(s(t)x_0, s(t)^2 \sigma_t^2 \mathrm{Id})`.

        :param torch.Tensor, float t: current time step

        :return: the mean of the condition distribution at time step `t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class VarianceExplodingDiffusion(DiffusionSDE):
    r"""
    Variance-Exploding Stochastic Differential Equation (VE-SDE).

    This class implements the reverse-time SDE of the Variance-Exploding SDE (VE-SDE) :footcite:t:`song2020score`.

    The forward-time SDE is defined as follows:

    .. math::
        d x_t = g(t) d w_t \quad \mbox{where } g(t) = \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t

    The reverse-time SDE is defined as follows:

    .. math::
        d x_t = -\frac{1 + \alpha}{2} \sigma_{\mathrm{min}}^2 \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^{2t} \nabla \log p_t(x_t) dt + \sqrt{\alpha} \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t d w_t

    where :math:`\alpha \in [0,1]` is a constant weighting the diffusion term.

    This class is the reverse-time SDE of the VE-SDE, serving as the generation process.

    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t` :math:`\nabla \log p_t(x_t)`.
    :param float sigma_min: the minimum noise level.
    :param float sigma_max: the maximum noise level.
    :param float alpha: the weighting factor of the diffusion term.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.


    """

    def __init__(
        self,
        denoiser: nn.Module = None,
        sigma_min: float = 0.005,
        sigma_max: float = 100,
        alpha: float = 1.0,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        def forward_drift(x, t, *args, **kwargs):
            r"""
            The drift term of the forward VE-SDE is :math:`0`.

            :param torch.Tensor x: The current state
            :param torch.Tensor, float t: The current time
            :return: The drift term, which is 0 for VE-SDE since it only has a diffusion term
            :rtype: float
            """
            return 0.0

        def forward_diffusion(t):
            r"""
            The diffusion coefficient of the forward VE-SDE.

            :param torch.Tensor, float t: The current time
            :return: The diffusion coefficient at time t
            :rtype: float
            """
            return self.sigma_t(t) * np.sqrt(
                2 * (np.log(sigma_max) - np.log(sigma_min))
            )

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

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sample_init(self, shape, rng: torch.Generator) -> Tensor:
        r"""
        Sample from the initial distribution of the reverse-time diffusion SDE, which is a Gaussian with zero mean and covariance matrix :math:`\sigma_{max}^2 \operatorname{Id}`.

        :param tuple shape: The shape of the sample to generate
        :param torch.Generator rng: Random number generator for reproducibility
        :return: A sample from the prior distribution
        :rtype: torch.Tensor
        """
        return (
            torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            * self.sigma_max
        )

    def sigma_t(self, t: Union[Tensor, float]) -> Tensor:
        t = self._handle_time_step(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def scale_t(self, t: Union[Tensor, float]) -> Tensor:
        return torch.ones_like(self._handle_time_step(t))

    def score(self, x: Tensor, t: Union[Tensor, float], *args, **kwargs) -> Tensor:
        std = self.sigma_t(t)
        denoised = self.denoiser(
            x.to(torch.float32), self.sigma_t(t).to(torch.float32), *args, **kwargs
        ).to(self.dtype)
        score = (denoised - x.to(self.dtype)) / std.pow(2)
        return score


class VariancePreservingDiffusion(DiffusionSDE):
    r"""
    Variance-Preserving Stochastic Differential Equation (VP-SDE).

    This class implements the reverse-time SDE of the Variance-Preserving SDE (VP-SDE) :footcite:t:`song2020score`.

    The forward-time SDE is defined as follows:

    .. math::
        d x_t = -\frac{1}{2} \beta(t) x_t dt + \sqrt{\beta(t)} d w_t \quad \mbox{ where } \beta(t) = \beta_{\mathrm{min}}  + t \left( \beta_{\mathrm{max}} - \beta_{\mathrm{min}} \right)

    The reverse-time SDE is defined as follows:

    .. math::
        d x_t = -\left(\frac{1}{2} \beta(t) x_t + \frac{1 + \alpha}{2} \beta(t) \nabla \log p_t(x_t) \right) dt + \sqrt{\alpha \beta(t)} d w_t

    where :math:`\alpha \in [0,1]` is a constant weighting the diffusion term.

    This class is the reverse-time SDE of the VP-SDE, serving as the generation process.

    :param deepinv.models.Denoiser denoiser: a denoiser used to provide an approximation of the score at time :math:`t`: :math:`\nabla \log p_t`.
    :param float beta_min: the minimum noise level.
    :param float beta_max: the maximum noise level.
    :param float alpha: the weighting factor of the diffusion term.
    :param deepinv.sampling.BaseSDESolver solver: the solver for solving the SDE.
    :param torch.dtype dtype: data type of the computation, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: device on which the computation is performed.


    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        beta_min: float = 0.0001,
        beta_max: float = 5.0,
        alpha: float = 1.0,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        def forward_drift(x, t, *args, **kwargs):
            r"""
            The drift term of the forward VE-SDE is :math:`0`.

            :param torch.Tensor x: The current state
            :param torch.Tensor, float t: The current time
            :return: The drift term, which is 0 for VE-SDE since it only has a diffusion term
            :rtype: float
            """
            return -0.5 * self._beta_t(t).view(-1, 1, 1, 1) * x

        def forward_diffusion(t):
            r"""
            The diffusion coefficient of the forward VE-SDE.

            :param torch.Tensor, float t: The current time
            :return: The diffusion coefficient at time t
            :rtype: float
            """
            return self._beta_t(t).view(-1, 1, 1, 1).sqrt()

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

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min

    def sample_init(self, shape, rng: torch.Generator) -> Tensor:
        r"""
        Sample from the initial distribution of the reverse-time diffusion SDE, which is the standard Gaussian distribution.

        :param tuple shape: The shape of the sample to generate
        :param torch.Generator rng: Random number generator for reproducibility
        :return: A sample from the prior distribution
        :rtype: torch.Tensor
        """
        return torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)

    def _beta_t(self, t: Union[Tensor, float]) -> Tensor:
        t = self._handle_time_step(t)
        return self.beta_min + t * self.beta_d

    def sigma_t(self, t: Union[Tensor, float]) -> Tensor:
        t = self._handle_time_step(t)
        return torch.sqrt(torch.exp(0.5 * t**2 * self.beta_d + t * self.beta_min) - 1.0)

    def scale_t(self, t: Union[Tensor, float]) -> Tensor:
        t = self._handle_time_step(t)
        return 1 / torch.sqrt(torch.exp(0.5 * t**2 * self.beta_d + t * self.beta_min))

    def score(self, x: Tensor, t: Union[Tensor, float], *args, **kwargs) -> Tensor:
        sigma = self.sigma_t(t)
        scale = self.scale_t(t)

        denoised = scale * self.denoiser(
            (x / scale).to(torch.float32),
            sigma.to(torch.float32),
            *args,
            **kwargs,
        ).to(self.dtype)
        score = (denoised - x.to(self.dtype)) / (scale * sigma).pow(2)

        return score


class PosteriorDiffusion(Reconstructor):
    r"""
    Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).

    Consider the acquisition model:

    .. math::
        y = \noise{\forw{x}}.

    This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:

    .. math::
        d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}

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
    :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.
    """

    def __init__(
        self,
        data_fidelity: NoisyDataFidelity = ZeroFidelity(),
        denoiser: Denoiser = None,
        sde: DiffusionSDE = None,
        solver: BaseSDESolver = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(device=device)
        self.data_fidelity = data_fidelity
        self.sde = sde
        assert (
            denoiser is not None or sde.denoiser is not None
        ), "A denoiser must be specified."
        if denoiser is not None:
            self.sde.denoiser = deepcopy(denoiser)
        if hasattr(self.data_fidelity, "denoiser"):
            self.data_fidelity.denoiser = deepcopy(denoiser)

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
                (1 + self.sde.alpha) / 2
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
        x_init: Optional[Tensor] = None,
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
        :param int seed: the random seed.
        :param torch.Tensor timesteps: the time steps for the solver. If `None`, the default time steps in the solver will be used.
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
            seed,
            y=y,
            physics=physics,
            timesteps=timesteps,
            get_trajectory=get_trajectory,
            verbose=self.verbose,
            *args,
            **kwargs,
        )
        if get_trajectory:
            return solution.sample, solution.trajectory
        else:
            return solution.sample

    def score(
        self,
        y: Tensor,
        physics: Physics,
        x: Tensor,
        t: Union[Tensor, float],
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
