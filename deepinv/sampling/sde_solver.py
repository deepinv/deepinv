from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Any
from numpy import ndarray
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepinv.sampling.diffusion_sde import (
        BaseSDE,
        EDMDiffusionSDE,
        PosteriorDiffusion,
    )


class SDEOutput(dict):
    r"""
    A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.

    Attributes:
    :attr torch.Tensor sample: the final samples of the sampling process, of shape ``(B, C, H, W)``.
    :attr torch.Tensor trajectory: the trajectory of the sampling process, of shape ``(num_steps, B, C, H, W)`` if ``full_trajectory`` is ``True``, otherwise of shape ``(B, C, H, W)``.
    :attr torch.Tensor timesteps: the time steps at which the samples were taken, of shape ``(num_steps,)``.
    :attr int nfe: the number of function evaluations performed during the integration.
    """

    def __init__(self, sample: Tensor, trajectory: Tensor, timesteps: Tensor, nfe: int):
        sol = {
            "sample": sample,
            "trajectory": trajectory,
            "timesteps": timesteps,
            "nfe": nfe,
        }
        super().__init__(sol)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class BaseSDESolver(nn.Module):
    r"""
    Base class for solving Stochastic Differential Equations (SDEs) from :class:`deepinv.sampling.BaseSDE` of the form:

    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient, and :math:`w_t` is a standard Brownian process.

    Currently only supported for fixed time steps for numerical integration.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.


    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator | None = None,
    ):
        super().__init__()
        if timesteps is None:
            if t_start is None or t_end is None or num_steps is None:
                raise ValueError(
                    "If timesteps is not provided, t_start, t_end, and num_steps must be specified."
                )
            timesteps = torch.linspace(t_start, t_end, num_steps)
        if isinstance(timesteps, ndarray):
            self.timesteps = torch.from_numpy(timesteps.copy())
        elif isinstance(timesteps, Tensor):
            self.timesteps = timesteps
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()
            self.timesteps = self.timesteps.to(rng.device)

    def step(
        self,
        sde: BaseSDE,
        t0: float,
        t1: float,
        x0: Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        r"""
        Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).

        :return torch.Tensor, int: Updated state of the system after the step and number of function evaluations (NFE) performed during the step.
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        sde: BaseSDE,
        x_init: Tensor,
        seed: int = None,
        *args,
        timesteps: Tensor | ndarray = None,
        get_trajectory: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the Stochastic Differential Equation (SDE) with given time steps.

        This function iteratively applies the SDE solver step for each time interval
        defined by the provided timesteps.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param torch.Tensor x_init: The initial state of the system.
        :param int seed: The seed for the random number generator, if `rng` is provided.
        :param torch.Tensor, numpy.ndarray, list timesteps: A sequence of time points at which to solve the SDE. If None, default timesteps will be used.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to False.
        :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.
        :param \*args: Variable length argument list to be passed to the step function.
        :param \*\*kwargs: Arbitrary keyword arguments to be passed to the step function.

        :return: SDEOutput
        """
        self.rng_manual_seed(seed)
        x = x_init
        nfe = 0
        trajectory = [x_init.clone()] if get_trajectory else []

        if timesteps is None:
            timesteps = self.timesteps.to(sde.device, sde.dtype)
        else:
            if isinstance(timesteps, ndarray):
                timesteps = torch.from_numpy(timesteps.copy())
            timesteps = timesteps.to(sde.device, sde.dtype)

        for t_cur, t_next in tqdm(
            zip(timesteps[:-1], timesteps[1:], strict=True),
            total=len(timesteps) - 1,
            disable=not verbose,
        ):
            x, cur_nfe = self.step(sde, t_cur, t_next, x, *args, **kwargs)
            nfe += cur_nfe
            if get_trajectory:
                trajectory.append(x.clone())
        if get_trajectory:
            trajectory = torch.stack(trajectory, dim=0)
        else:
            trajectory = x
        output = SDEOutput(
            sample=x, trajectory=trajectory, timesteps=timesteps, nfe=nfe
        )

        return output

    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        :param int seed: the seed to set for the random number generator. If not provided, the current state of the random number generator is used.
            Note: it will be ignored if the random number generator is not initialized.
        """
        if seed is not None:
            if self.rng is not None:
                self.rng = self.rng.manual_seed(seed)
            else:
                warnings.warn(
                    "Cannot set seed for random number generator because it is not initialized. The `seed` parameter is ignored."
                )

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        self.rng.set_state(self.initial_random_state)

    def randn_like(self, input: torch.Tensor, seed: int = None) -> torch.Tensor:
        r"""
        Equivalent to :func:`torch.randn_like` but supports a pseudorandom number generator argument.

        :param torch.Tensor input: The input tensor whose size will be used.
        :param int seed: The seed for the random number generator, if `rng` is provided.

        :return: A tensor of the same size as input filled with random numbers from a normal distribution.
        :rtype: torch.Tensor

        This method uses the `rng` attribute of the class, which is a pseudo-random number generator
        for reproducibility. If a seed is provided, it will be used to set the state of `rng` before
        generating the random numbers.

        .. note::
           The `rng` attribute must be initialized for this method to work properly.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class EulerSolver(BaseSDESolver):
    r"""
    Euler-Maruyama solver for SDEs.

    This solver uses the Euler-Maruyama method to numerically integrate SDEs. It is a first-order method that
    approximates the solution using the following update rule:

    .. math::

        x_{t+dt} = x_t + f(x_t,t)dt + g(t) W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.

    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)

    def step(
        self, sde: BaseSDE, t0: float, t1: float, x0: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, int]:
        r"""
        Perform a single Euler-Maruyama step from time `t0` to time `t1`, with current state `x0`.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).
        :param \*args: additional arguments for the drift of the SDE.
        :param \*\*kwargs: additional keyword arguments for the drift of the SDE.

        :return torch.Tensor, int: Updated state of the system after the step and number of function evaluations (NFE) performed during the step (here 1).
        """
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift, diffusion = sde.discretize(x0, t0, *args, **kwargs)
        return x0 + drift * dt + diffusion * dW, 1


class HeunSolver(BaseSDESolver):
    r"""
    Heun solver for SDEs.

    This solver uses the second-order Heun method to numerically integrate SDEs, defined as:

    .. math::
        \tilde{x}_{t+dt} &= x_t + f(x_t,t)dt + g(t) W_{dt} \\
        x_{t+dt} &= x_t + \frac{1}{2}[f(x_t,t) + f(\tilde{x}_{t+dt},t+dt)]dt + \frac{1}{2}[g(t) + g(t+dt)] W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param torch.Generator rng: A random number generator for reproducibility.
    
    .. note::
    
        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.
    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)

    def step(
        self,
        sde: BaseSDE,
        t0: float,
        t1: float,
        x0: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        r"""
        Perform a single Heun step from time `t0` to time `t1`, with current state `x0`:
        an Euler-Maruyama prediction, corrected with the drift and diffusion evaluated at the prediction.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).
        :param \*args: additional arguments for the drift of the SDE.
        :param \*\*kwargs: additional keyword arguments for the drift of the SDE.

        :return torch.Tensor, int: Updated state of the system after the step and number of function evaluations (NFE) performed during the step (here 2).
        """
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift_0, diffusion_0 = sde.discretize(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_0 * dt + diffusion_0 * dW
        drift_1, diffusion_1 = sde.discretize(x_euler, t1, *args, **kwargs)

        return (
            x0
            + 0.5 * (drift_0 + drift_1) * dt
            + 0.5 * (diffusion_0 + diffusion_1) * dW,
            2,
        )


class AncestralSolver(BaseSDESolver):
    r"""
    Ancestral solver for reverse-time diffusion SDEs, generalizing the DDPM and DDIM samplers.

    It solves the reverse-time SDE (see :class:`deepinv.sampling.EDMDiffusionSDE`), from :math:`t = T` to :math:`t = 0`:

    .. math::
        d x_t = \left(\frac{s'(t)}{s(t)} x_t - (1 + \alpha(t)) s(t)^2 \sigma(t) \sigma'(t) \nabla \log p_t(x_t) \right) dt + s(t) \sqrt{2 \alpha(t) \sigma(t) \sigma'(t)} d w_t.

    On a step from :math:`t` to :math:`t + dt` (with :math:`dt < 0` for reverse-time sampling), the solver computes the next state :math:`x_{t+dt}` as:

    .. math::
        x_{t+dt} = \frac{s(t+dt)}{s(t)} x_t + s(t) s(t+dt) \sigma(t)^2 \left(1 - r^{1 + \alpha}\right) \nabla \log p_t(x_t)
        + s(t+dt) \sigma(t+dt) \sqrt{1 - r^{2 \alpha}} \, z, \quad z \sim \mathcal{N}(0, \mathrm{Id}),

    with :math:`r = \sigma(t+dt) / \sigma(t)` and :math:`\alpha = \alpha(t)`. The noise level of the next state is exactly :math:`\sigma(t+dt)`.

    Compared to a Euler-Maruyama step of :class:`deepinv.sampling.EulerSolver`, it integrates the linear part
    and the noise exactly, and freezes the non-linear term. It is thus more accurate when discretizing with few steps.

    The ancestral DDPM sampler :footcite:p:`ho2020denoising` is obtained for :math:`\alpha = 1`, and the deterministic DDIM sampler
    :footcite:p:`song2020denoising` for :math:`\alpha = 0`. See :class:`deepinv.sampling.DDPMSolver` and :class:`deepinv.sampling.DDIMSolver`.

    If the parameter :math:`\eta` of DDIM :footcite:p:`song2020denoising` is given, the `alpha` of the SDE is ignored, and replaced on each step by

    .. math::
        \alpha_\eta = \frac{\log\left(1 - \eta^2 (1 - r^2)\right)}{2 \log r},

    for which the step is exactly the DDIM step with parameter :math:`\eta`.

    With `variance="large"`, the noise :math:`s(t+dt) \sigma(t+dt) \sqrt{1 - r^{2 \alpha}}` is replaced by :math:`s(t) \sigma(t) \sqrt{1 - r^{2 \alpha}}`.
    For :math:`\alpha = 1`, this replaces the posterior variance :math:`\tilde{\beta}_t` of DDPM by the variance :math:`\beta_t` of the forward transition,
    see Section 3.2 of :footcite:t:`ho2020denoising` and :class:`deepinv.sampling.DDPMSolver`.

    .. note::

        The solver requires `sde.sigma_t`, `sde.scale_t`, `sde.score` and, if `eta` is `None`, `sde.alpha`,
        provided by :class:`deepinv.sampling.EDMDiffusionSDE` (and its subclasses) and by :class:`deepinv.sampling.PosteriorDiffusion`.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float eta: the stochasticity parameter :math:`\eta \in [0, 1]` of DDIM, optional. If given, it replaces the `alpha` of the SDE by :math:`\alpha_\eta`. If `None` (default), the `alpha` of the SDE is used.
    :param str variance: the variance of the noise added on each step, either `"small"` (default) for the posterior variance (:math:`\tilde{\beta}_t` for DDPM), or `"large"` for the variance of the forward transition (:math:`\beta_t` for DDPM).
    :param torch.Generator rng: A random number generator for reproducibility.

    .. note::

        You can either provide the `timesteps` argument directly, or specify `t_start`, `t_end`, and `num_steps` to generate the time steps automatically (linearly with constant stepsize). If both are provided, the `timesteps` argument will take precedence.

    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        eta: float | None = None,
        variance: str = "small",
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, t_start, t_end, num_steps, rng=rng)
        if variance.lower() not in ("small", "large"):
            raise ValueError(
                f"variance must be either 'small' or 'large', got {variance}"
            )
        self.eta = eta
        self.variance = variance.lower()

    def step(
        self,
        sde: EDMDiffusionSDE | PosteriorDiffusion,
        t0: float,
        t1: float,
        x0: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        r"""
        Perform a single ancestral step from time `t0` to time `t1`, with current state `x0`, solving the reverse-time SDE.

        :param deepinv.sampling.EDMDiffusionSDE, deepinv.sampling.PosteriorDiffusion sde: the SDE to solve, which must provide `sigma_t`, `scale_t`, `score` and, if `eta` is `None`, `alpha`.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).
        :param \*args: additional arguments for the score of the SDE.
        :param \*\*kwargs: additional keyword arguments for the score of the SDE.

        :return torch.Tensor, int: Updated state of the system after the step and number of function evaluations (NFE) performed during the step (here 1).
        """
        scale_0, sigma_0 = sde.scale_t(t0), sde.sigma_t(t0)
        scale_1, sigma_1 = sde.scale_t(t1), sde.sigma_t(t1)
        score = sde.score(x0, t0, *args, **kwargs)
        ratio = sigma_1 / sigma_0
        if self.eta is None:
            ratio_2alpha = ratio ** (2 * sde.alpha(t0))
        else:
            # r^(2 alpha_eta), computed from eta
            ratio_2alpha = 1 - self.eta**2 * (1 - ratio**2)
        coef = ratio * ratio_2alpha.clamp(min=0).sqrt()  # r^(1 + alpha)

        # posterior variance
        noise_std = sigma_1 * (1 - ratio_2alpha).clamp(min=0).sqrt()

        # variance of the forward transition instead of the posterior variance
        if self.variance == "large" and sigma_1 > 0:
            noise_std = noise_std * (scale_0 * sigma_0) / (scale_1 * sigma_1)

        x1 = (scale_1 / scale_0) * x0 + scale_0 * scale_1 * sigma_0**2 * (
            1 - coef
        ) * score
        if noise_std > 0:
            x1 = x1 + scale_1 * noise_std * self.randn_like(x0)
        return x1, 1


class DDPMSolver(AncestralSolver):
    r"""
    DDPM solver for reverse-time diffusion SDEs.

    Ancestral sampler of DDPM :footcite:p:`ho2020denoising`, i.e. :class:`deepinv.sampling.AncestralSolver` with :math:`\eta = 1` and the stochastic term `alpha` of the SDE is ignored (fixed to 1).

    For :class:`deepinv.sampling.VariancePreservingDiffusion` with time steps matching the training time steps of a discrete DDPM model,
    this is exactly the DDPM sampler with the posterior variance :math:`\tilde{\beta}_t`.
    With `variance="large"`, the posterior variance :math:`\tilde{\beta}_t` is replaced by the variance :math:`\beta_t` of the forward transition,
    see Section 3.2 of :footcite:t:`ho2020denoising` and :class:`deepinv.sampling.AncestralSolver`.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param str variance: the variance of the noise added on each step, either `"small"` (default) for the posterior variance :math:`\tilde{\beta}_t`, or `"large"` for the variance :math:`\beta_t` of the forward transition.
    :param torch.Generator rng: A random number generator for reproducibility.

    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        variance: str = "small",
        rng: torch.Generator = None,
    ):
        super().__init__(
            timesteps, t_start, t_end, num_steps, eta=1.0, variance=variance, rng=rng
        )


class DDIMSolver(AncestralSolver):
    r"""
    DDIM solver for reverse-time diffusion SDEs.

    Sampler of DDIM :footcite:p:`song2020denoising` with the stochasticity parameter :math:`\eta`.
    The default :math:`\eta = 0` gives the deterministic DDIM sampler and :math:`\eta = 1` gives the DDPM sampler,
    see :class:`deepinv.sampling.DDPMSolver`.

    The `alpha` of the SDE is ignored, and replaced on each step of :class:`deepinv.sampling.AncestralSolver` by

    .. math::
        \alpha_\eta = \frac{\log\left(1 - \eta^2 (1 - r^2)\right)}{2 \log r}, \quad r = \frac{\sigma(t+dt)}{\sigma(t)}.

    This relation depends on the step: :math:`\alpha_\eta = \eta` for :math:`\eta \in \{0, 1\}`, and :math:`\alpha_\eta` tends to :math:`\eta^2` for small steps.

    With `variance="large"`, the noise is scaled by :math:`s(t) \sigma(t) / (s(t+dt) \sigma(t+dt))`, see :class:`deepinv.sampling.AncestralSolver`.
    For :math:`\eta = 1`, this is the DDPM sampler with the variance :math:`\beta_t` of the forward transition.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.
    :param float t_start: the starting time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float t_end: the ending time of the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param int num_steps: the number of time steps for the SDE, optional. If not provided, it will be inferred from the `timesteps` argument.
    :param float eta: the stochasticity parameter :math:`\eta \in [0, 1]` of DDIM. Default to `0`.
    :param str variance: the variance of the noise added on each step, either `"small"` (default) or `"large"`, see :class:`deepinv.sampling.AncestralSolver`.
    :param torch.Generator rng: A random number generator for reproducibility.

    """

    def __init__(
        self,
        timesteps: Tensor | ndarray = None,
        t_start: float | None = None,
        t_end: float | None = None,
        num_steps: int | None = None,
        eta: float = 0.0,
        variance: str = "small",
        rng: torch.Generator = None,
    ):
        super().__init__(
            timesteps, t_start, t_end, num_steps, eta=eta, variance=variance, rng=rng
        )
