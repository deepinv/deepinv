import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Tuple, Optional, List
import numpy as np
from deepinv.physics import Physics
from deepinv.sampling.sde_solver import BaseSDESolver, SDEOutput
from deepinv.models.base import Reconstructor
from deepinv.optim.data_fidelity import Zero
from deepinv.sampling.noisy_datafidelity import NoisyDataFidelity


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
        if isinstance(x_init, (Tuple, List, torch.Size)):
            x_init = self.prior_sample(x_init, rng=solver.rng)

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

    def prior_sample(
        self, shape: Union[List, Tuple, torch.Size], rng: torch.Generator = None
    ) -> Tensor:
        r"""
        Sample from the end-point distribution :math:`p_T` of the forward-SDE.
        :param shape: The shape of the the sample, of the form `(B, C, H, W)`.
        """
        raise NotImplementedError


class DiffusionSDE(BaseSDE):
    r"""
    Reverse-time Diffusion Stochastic Differential Equation defined by

    .. math::
        d\, x_{t} = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla \log p_t(x_t) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}.

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param callable alpha: a scalar weighting the diffusion term. :math:`\alpha = 0` corresponds to the ODE sampling and :math:`\alpha > 0` corresponds to the SDE sampling.
    :param callable denoiser: a denoiser used to provide an approximation of the score at time :math:`t` :math:`\nabla \log p_t`.
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
        rescale: bool = False,
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
            return np.sqrt(alpha) * forward_diffusion(t)

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

        if rescale:
            self.denoiser = (
                lambda x, sigma, *args, **kwargs: denoiser(
                    (x + 1) * 0.5, sigma / 2, *args, **kwargs
                )
                * 2.0
                - 1.0
            )
        else:
            self.denoiser = denoiser

    def score(self, x: Tensor, t: Union[Tensor, float], *args, **kwargs) -> Tensor:
        r"""
        Approximating the score function :math:`\nabla \log p_t` by the denoiser.
        :param torch.Tensor x: current state
        :param Union[torch.Tensor, float] t: current time step
        :param args: additional arguments for the `denoiser`.
        :param kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.
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
        The std of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(..., \sigma_t^2 \mathrm{Id})`.
        :param Union[torch.Tensor, float] t: current time step
        :return torch.Tensor: the noise level at time step :attr:`t`.
        """
        raise NotImplementedError


class VarianceExplodingDiffusion(DiffusionSDE):
    r"""
    `Variance-Exploding Stochastic Differential Equation (VE-SDE) <https://arxiv.org/abs/2011.13456>`_

    The forward-time SDE is defined as follows:

    .. math::
        d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}} \left( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \right)^t

    This class is the reverse-time SDE of the VE-SDE, serving as the generation process.
    """

    def __init__(
        self,
        denoiser: nn.Module = None,
        rescale: bool = False,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        alpha: float = 1.0,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):

        def forward_drift(x, t, *args, **kwargs):
            return 0.0

        def forward_diffusion(t):
            return self.sigma_t(t) * np.sqrt(
                2 * (np.log(sigma_max) - np.log(sigma_min))
            )

        super().__init__(
            forward_drift=forward_drift,
            forward_diffusion=forward_diffusion,
            alpha=alpha,
            denoiser=denoiser,
            rescale=rescale,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def prior_sample(self, shape, rng: torch.Generator) -> Tensor:
        return (
            torch.randn(shape, generator=rng, device=self.device, dtype=self.dtype)
            * self.sigma_max
        )

    def sigma_t(self, t: Union[Tensor, float]) -> Tensor:
        t = self._handle_time_step(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def score(self, x: Tensor, t: Union[Tensor, float], *args, **kwargs) -> Tensor:
        return self.sigma_t(t).view(-1, 1, 1, 1) ** (-2) * (
            self.denoiser(
                x.to(torch.float32), self.sigma_t(t).to(torch.float32), *args, **kwargs
            ).to(self.dtype)
            - x
        )


class PosteriorDiffusion(Reconstructor):
    r"""
    Posterior distribution sampling  for inverse problems using diffusion models by Reverse-time Stochastic Differential Equation (SDE).

    Consider the acquisition model:
    .. math::
        y = \noise{\forw{x}}.

    This class defines the reverse-time SDE for the posterior distribution :math:`p(x|y)` given the data :math:`y`:

    .. math::
        d\, x_t = \left( f(x_t, t) - \frac{1 + \alpha}{2} g(t)^2 \nabla_{x_t} \log p_t(x_t | y) \right) d\,t + g(t) \sqrt{\alpha} d\, w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion. The drift term and the diffusion coefficient are defined by the underlying (unconditional) forward-time SDE `unconditional_sde`. The (conditional) score function :math:`\nabla_{x_t} \log p_t(x_t | y)` can be decomposed using the Bayes' rule:

    .. math::
        \nabla_{x_t} \log p_t(x_t | y) = \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y | x_t).

    The first term is the score function of the unconditional SDE, which is typically approximated by a MMSE denoiser using the well-known Tweedie's formula, while the second term is approximated by the (noisy) data-fidelity term. We implement various data-fidelity terms in :class:`deepinv.sampling.NoisyDataFidelity`.

    :param NoisyDataFidelity data_fidelity: the noisy data-fidelity term, used to approximate the score :math:`\nabla_{x_t} \log p_t(y \vert x_t)`. Default to :class:`deepinv.optim.data_fidelity.Zero`, which corresponds to the zero data-fidelity term and the sampling process boils down to the unconditional SDE sampling.
    :param DiffusionSDE unconditional_sde: the forward-time SDE, which defines the drift and diffusion terms of the reverse-time SDE.

    :param torch.dtype dtype: the data type of the sampling solver, except for the ``denoiser`` which will use ``torch.float32``.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since most computation cost is from evaluating the ``denoiser``, which will be always computed in ``torch.float32``.
    :param torch.device device: the device for the computations.

    """

    def __init__(
        self,
        data_fidelity: NoisyDataFidelity = Zero,
        unconditional_sde: DiffusionSDE = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(device=device)
        self.data_fidelity = data_fidelity
        self.unconditional_sde = unconditional_sde
        self.dtype = dtype
        self.device = device

        def backward_drift(x, t, y, physics, *args, **kwargs):
            return -self.unconditional_sde.forward_drift(x, t) + (
                (1 + self.unconditional_sde.alpha) / 2
            ) * self.unconditional_sde.forward_diffusion(t) ** 2 * self.score(
                y, physics, x, t, *args, **kwargs
            )

        def backward_diffusion(t):
            return self.unconditional_sde.diffusion(t)

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
        solver: BaseSDESolver = None,
        seed: int = None,
        timesteps: Tensor = None,
        *args,
        **kwargs,
    ):
        r"""
        Sample the posterior distribution :math:`p(x|y)` given the data measurement :math:`y`.

        :param torch.Tensor y: the data measurement.
        :param deepinv.physics.Physics physics: the forward operator.
        :param torch.Tensor x_init: the initial value for the sampling.
        :param BaseSDESolver solver: the solver for the SDE.
        :param int seed: the random seed.
        :param torch.Tensor timesteps: the time steps for the solver.

        :return SDEOutput: a namespaced container of the output.
        """
        solver.rng_manual_seed(seed)
        if isinstance(x_init, (Tuple, List, torch.Size)):
            x_init = self.unconditional_sde.prior_sample(x_init, rng=solver.rng)

        return solver.sample(
            self.posterior,
            x_init,
            seed,
            y=y,
            physics=physics,
            timesteps=timesteps,
            *args,
            **kwargs,
        )

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
        :param Union[torch.Tensor, float] t: the current time step.
        :param args: additional arguments for the score function of the unconditional SDE.
        :param kwargs: additional keyword arguments for the score function of the unconditional SDE.

        :return: the score function :math:`\nabla_{x_t} \log p_t(x_t \vert y)`.
        :rtype: torch.Tensor
        """
        sigma = self.unconditional_sde.sigma_t(t).to(torch.float32)

        if isinstance(self.data_fidelity, Zero):
            return self.unconditional_sde.score(x, t, *args, **kwargs).to(self.dtype)
        else:
            return self.unconditional_sde.score(
                x, t, *args, **kwargs
            ) - self.data_fidelity.grad(
                x.to(torch.float32), y.to(torch.float32), physics, sigma
            ).to(
                self.dtype
            )


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.models import NCSNpp, EDMPrecond
    from deepinv.sampling.diffusion_sde import VarianceExplodingDiffusion
    from deepinv.sampling.sde_solver import HeunSolver
    from deepinv.sampling.noisy_datafidelity import DPSDataFidelity

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
    denoiser = EDMPrecond(model=unet).to(device)
    sigma_min = 0.02
    sigma_max = 10
    num_steps = 10

    sde = VarianceExplodingDiffusion(
        denoiser=denoiser,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        device=device,
        dtype=dtype,
        alpha=0.5,
    )

    rng = torch.Generator(device).manual_seed(42)
    timesteps = np.linspace(0.001, 1, num_steps)[::-1]
    solver = HeunSolver(timesteps=timesteps, full_trajectory=True, rng=rng)
    solution = sde.sample((1, 3, 64, 64), solver=solver, seed=1)
    x = solution.sample.clone()
    dinv.utils.plot(x, titles="Original sample", show=True)

    posterior = PosteriorDiffusion(
        data_fidelity=Zero(),
        unconditional_sde=sde,
        dtype=dtype,
        device=device,
    )

    posterior_sample = posterior.forward(
        None, None, solver=solver, x_init=(1, 3, 64, 64), seed=1, timesteps=timesteps
    )
    dinv.utils.plot([x, posterior_sample.sample], show=True)

    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoiser),
        unconditional_sde=sde,
        dtype=dtype,
        device=device,
    )

    physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.5, device=device)
    y = physics(x)

    posterior_sample = posterior.forward(
        y, physics, solver=solver, x_init=(1, 3, 64, 64), seed=1, timesteps=timesteps
    )
    dinv.utils.plot([x, y, posterior_sample.sample], show=True)
    dinv.utils.plot_videos(posterior_sample.trajectory, display=True, time_dim=0)
