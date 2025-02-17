import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Tuple, Optional, List
import numpy as np
from .sde_solver import BaseSDESolver, SDEOutput
from deepinv.models.base import Reconstructor
from deepinv.optim.data_fidelity import Zero


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

    def _handle_time_step(self, t) -> Tensor:
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


class PosteriorDiffusion(Reconstructor):
    r"""
    Posterior distribution sampling using diffusion models.
    """

    def __init__(
        self,
        data_fidelity: Zero,
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
        y,
        physics,
        x_init: Optional[Tensor] = None,
        solver: BaseSDESolver = None,
        seed: int = None,
        timesteps: Tensor = None,
        *args,
        **kwargs,
    ):
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

    def score(self, y, physics, x, t, *args, **kwargs):
        sigma = self.unconditional_sde.sigma_t(t).to(torch.float32)
        return self.unconditional_sde.score(
            x, t, *args, **kwargs
        ) - self.data_fidelity.grad(
            x.to(torch.float32), y.to(torch.float32), physics, sigma
        ).to(
            self.dtype
        )


####################
from deepinv.optim.data_fidelity import L2

# This file implements the p(y|x) terms as proposed in the `review paper <https://arxiv.org/pdf/2410.00083>`_ by Daras et al.


class NoisyDataFidelity(L2):
    r"""
    Preconditioned data fidelity term for noisy data :math:`\datafid{x}{y}=\distance{\forw{x'}}{y'}`.

    This is a base class for the conditional classes for approximating :math:`\log p(y|x)` used in diffusion
    algorithms for inverse problems. Here, :math:`x'` and :math:`y'` are perturbed versions of :math:`x` and :math:`y`
    and the associated data fidelity term is :math:`\datafid{x}{y}=\distance{\forw{x'}}{y'}`.

    It comes with a `.grad` method computing the score

    .. math::

        \begin{equation*}
            \nabla_x \log p(y|x) = P(\forw{x'}-y'),
        \end{equation*}


    where :math:`P` is a preconditioner. By default, :math:`P` is defined as :math:`A^\top` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.
    """

    def __init__(self):
        super(NoisyDataFidelity, self).__init__()

    def precond(self, u: torch.Tensor, physics) -> torch.Tensor:
        r"""
        The preconditioner :math:`P = A^\top` for the data fidelity term.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.FloatTensor) preconditionned tensor :math:`P(u)`.
        """
        return physics.A_adjoint(u)

    def diff(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param physics: physics model
        :param float sigma: Standard deviation of the noise.
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics, sigma))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y)


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    The DPS data-fidelity term.

    This corresponds to the :math:`p(y|x)` prior as proposed in `Diffusion Probabilistic Models <https://arxiv.org/abs/2209.14687>`_.

    :param denoiser: Denoiser network. # TODO: type?
    """

    def __init__(self, denoiser=None):
        super(DPSDataFidelity, self).__init__()

        self.denoiser = denoiser

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        As explained in `Daras et al. <https://arxiv.org/abs/2410.00083>`_, the score is defined as

        .. math::

                \nabla_x \log p(y|x) = \left(\operatorname{Id}+\nabla_x D(x)\right)^\top A^\top \left(y-\forw{D(x)}\right)

        .. note::
            The preconditioning term is computed with autodiff.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param physics: physics model
        :param float sigma: Standard deviation of the noise. (unused)
        :return: (torch.Tensor) score term.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            l2_loss = self.forward(x, y, physics, sigma, *args, **kwargs)

        norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, clip=False
    ) -> torch.Tensor:
        r"""
        Returns the loss term :math:`\distance{\forw{D(x)}}{y}`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :param bool clip: whether to clip the output of the denoiser to the range [-1, 1].
        :return: (torch.Tensor) loss term.
        """
        # TODO: check that the following does not belong to the grad method but to the denoiser method itself.
        # aux_x = x / 2 + 0.5
        # x0_t = 2 * self.denoiser(x, sigma / 2) - 1

        x0_t = self.denoiser(x, sigma)

        if clip:
            x0_t = torch.clip(x0_t, 0.0, 1.0)  # optional

        l2_loss = self.d(physics.A(x0_t), y)

        return l2_loss


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.models import NCSNpp, EDMPrecond
    from deepinv.sampling.sde import VarianceExplodingSDE
    from deepinv.sampling.sde_solver import HeunSolver

    device = "cuda"
    dtype = torch.float64

    unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
    denoiser = EDMPrecond(model=unet).to(device)
    sigma_min = 0.02
    sigma_max = 10
    num_steps = 100

    sde = VarianceExplodingSDE(
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
