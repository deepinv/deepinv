# %%
import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Tuple, List, Any
import numpy as np
from numpy import ndarray
import warnings
from sde_solver import select_solver, SDEOutput


class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):
    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient and :math:`w` is the standard Brownian motion.

    It defines the common interface for drift and diffusion functions.

    :param callable drift: a time-dependent drift function f(x, t)
    :param callable diffusion: a time-dependent diffusion function g(t)
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    :param torch.dtype dtype: the data type of the computations.
    :param str device: the device for the computations.
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        rng: torch.Generator = None,
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
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def sample(
        self,
        x_init: Tensor = None,
        *args,
        timesteps: Union[Tensor, ndarray] = None,
        method: str = "euler",
        seed: int = None,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the SDE with the given timesteps.
        :param torch.Tensor x_init: initial value.
        :param timesteps: time steps at which to discretize the SDE, of shape `(n_steps,)`.
        :param str method: method for solving the SDE. One of the methods available in :meth:`deepinv.sampling.sde_solver`.

        :rtype SDEOutput.
        """
        self.rng_manual_seed(seed)
        solver_fn = select_solver(method)
        solver = solver_fn(sde=self, rng=self.rng)
        solution = solver.sample(x_init, timesteps=timesteps, *args, **kwargs)
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
        r""",
        Reset the random number generator to its initial state.
        """
        self.rng.set_state(self.initial_random_state)

    def randn_like(self, input: torch.Tensor, seed: int = None):
        r"""
        Equivalent to `torch.randn_like` but supports a pseudorandom number generator argument.
        :param int seed: the seed for the random number generator, if `rng` is provided.

        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


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


    The score function can be computed using Tweedie's formula, given a MMSE denoiser :math:`D`:

    .. math::
            \nabla \log p_{t}(x) = \left(D(x,\sigma(t)) - x \right) / \sigma(t)^2

    where :math:`sigma(t)` is the noise level at time :math:`t`, which can be accessed through the attribute :meth:`sigma_t`

    Default parameters correspond to the `Ornstein-Uhlenbeck process <https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process>`_ SDE, defined in the time interval `[0,1]`.

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param callable denoiser: a pre-trained MMSE denoiser which will be used to approximate the score function by Tweedie's formula.
    :param bool use_backward_ode: a boolean indicating whether to use the deterministic probability flow ODE for the backward process.
    :param torch.Generator rng: pseudo-random number generator for reproducibility.
    :param torch.dtype dtype: data type of the computation, except for the `denoiser` which will be always compute in `float32`.
        We recommend using `torch.float64` for better stability and less numerical error when solving the SDE in discrete time, since
        most computation cost is from evaluating the `denoiser`, which will be always computed in `float32`.
    :param torch.device device: device on which the computation is performed.
    """

    def __init__(
        self,
        drift: Callable = lambda x, t: -x,  # Default to Ornstein-Uhlenbeck process
        diffusion: Callable = lambda t: math.sqrt(2.0),
        use_backward_ode=False,
        denoiser: nn.Module = None,
        rng: torch.Generator = None,
        dtype=torch.float64,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.rng = rng
        self.device = device
        self.dtype = dtype
        self.use_backward_ode = use_backward_ode
        forward_drift = lambda x, t, *args, **kwargs: drift(x, t, *args, **kwargs)
        forward_diff = lambda t: diffusion(t)
        self.forward_sde = BaseSDE(
            drift=forward_drift,
            diffusion=forward_diff,
            rng=rng,
            dtype=dtype,
            device=device,
        )

        if self.use_backward_ode:
            backward_drift = lambda x, t, *args, **kwargs: -drift(x, t) + 0.5 * (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
            backward_diff = lambda t: 0.0
        else:
            backward_drift = lambda x, t, *args, **kwargs: -drift(x, t) + (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
            backward_diff = lambda t: diffusion(t)
        self.backward_sde = BaseSDE(
            drift=backward_drift,
            diffusion=backward_diff,
            rng=rng,
            dtype=dtype,
            device=device,
        )

    @torch.no_grad()
    def forward(
        self,
        x_init: Union[Tensor, Tuple],
        timesteps: Union[Tensor, List, ndarray],
        method: str = "Euler",
        *args,
        **kwargs,
    ):
        r"""
        Sample the backward-SDE.

        :param Union[Tensor, Tuple] x_init: Initial condition of the backward-SDE.
            If it is a tensor, `x_init` should follow the distribution of :math:`p_T`, which is usually   :math:`\mathcal{N}(0, \sigma_{\mathrm{max}}^2 \mathrm{Id}})`.
            If it is a tuple, it should be of the form `(B, C, H, W)`. A sample from the distribution :math:`p_T` will be generated automatically.

        :param torch.Tensor timesteps: The time steps at which to discretize the backward-SDE, should be of shape `(n_steps,)`.
        :param str method: The method to discretize the backward-SDE, can be one of the methods available in :meth:`deepinv.sampling.sde_solver`.
        :param args: additional arguments for the backward drift (passed to the `denoiser`).
        :param kwargs: additional keyword arguments for the backward drift (passed to the `denoiser`), e.g., `class_labels` for class-conditional models.

        :rtype: SDESolution.

        """
        if isinstance(x_init, (Tuple, List, torch.Size)):
            x_init = self.prior_sample(x_init)
        return self.backward_sde.sample(
            x_init, timesteps=timesteps, method=method, *args, **kwargs
        )

    def prior_sample(self, shape: Union[List, Tuple, torch.Size]) -> Tensor:
        r"""
        Sample from the end-point distribution :math:`p_T` of the forward-SDE.

        :param shape: The shape of the the sample, of the form `(B, C, H, W)`.
        """
        return torch.randn(
            shape, generator=self.rng, device=self.device, dtype=self.dtype
        ) * self.sigma_t(1.0).view(-1, 1, 1, 1)

    def score(self, x: Tensor, t: Any, *args, **kwargs) -> Tensor:
        r"""
        Approximating the score function :math:`\nabla \log p_t` by the denoiser.

        :param torch.Tensor x: current state
        :param Any t: current time step
        :param args: additional arguments for the `denoiser`.
        :param kwargs: additional keyword arguments for the `denoiser`, e.g., `class_labels` for class-conditional models.

        :return: the score function :math:`nabla \log p_t(x)`.
        :rtype: torch.Tensor
        """
        t = self._handle_time_step(t)
        exp_minus_t = torch.exp(-t) if isinstance(t, Tensor) else np.exp(-t)
        return self.sigma_t(t).view(-1, 1, 1, 1) ** (-2) * (
            exp_minus_t
            * self.denoiser(x.to(torch.float32), self.sigma_t(t), *args, **kwargs).to(
                self.dtype
            )
            - x
        )

    def _handle_time_step(self, t):
        t = torch.as_tensor(t, device=self.device, dtype=self.dtype)
        return t

    def sigma_t(self, t: Any) -> Tensor:
        r"""
        The std of the condition distribution :math:`p(x_t \vert x_0) \sim \mathcal{N}(..., \sigma_t^2 \mathrm{\Id})`.

        :param Any t: time step.
        """
        t = self._handle_time_step(t)
        return torch.sqrt(1.0 - torch.exp(-2.0 * t))


class VESDE(DiffusionSDE):
    r"""
    Variance-Exploding Stochastic Differential Equation (VE-SDE), described in the paper: https://arxiv.org/abs/2011.13456

    The forward-time SDE is defined as follows:

    .. math::
        d\, x_t = \sigma(t) d\, w_t \quad \mbox{where } \sigma(t) = \sigma_{\mathrm{min}} \( \frac{\sigma_{\mathrm{max}}}{\sigma_{\mathrm{min}}} \)^t

    """

    def __init__(
        self,
        denoiser: nn.Module,
        sigma_min: float = 0.02,
        sigma_max: float = 100,
        use_backward_ode: bool = False,
        rng: torch.Generator = None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        forward_drift = lambda x, t, *args, **kwargs: 0.0
        forward_diff = lambda t: self.sigma_t(t) * np.sqrt(
            2 * (np.log(sigma_max) - np.log(sigma_min))
        )
        super().__init__(
            drift=forward_drift,
            diffusion=forward_diff,
            use_backward_ode=use_backward_ode,
            denoiser=denoiser,
            rng=rng,
            dtype=dtype,
            device=device,
            *args,
            *kwargs,
        )

    def prior_sample(self, shape):
        return (
            torch.randn(shape, generator=self.rng, device=self.device, dtype=self.dtype)
            * self.sigma_max
        )

    def sigma_t(self, t):
        t = self._handle_time_step(t)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def score(self, x, t, *args, **kwargs):
        t = self._handle_time_step(t)
        return self.sigma_t(t).view(-1, 1, 1, 1) ** (-2) * (
            self.denoiser(x.to(torch.float32), self.sigma_t(t), *args, **kwargs).to(
                self.dtype
            )
            - x
        )


# %%
if __name__ == "__main__":
    # %%
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv
    import matplotlib.pyplot as plt
    from deepinv.models.edm import NCSNpp, ADMUNet, EDMPrecond

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unet = dinv.models.DRUNet(pretrained="download").to(device)
    # unet = NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")
    unet = ADMUNet.from_pretrained("imagenet64-cond")
    denoiser = EDMPrecond(model=unet).to(device)


    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    t = 100.0
    x_noisy = x + torch.randn_like(x) * t
    class_labels = torch.eye(1000, device=device)[0:1]
    x_denoised = denoiser(x_noisy, t, class_labels=class_labels)
    dinv.utils.plot([x, x_noisy, x_denoised], titles=["sample", "y", "denoised"])
    _ = plt.hist(x_denoised.detach().cpu().numpy().ravel(), bins=100)
    plt.show()
    # %%
    x_noisy = (x_noisy + 1) * 0.5
    x_denoised = denoiser(x_noisy, t * 0.5, class_labels=class_labels)
    x_denoised = x_denoised * 2 - 1
    dinv.utils.plot([x, x_noisy, x_denoised], titles=["sample", "y", "denoised"])
    _ = plt.hist(x_denoised.detach().cpu().numpy().ravel(), bins=100)
    plt.show()
    # %%
    # VESDE
    sigma_min = 0.02
    sigma_max = 10
    timesteps = np.linspace(0.001, 1.0, 200)
    sigma_t = lambda t: sigma_min * (sigma_max / sigma_min) ** t
    rng = torch.Generator(device).manual_seed(42)

    # prior = dinv.optim.prior.DiffusionScorePrior(
    #     denoiser=denoiser, sigma_t=sigma_t, rescale=False
    # )

    sde = VESDE(
        denoiser=denoiser,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        rng=rng,
        device=device,
        use_backward_ode=False,
    )

    # sde = DiffusionSDE(denoiser=denoiser, rng=rng, device=device)

    # Check forward
    x_init = x.clone()

    solution = sde.forward_sde.sample(
        x_init, timesteps=timesteps, method="euler", class_labels=class_labels
    )
    dinv.utils.plot(solution.sample, suptitle=f"Forward sample, nfe = {solution.nfe}")
    _ = plt.hist(solution.sample.ravel().cpu().numpy(), bins=100)
    plt.show()
    # %% Check backward
    # x_init = torch.randn((1, 3, 64, 64), device=device, generator=rng) * sigma_max

    solution = sde(
        (1, 3, 64, 64),
        timesteps=timesteps[::-1],
        method="Euler",
        seed=1,
        class_labels=class_labels,
    )
    dinv.utils.plot(solution.sample, suptitle=f"Backward sample, nfe = {solution.nfe}")
    _ = plt.hist(solution.sample.ravel().cpu().numpy(), bins=100)
    plt.show()

# %%
