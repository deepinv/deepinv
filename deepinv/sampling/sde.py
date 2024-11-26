# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional, Tuple, List
import numpy as np
from numpy import ndarray
import warnings
from utils import get_edm_parameters
from deepinv.optim.prior import ScorePrior
from deepinv.physics import Physics
from sde_solver import select_solver


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
    ):
        r"""
        Solve the SDE with the given time-step.
        :param torch.Tensor x_init: initial value.
        :param timesteps: time steps at which to sample the SDE.
        :param str method: method for solving the SDE.

        :return torch.Tensor: samples from the SDE.
        """
        self.rng_manual_seed(seed)
        solver_fn = select_solver(method)
        solver = solver_fn(sde=self, rng=self.rng, **kwargs)

        samples = solver.sample(x_init, timesteps=timesteps, *args, **kwargs)
        return samples

    def discretize(
        self, x: Tensor, t: Union[Tensor, float], *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
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
            d x_{t} = f(x_t, t) dt + g(t) d w_{t}.

    This forward SDE can be reversed by the following SDE running backward in time:

    .. math::
            d x_{t} = \left( f(x_t, t) - g(t)^2 \nabla \log p_t(x_t) \right) dt + g(t) d w_{t}.

    There also exists a deterministic probability flow ODE whose trajectories share the same marginal distribution as the SDEs:

    .. math::
            d x_{t} = \left( f(x_t, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x_t) \right) dt.

    Both forward and backward SDE are solved in forward time.

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param deepinv.prior.ScorePrior prior: a time-dependent score prior, corresponding to :math:`\nabla \log p_t`
    :param bool use_backward_ode: a boolean indicating whether to use the deterministic probability flow ODE for the backward process.
    :param torch.Generator rng: pseudo-random number generator for reproducibility.
    """

    def __init__(
        self,
        drift: Callable = lambda x, t: -x,  # Default to Ornstein-Uhlenbeck process
        diffusion: Callable = lambda t: math.sqrt(2.0),
        use_backward_ode=False,
        prior: ScorePrior = None,
        rng: torch.Generator = None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__()
        self.prior = prior
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
            ) * self.prior.score(x, t, *args, **kwargs)
            backward_diff = lambda t: 0.0
        else:
            backward_drift = lambda x, t, *args, **kwargs: -drift(x, t) + (
                diffusion(t) ** 2
            ) * self.prior.score(x, t, *args, **kwargs)
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
        x_init: Tensor,
        timesteps: Tensor,
        method: str = "Euler",
        *args,
        **kwargs,
    ):
        r"""
        Sample the backward-SDE.

        :param torch.Tensor x_init: Initial condition of the backward-SDE, x_init should follow the distribution of :math:`p_T`, which is usually :math:`\mathcal{N}(0, \sigma_{\mathrm{max}}^2 \mathrm{Id}})`.
        :param torch.Tensor timesteps: The time steps at which to discretize the backward-SDE, should be in ascending order.
        :param str method: The method to discretize the backward-SDE, can be one of the methods available in :meth:`deepinv.sampling.sde_solver`.

        :return: SDESolution.

        """
        if isinstance(x_init, (Tuple, List, torch.Size)):
            x_init = self.prior_sample(x_init)
        return self.backward_sde.sample(
            x_init, timesteps=timesteps, method=method, *args, **kwargs
        )

    def prior_sample(self, shape):
        r"""
        Sample from the end-point distribution :math:`p_T` of the forward-SDE.
        """
        raise NotImplementedError


class VESDE(DiffusionSDE):
    def __init__(
        self,
        prior: ScorePrior,
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
        self.sigma_t = lambda t: sigma_min * (sigma_max / sigma_min) ** t
        forward_drift = lambda x, t, *args, **kwargs: 0.0
        forward_diff = lambda t: self.sigma_t(t) * np.sqrt(
            2 * (np.log(sigma_max) - np.log(sigma_min))
        )
        super().__init__(
            drift=forward_drift,
            diffusion=forward_diff,
            use_backward_ode=use_backward_ode,
            prior=prior,
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


# %%
# if __name__ == "__main__":
import numpy as np
from deepinv.utils.demo import load_url_image, get_image_url
import deepinv as dinv
import matplotlib.pyplot as plt
from deepinv.models.edm import SongUNet, EDMPrecond

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# denoiser = dinv.models.DRUNet(pretrained="download").to(device)
unet = SongUNet.from_pretrained("song-unet-edm-ffhq64-uncond-ve")
denoiser = EDMPrecond(model=unet).to(device)

url = get_image_url("CBSD_0010.png")
x = load_url_image(url=url, img_size=64, device=device)
t = 100.0
x_noisy = x + torch.randn_like(x) * t
x_denoised = denoiser(x_noisy, t)
dinv.utils.plot([x, x_noisy, x_denoised], titles=["sample", "y", "denoised"])
_ = plt.hist(x_denoised.detach().cpu().numpy().ravel(), bins=100)
plt.show()
# %%
x_noisy = (x_noisy + 1) * 0.5
x_denoised = denoiser(x_noisy, t * 0.5)
x_denoised = x_denoised * 2 - 1
dinv.utils.plot([x, x_noisy, x_denoised], titles=["sample", "y", "denoised"])
_ = plt.hist(x_denoised.detach().cpu().numpy().ravel(), bins=100)
plt.show()
# %%
# VESDE
sigma_min = 0.02
sigma_max = 50.0
timesteps = np.linspace(0.001, 1.0, 100)
sigma_t = lambda t: sigma_min * (sigma_max / sigma_min) ** t
rng = torch.Generator(device).manual_seed(42)


prior = dinv.optim.prior.DiffusionScorePrior(
    denoiser=denoiser, sigma_t=sigma_t, rescale=False
)

sde = VESDE(
    prior=prior,
    sigma_max=sigma_max,
    sigma_min=sigma_min,
    rng=rng,
    device=device,
    use_backward_ode=False,
)

# Check forward
x_init = x.clone()

solution = sde.forward_sde.sample(x_init, timesteps=timesteps, method="euler")
dinv.utils.plot(solution.sample, suptitle=f"Forward sample, nfe = {solution.nfe}")
_ = plt.hist(solution.sample.ravel().cpu().numpy(), bins=100)
plt.show()
# %% Check backward
# x_init = torch.randn((1, 3, 64, 64), device=device, generator=rng) * sigma_max

solution = sde((1, 3, 64, 64), timesteps=timesteps[::-1], method="Euler", seed=1)
dinv.utils.plot(solution.sample, suptitle=f"Backward sample, nfe = {solution.nfe}")
_ = plt.hist(solution.sample.ravel().cpu().numpy(), bins=100)
plt.show()

# %%
