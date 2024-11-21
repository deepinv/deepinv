# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional, Tuple
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
        **kwargs,
    ):
        r"""
        Solve the SDE with the given time-step.
        :param torch.Tensor x_init: initial value.
        :param timesteps: time steps at which to sample the SDE.
        :param str method: method for solving the SDE.

        :return torch.Tensor: samples from the SDE.
        """
        solver_fn = select_solver(method)
        solver = solver_fn(sde=self, rng=self.rng, **kwargs)
        samples = solver.sample(x_init, timesteps=timesteps, *args, **kwargs)
        return samples

    def discretize(
        self, x: Tensor, t: Union[Tensor, float], *args, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        return self.drift(x, t, *args, **kwargs), self.diffusion(t)

    def to(self, dtype=None, device=None):
        r"""
        Send the SDE to the desired device or dtype.
        This is useful when the drift of the diffusion term is parameterized (e.g., `deepinv.optim.ScorePrior`).
        """
        # Define the function to apply to each submodule
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if torch.device(device) != self.rng.device:
            self.rng = torch.Generator(device).set_state(self.initial_random_state)

        def apply_fn(module):
            module.to(device=device, dtype=dtype)

        # Use apply to run apply_fn on all submodules
        self.apply(apply_fn)

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
            d x_{t} = \left( f(x_t, t) - \frac{1}{2} g(t)^2 \nabla \log p_t(x_t) \right) dt

    :param callable drift: a time-dependent drift function :math:`f(x, t)` of the forward-time SDE.
    :param callable diffusion: a time-dependent diffusion function :math:`g(t)` of the forward-time SDE.
    :param deepinv.prior.ScorePrior prior: a time-dependent score prior, corresponding to :math:`\nabla \log p_t`
    :param torch.Generator rng: pseudo-random number generator for reproducibility.
    """

    def __init__(
        self,
        drift: Callable = lambda x, t: -x,
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
        self.use_backward_ode = use_backward_ode
        drift_forw = lambda x, t, *args, **kwargs: drift(x, t)
        diff_forw = lambda t: diffusion(t)
        self.forward_sde = BaseSDE(
            drift=drift_forw, diffusion=diff_forw, rng=rng, dtype=dtype, device=device
        )

        if self.use_backward_ode:
            drift_back = lambda x, t, *args, **kwargs: -drift(x, t) + 0.5 * (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        else:
            drift_back = lambda x, t, *args, **kwargs: -drift(x, t) + (
                diffusion(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        diff_back = lambda t: diffusion(t)
        self.backward_sde = BaseSDE(
            drift=drift_back, diffusion=diff_back, rng=rng, dtype=dtype, device=device
        )

    def score(self, x: Tensor, sigma: Union[Tensor, float], rescale: bool = False):
        if rescale:
            x = (x + 1) * 0.5
            sigma_in = sigma * 0.5
        else:
            sigma_in = sigma
        score = -self.prior.grad(x, sigma_in)
        if rescale:
            score = score * 2 - 1
        return score

    @torch.no_grad()
    def forward(
        self, x_init: Optional[Tensor], timesteps: Tensor, method: str = "Euler"
    ):
        return self.backward_sde.sample(x_init, timesteps=timesteps, method=method)


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        name: str = "ve",
        use_backward_ode=True,
        rng: torch.Generator = None,
        dtype=torch.float32,
        device=torch.device("cpu"),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_backward_ode = use_backward_ode
        params = get_edm_parameters(name)

        self.timesteps_fn = params["timesteps_fn"]
        self.sigma_max = params["sigma_max"]

        sigma_fn = params["sigma_fn"]
        sigma_deriv_fn = params["sigma_deriv_fn"]
        beta_fn = params["beta_fn"]
        s_fn = params["s_fn"]
        s_deriv_fn = params["s_deriv_fn"]

        # Forward SDE
        drift_forw = lambda x, t, *args, **kwargs: (
            -sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
        ) * self.score(x, sigma_fn(t), *args, **kwargs)
        diff_forw = lambda t: sigma_fn(t) * (2 * beta_fn(t)) ** 0.5

        # Backward SDE
        if self.use_backward_ode:
            diff_back = lambda t: 0.0
            drift_back = lambda x, t, *args, **kwargs: -(
                (s_deriv_fn(t) / s_fn(t)) * x
                - (s_fn(t) ** 2)
                * sigma_deriv_fn(t)
                * sigma_fn(t)
                * self.score(x, sigma_fn(t), *args, **kwargs)
            )
        else:
            drift_back = lambda x, t, *args, **kwargs: (
                sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
            ) * self.score(x, sigma_fn(t), *args, **kwargs)
            diff_back = diff_forw

        self.forward_sde = BaseSDE(
            drift=drift_forw,
            diffusion=diff_forw,
            rng=rng,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )
        self.backward_sde = BaseSDE(
            drift=drift_back,
            diffusion=diff_back,
            rng=rng,
            dtype=dtype,
            device=device,
            *args,
            **kwargs,
        )

    @torch.no_grad()
    def forward(
        self,
        latents: Optional[Tensor] = None,
        shape: Tuple[int, ...] = None,
        method: str = "Euler",
        max_iter: int = 100,
    ):
        if latents is None:
            latents = (
                torch.randn(shape, device=device, generator=self.rng) * self.sigma_max
            )
        return self.backward_sde.sample(
            latents, timesteps=self.timesteps_fn(max_iter), method=method
        )


if __name__ == "__main__":
    from edm import load_model
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    x_noisy = x + torch.randn_like(x) * 0.3
    dinv.utils.plot(
        [x, x_noisy, denoiser(x_noisy, 0.3)], titles=["sample", "y", "denoised"]
    )

    # denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
    # denoiser = dinv.models.DRUNet(device=device)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

    # EDM generation
    sde = EDMSDE(name="ve", prior=prior, use_backward_ode=False)
    sde.to("cpu")
    x_cpu = sde(shape=(1, 3, 64, 64), max_iter=10, method="heun")
    sde.to("cuda")
    x = sde(shape=(1, 3, 64, 64), max_iter=10, method="heun")
    dinv.utils.plot([x_cpu, x], titles=["cpu", "cuda"])
