# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
import warnings
from utils import get_edm_parameters
from noisy_datafidelity import NoisyDataFidelity, DPSDataFidelity

class SDE_solver(nn.Module):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def step(self, t0, t1, x0):
        pass

    def sample(self, x_init: Tensor, *args, timesteps: Tensor = None, **kwargs) -> Tensor:
        x = x_init
        for i, t in enumerate(timesteps[:-1]):
            x = self.step(t, timesteps[i + 1], x, *args, **kwargs)
        return x

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


class Euler_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng=rng)

    def step(self, t0, t1, x0, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        return x0 + self.drift(x0, t0, *args, **kwargs) * dt + self.diffusion(t0) * dW


class Heun_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng=rng)

    def step(self, t0, t1, x0, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        diff_x0 = self.diffusion(t0)
        drift_x0 = self.drift(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_x0 * dt + diff_x0 * dW
        diff_x1 = self.diffusion(t1)
        drift_x1 = self.drift(x_euler, t1, *args, **kwargs)
        return x0 + 0.5 * (drift_x0 + drift_x1) * dt + 0.5 * (diff_x0 + diff_x1) * dW


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        prior: Callable = None,
        solver_name: str = 'Euler',
        rng: torch.Generator = None,
        use_backward_ode=False,
    ):
        super().__init__()
        self.prior = prior
        self.use_backward_ode = use_backward_ode
        self.drift_forw = lambda x, t, *args, **kwargs: f(x, t)
        self.diff_forw = lambda t: g(t)
        self.forward_sde = Euler_solver(
            drift=self.drift_forw, diffusion=self.diff_forw, rng=rng
        )
        if self.use_backward_ode:
            self.drift_back = lambda x, t, *args, **kwargs: f(x, t) - 0.5 * (g(t) ** 2) * self.score(x, t, *args, **kwargs)
        else:
            self.drift_back = lambda x, t, *args, **kwargs: f(x, t) - (g(t) ** 2) * self.score(x, t, *args, **kwargs)
        self.diff_back = lambda t: g(t)
        self.solver_name = solver_name
        self.rng = rng
        self.define_solvers()
    
    def define_solvers(self):
        if self.solver_name.lower() == 'euler':
           self.backward_sde = Euler_solver(
            drift=self.drift_back, diffusion=self.diff_back, rng=self.rng
        ) 
        elif self.solver_name.lower() == 'heun':
            self.backward_sde = Heun_solver(
            drift=self.drift_back, diffusion=self.diff_back, rng=self.rng
            )
           
    def score(self, x, sigma):
        return -self.prior.grad(x, sigma)

class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        *args,
        name: str  = 'VE',
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        params = get_edm_parameters(name)
        self.timesteps_fn = params["timesteps_fn"]
        sigma_fn = params["sigma_fn"]
        sigma_deriv = params["sigma_deriv"]
        beta_fn = params["beta_fn"]
        self.sigma_max = params["sigma_max"]
        s_fn = params["s_fn"]
        s_deriv = params["s_deriv"]
        self.drift_forw = lambda x, t, *args, **kwargs: (
            -sigma_deriv(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
        ) *  self.score(x, sigma_fn(t), *args, **kwargs)
        self.diff_forw = lambda t: sigma_fn(t) * (2 * beta_fn(t)) ** 0.5
        if self.use_backward_ode:
            self.diff_back = lambda t: 0.0
            self.drift_back = lambda x, t, *args, **kwargs: -(
                (s_deriv(t) / s_fn(t)) * x
                - (s_fn(t) ** 2)
                * sigma_deriv(t)
                * sigma_fn(t)
                * self.score(x, sigma_fn(t), *args, **kwargs)
            )
        else:
            self.drift_back = lambda x, t, *args, **kwargs: (
                sigma_deriv(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
            ) * self.score(x, sigma_fn(t), *args, **kwargs)
            self.diff_back = self.diff_forw
        self.define_solvers()

    def forward(self, shape, max_iter = 100):
        with torch.no_grad():
            noise = torch.randn(shape, device=device) * self.sigma_max
            return self.backward_sde.sample(noise, timesteps=self.timesteps_fn(max_iter))

class PosteriorEDMSDE(EDMSDE):
    def __init__(
        self,
        prior: Callable,
        data_fidelity: Callable,
        *args,
        **kwargs
    ):
        super().__init__(
            prior=prior,
            *args,
            **kwargs
        )
        self.data_fidelity = data_fidelity

    def score(self, x, sigma, y, physics):
        return -self.prior.grad(x, sigma) - self.data_fidelity.grad(x, y, physics, sigma)
    
    def forward(self, y, physics, max_iter = 100):
        with torch.no_grad():
            noise = torch.randn_like(y) * self.sigma_max
            return self.backward_sde.sample(noise, y, physics, timesteps=self.timesteps_fn(max_iter))


if __name__ == "__main__":
    from edm import load_model
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
    denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

    # EDM generation
    # sde = EDMSDE(name = 've', prior=prior, use_backward_ode=True, solver_name = 'Heun')
    # sample = sde((1, 3, 64, 64), max_iter = 20)

    # Posterior EDM generation
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device) 
    physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=.5, device=device) 
    noisy_data_fidelity = DPSDataFidelity(denoiser = denoiser)
    y = physics(x)
    posterior_sde = PosteriorEDMSDE(prior=prior, data_fidelity = noisy_data_fidelity, name = 've', use_backward_ode=True, solver_name = 'Heun')
    posterior_sample = posterior_sde(y, physics, max_iter = 20)

    # Plotting the samples
    dinv.utils.plot(posterior_sample)

