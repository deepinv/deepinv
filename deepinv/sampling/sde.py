# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
import warnings


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

    def sample(self, x_init: Tensor, timesteps: Tensor = None):
        x = x_init
        for i, t in enumerate(timesteps[:-1]):
            x = self.step(t, timesteps[i + 1], x)
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

    def step(self, t0, t1, x0):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        return x0 + self.drift(x0, t0) * dt + self.diffusion(t0) * dW


class Heun_solver(SDE_solver):
    def __init__(
        self, drift: Callable, diffusion: Callable, rng: torch.Generator = None
    ):
        super().__init__(drift, diffusion, rng=rng)

    def step(self, t0, t1, x0):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        diff_x0 = self.diffusion(t0)
        drift_x0 = self.drift(x0, t0)
        x_euler = x0 + drift_x0 * dt + diff_x0 * dW
        diff_x1 = self.diffusion(t1)
        drift_x1 = self.drift(x_euler, t1)
        return x0 + 0.5 * (drift_x0 + drift_x1) * dt + 0.5 * (diff_x0 + diff_x1) * dW


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        prior: Callable = None,
        rng: torch.Generator = None,
        use_backward_ode=False,
    ):
        super().__init__()
        self.prior = prior
        self.use_backward_ode = use_backward_ode
        self.drift_forw = lambda x, t: f(x, t)
        self.diff_forw = lambda t: g(t)
        self.forward_sde = Euler_solver(
            drift=self.drift_forw, diffusion=self.diff_forw, rng=rng
        )
        if self.use_backward_ode:
            self.drift_back = lambda x, t: f(x, t) - 0.5 * (g(t) ** 2) * self.score(
                x, t
            )
        else:
            self.drift_back = lambda x, t: f(x, t) - (g(t) ** 2) * self.score(x, t)
        self.diff_back = lambda t: g(t)
        self.backward_sde = Euler_solver(
            drift=self.drift_back, diffusion=self.diff_back, rng=rng
        )

    def score(self, x, t, *args):
        return -self.prior.grad(x, t)


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        prior: Callable,
        sigma: Callable = lambda t: t,
        sigma_prime: Callable = lambda t: 1.0,
        s: Callable = lambda t: 1.0,
        s_prime: Callable = lambda t: 0.0,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None,
    ):
        super().__init__(prior=prior, rng=rng)
        self.drift_forw = lambda x, t: (
            -sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2
        ) * (-prior.grad(x, sigma(t)))
        self.diff_forw = lambda t: sigma(t) * (2 * beta(t)) ** 0.5
        if self.use_backward_ode:
            self.diff_back = lambda t: 0.0
            self.drift_back = lambda x, t: -(
                (s_prime(t) / s(t)) * x
                - (s(t) ** 2)
                * sigma_prime(t)
                * sigma(t)
                * (-prior.grad(x / s(t), sigma(t)))
            )
        else:
            self.drift_back = lambda x, t: (
                sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2
            ) * (-prior.grad(x, sigma(t)))
            self.diff_back = self.diff_forw
        self.forward_sde = Euler_solver(
            drift=self.drift_forw, diffusion=self.diff_forw, rng=rng
        )
        self.backward_sde = Heun_solver(
            drift=self.drift_back, diffusion=self.diff_back, rng=rng
        )


class PosteriorSDE(EDMSDE):
    def __init__(
        self,
        prior: Callable,
        data_fidelity: Callable,
        sigma: Callable = lambda t: t,
        sigma_prime: Callable = lambda t: 1.0,
        s: Callable = lambda t: 1.0,
        s_prime: Callable = lambda t: 0.0,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None,
    ):
        super().__init__(prior=prior, rng=rng)
        self.drift_forw = lambda x, t: (
            -sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2
        ) * self.score(x, sigma(t))
        self.diff_forw = lambda t: sigma(t) * (2 * beta(t)) ** 0.5
        if self.use_backward_ode:
            self.diff_back = lambda t: 0.0
            self.drift_back = lambda x, t: -(
                (s_prime(t) / s(t)) * x
                - (s(t) ** 2)
                * sigma_prime(t)
                * sigma(t)
                * self.score(x / s(t), sigma(t))
            )
        else:
            self.drift_back = lambda x, t: (
                sigma_prime(t) * sigma(t) + beta(t) * sigma(t) ** 2
            ) * self.score(x, sigma(t))
            self.diff_back = self.diff_forw
        self.forward_sde = Euler_solver(
            drift=self.drift_forw, diffusion=self.diff_forw, rng=rng
        )
        self.backward_sde = Heun_solver(
            drift=self.drift_back, diffusion=self.diff_back, rng=rng
        )


class PosteriorSDE(DiffusionSDE):
    def __init__(
        self,
        prior: Callable,
        data_fidelity: Callable,
        sigma: Callable = lambda t: t,
        sigma_prime: Callable = lambda t: 1.0,
        s: Callable = lambda t: 1.0,
        s_prime: Callable = lambda t: 0.0,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None,
    ):
        super().__init__(
            prior=prior,
            sigma=sigma,
            sigma_prime=sigma_prime,
            s=s,
            s_prime=s_prime,
            beta=beta,
            rng=rng,
        )
        self.data_fidelity = data_fidelity

    def score(self, x, y, t):
        return -self.prior.grad(x, t) - self.data_fidelity.grad(x, y, t)


class PosteriorEDMSDE(EDMSDE):
    def __init__(
        self,
        prior: Callable,
        data_fidelity: Callable,
        sigma: Callable = lambda t: t,
        sigma_prime: Callable = lambda t: 1.0,
        s: Callable = lambda t: 1.0,
        s_prime: Callable = lambda t: 0.0,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None,
    ):
        super().__init__(
            prior=prior,
            sigma=sigma,
            sigma_prime=sigma_prime,
            s=s,
            s_prime=s_prime,
            beta=beta,
            rng=rng,
        )
        self.data_fidelity = data_fidelity

    def score(self, x, y, t):
        return -self.prior.grad(x, t) - self.data_fidelity.grad(x, y, t)


if __name__ == "__main__":
    from edm import load_model
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
    denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)
    # url = get_image_url("CBSD_0010.png")
    # x = load_url_image(url=url, img_size=64, device=device)

    ve_sigma = lambda t: t**0.5
    ve_sigma_prime = lambda t: 1 / (2 * t**0.5)
    ve_beta = lambda t: 0
    ve_sigma_max = 100
    ve_sigma_min = 0.02
    num_steps = 20
    ve_timesteps = ve_sigma_max**2 * (ve_sigma_min**2 / ve_sigma_max**2) ** (
        np.arange(num_steps) / (num_steps - 1)
    )
    sde = EDMSDE(prior=prior, beta=ve_beta, sigma=ve_sigma, sigma_prime=ve_sigma_prime)

    with torch.no_grad():
        noise = torch.randn(2, 3, 64, 64, device=device) * ve_sigma_max
        samples = sde.backward_sde.sample(noise, timesteps=ve_timesteps)
    dinv.utils.plot(samples)
