import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
import warnings


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        beta: Callable = lambda t: 1,
        prior: Callable = None,
        T: float = 1.0,
        rng: torch.Generator = None,
    ):
        super().__init__()
        self.T = T
        self.f = f
        self.g = g
        self.drift_forw = lambda x, t: self.f(x, t)
        self.diff_forw = lambda t: self.g(t)
        self.drift_back = (
            lambda x, t: -self.f(x, T - t) - (1 + beta(t)) * prior.grad(x, T - t) * self.g(T - t) ** 2
        )
        self.diff_back = lambda t : np.sqrt(beta(t)) * self.g(T - t)

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def forward_sde(self, x: Tensor, num_steps: int = 100) -> Tensor:
        stepsize = self.T / num_steps
        x = x.clone()
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_forw(x, t) + self.diff_forw(t) * np.sqrt(stepsize) * self.randn_like(x)
        return x

    def backward_sde(
        self, x: Tensor, num_steps: int = 100) -> Tensor:
        stepsize = self.T / num_steps
        x = x.clone()
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_back(x, t) + self.diff_back(t) * np.sqrt(stepsize) * self.randn_like(x)
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


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        prior: Callable,
        T: float,
        sigma: lambda t: t,
        alpha: lambda t: np.sqrt(2) * t,
        rng: torch.Generator = None,
    ):
        super().__init__(prior=prior, T=T, rng=rng)
        self.sigma = sigma
        self.alpha = alpha

        self.drift_forw = lambda x, t: (
            self.sigma(t) - 0.5 * self.alpha(t) ** 2
        ) * prior.grad(x, sigma(t))
        self.diff_forw = lambda t: alpha(t)

        self.drift_back = lambda x, t: (
            -self.sigma(t) - 0.5 * self.alpha(t) ** 2
        ) * prior.grad(x, sigma(t))
        self.diff_back = self.diff_forw


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.utils.demo import load_url_image, get_image_url

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    x = x*2 - 1
    denoiser = dinv.models.DiffUNet().to(device)
    prior = dinv.optim.prior.ScorePrior(denoiser = denoiser)
    OUSDE = DiffusionSDE(prior=prior, T=1.0)
    with torch.no_grad():
        sample_noise = OUSDE.forward_sde(x)
        noise = torch.randn_like(x)
        sample = OUSDE.backward_sde(noise, num_steps = 10)
    dinv.utils.plot([(x + 1/2), sample_noise, (sample + 1)/2])
