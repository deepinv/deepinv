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
        prior: Callable = None,
        T: float = 1.0,
        rng: torch.Generator = None,
    ):
        super().__init__()
        self.T = T
        self.drift_forw = lambda x,t : f(x, t)
        self.diff_forw = lambda t : g(t)
        self.drift_back = lambda x,t,alpha : - f(x, T-t) - (1 + alpha**2) * prior.grad(x, T-t) * g(T-t)**2
        self.diff_back = lambda t, alpha : alpha * g(T-t)

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def forward_sde(self, x: Tensor, num_steps: int = 1000) -> Tensor:
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_forw(x,t) + self.diff_forw(t) * np.sqrt(stepsize) * torch.randn_like(x)
        return x

    def backward_sde(
        self, x: Tensor, num_steps: int = 100, alpha: float = 1.0
    ) -> Tensor:
        stepsize = self.T / num_steps
        for k in range(num_steps):
            t = k * stepsize
            x += stepsize * self.drift_back(x,t,alpha) + self.diff_back(t,alpha) * np.sqrt(stepsize) * torch.randn_like(x)
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


def get_edm_default_noise_scheduler():
    sigma_min = 0.002
    sigma_max = 80.0
    sigma_data = 0.5
    rho = 0.7
    P_mean = -1.2
    P_std = 1.2
    sigma = lambda t: t
    alpha = lambda t: (2 * beta(t)) * sigma(t)


class EDMSDE(DiffusionSDE):
    def __init__(self, score: Callable, T: float, sigma: Callable, alpha: Callable):
        super().__init__(score=score, T=T)


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.utils.demo import load_url_image, get_image_url

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url, grayscale=False).to(device)

    denoiser = dinv.models.WaveletDenoiser(wv="db8", level=4, device=device)
    prior = dinv.optim.prior.ScorePrior(denoiser = denoiser)
    
    OUSDE = DiffusionSDE(prior=prior, T=1.0)
    sample_noise = OUSDE.forward_sde(x)
    sample = OUSDE.backward_sde(torch.randn_like(x))
    dinv.utils.plot([x, sample_noise, sample])
