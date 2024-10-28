import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import warnings


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        score: Callable = None,
        T: float = 1.0,
        rng: torch.Generator = None,
    ):
        super().__init__()
        self.f = f
        self.g = g
        self.score = score
        self.T = T

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def forward_sde(self, x: Tensor, num_steps: int) -> Tensor:
        x_new = x
        stepsize = 1.0 / num_steps
        for t in range(num_steps):
            dw = self.randn_like(x_new)
            f_dt = self.f(x_new, t)
            g_dw = self.g(t) * dw
            x_new = x_new + stepsize * (f_dt + g_dw)
        return x_new

    def backward_sde(
        self, x: Tensor, num_steps: int = 1000, alpha: float = 1.0
    ) -> Tensor:
        dt = self.T / num_steps
        t = 0
        for n in range(num_steps):
            rt = self.T - t * n
            g = self.g(rt)
            drift = self.f(x, rt) - (1 + alpha**2) * g**2 * self.score(x, rt)
            diffusion = alpha * g
            dw = self.randn_like(x) * dt
            x = x + drift * dt + diffusion * dw

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
        self, score: Callable, T: float, sigma_min: float = 0.002, sigma_max: float = 80
    ):
        super().__init__(score=score, T=T)


if __name__ == "__main__":
    import deepinv as dinv

    device = torch.device("cuda")

    score = dinv.models.WaveletDenoiser(wv="db8", level=4, device=device)
    rng = torch.Generator(device=device).manual_seed(42)
    OUSDE = DiffusionSDE(score=score, T=1.0, rng=rng)

    x = torch.randn((2, 1, 28, 28), device=device)
    sample = OUSDE.backward_sde(x)

    dinv.utils.plot([x, sample])
