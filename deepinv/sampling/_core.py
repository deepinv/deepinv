import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional, Tuple
from numpy import ndarray
import warnings


class BaseSDE(nn.Module):
    r"""
    Base class for Stochastic Differential Equation (SDE):
    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`w` is the standard Brownian motion.

    It defines the common interface for drift and diffusion functions.

    :param callable drift: a time-dependent drift function f(x, t)
    :param callable diffusion: a time-dependent diffusion function g(t)
    """

    def __init__(
        self,
        drift: Callable,
        diffusion: Callable,
        dtype=torch.float32,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.dtype = dtype
        self.device = device

    @property
    def dtype(self):
        return self.dtype

    @property
    def device(self):
        return self.device

    def discretize(self, x: Tensor, t: Union[Tensor, float]) -> Tuple[Tensor, Tensor]:
        return self.drift(x, t), self.diffusion(t)

    def to(self, dtype=None, device=None):
        r"""
        Send the SDE to the desired device or dtype. This is useful when the drift of the diffusion term is parameterized (e.g., `deepinv.optim.ScorePrior`).
        """
        # Define the function to apply to each submodule
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device

        def apply_fn(module):
            module.to(device=device, dtype=dtype)

        # Use apply to run apply_fn on all submodules
        self.apply(apply_fn)


class BaseSDESolver(nn.Module):
    def __init__(
        self,
        sde: BaseSDE,
        rng: Optional[torch.Generator] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sde = sde

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def step(self, t0, t1, x0: Tensor, *args, **kwargs):
        r"""
        Defaults to Euler step
        """
        drift, diffusion = self.sde.discretize(x0, t0)
        dt = t1 - t0
        dW = self.randn_like(x0) * abs(dt) ** 0.5
        return x0 + drift(x0, t0, *args, **kwargs) * dt + diffusion(t0) * dW

    @torch.no_grad()
    def sample(
        self, x_init: Tensor, *args, timesteps: Union[Tensor, ndarray] = None, **kwargs
    ) -> Tensor:
        x = x_init
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            x = self.step(t_cur, t_next, x, *args, **kwargs)
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
