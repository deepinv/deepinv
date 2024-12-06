import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Optional, Union, Any
from numpy import ndarray
import numpy as np


class SDEOutput(dict):
    r"""
    A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.
    """

    def __init__(self, sample: Tensor, trajectory: Tensor, timesteps: Tensor, nfe: int):
        sol = {
            "sample": sample,
            "trajectory": trajectory,
            "timesteps": timesteps,
            "nfe": nfe,
        }
        super().__init__(sol)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class BaseSDESolver(nn.Module):
    r"""
    Base class for solving Stochastic Differential Equations (SDEs) from :class:`deepinv.sampling.BaseSDE` of the form:

    ..math:

        d\, x_t = f(x_t, t) d\,t  + g(t) d\,w_t

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient, and :math:`w_t` is a standard Brownian process.

    Currently only supported for fixed time steps for numerical integration.

    :param deepinv.sampling.BaseSDE sde: the SDE to solve.
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    """

    def __init__(
        self,
        sde,
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        self.sde = sde

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def step(self, t0: float, t1: float, x0: Tensor, *args, **kwargs) -> Tensor:
        r"""
        Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

        :param float or Tensor t0: Time at the start of the step, of size (,).
        :param float or Tensor t1: Time at the end of the step, of size (,).
        :param Tensor x0: Current state of the system, of size (batch_size, d).
        :param \*args: Variable length argument list.
        :param \**kwargs: Arbitrary keyword arguments.
        :return: Updated state of the system after the step.

        :rtype: Tensor
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        *args,
        timesteps: Union[Tensor, ndarray] = None,
        full_trajectory: bool = False,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the Stochastic Differential Equation (SDE) with given time steps.

        This function iteratively applies the SDE solver step for each time interval
        defined by the provided timesteps.

        :param Tensor x_init: The initial state of the system.
        :param Union[Tensor, ndarray] timesteps: A sequence of time points at which to solve the SDE. If None, default timesteps will be used.
        :param \*args: Variable length argument list to be passed to the step function.
        :param \**kwargs: Arbitrary keyword arguments to be passed to the step function.

        :return: The solution of the system after solving the SDE across all timesteps, with the following attributes:
            sample torch.Tensor: the final sample of the SDE, of the same shape as `x_init`.
            trajectory torch.Tensor: the trajectory of the SDE, of shape `(num_timesteps, *x_init.shape)` if `full_trajectory = True`, otherwise equal to `sample`.
            timestep torch.Tensor: the discrete timesteps.
            nfe int: the number of function evaluations.
        """
        x = x_init
        nfe = 0
        trajectory = [x_init.clone()] if full_trajectory else []
        for t_cur, t_next in zip(timesteps[:-1], timesteps[1:]):
            x, cur_nfe = self.step(t_cur, t_next, x, *args, **kwargs)
            nfe += cur_nfe
            if full_trajectory:
                trajectory.append(x.clone())
        if full_trajectory:
            trajectory = torch.stack(trajectory, dim=0)
        else:
            trajectory = x
        output = SDEOutput(
            sample=x, trajectory=trajectory, timesteps=timesteps, nfe=nfe
        )

        return output

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
        Equivalent to :func:`torch.randn_like` but supports a pseudorandom number generator argument.

        :param torch.Tensor input: The input tensor whose size will be used.
        :param int seed: The seed for the random number generator, if :attr:`rng` is provided.

        :return: A tensor of the same size as input filled with random numbers from a normal distribution.
        :rtype: torch.Tensor

        This method uses the :attr:`rng` attribute of the class, which is a pseudo-random number generator
        for reproducibility. If a seed is provided, it will be used to set the state of :attr:`rng` before
        generating the random numbers.

        .. note::
           The :attr:`rng` attribute must be initialized for this method to work properly.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class EulerSolver(BaseSDESolver):
    def __init__(self, sde, rng: torch.Generator = None):
        super().__init__(sde, rng=rng)

    def step(self, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift, diffusion = self.sde.discretize(x0, t0, *args, **kwargs)
        return x0 + drift * dt + diffusion * dW, 1


class HeunSolver(BaseSDESolver):
    def __init__(
        self,
        sde,
        rng: torch.Generator = None,
    ):
        super().__init__(sde, rng=rng)

    def step(self, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift_0, diffusion_0 = self.sde.discretize(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_0 * dt + diffusion_0 * dW
        drift_1, diffusion_1 = self.sde.discretize(x_euler, t1, *args, **kwargs)

        return (
            x0
            + 0.5 * (drift_0 + drift_1) * dt
            + 0.5 * (diffusion_0 + diffusion_1) * dW,
            2,
        )


def select_solver(name: str = "euler") -> BaseSDESolver:
    if name.lower() == "euler":
        return EulerSolver
    elif name.lower() == "heun":
        return HeunSolver
    else:
        raise ValueError("Invalid SDE solver name.")
