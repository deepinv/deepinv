import torch
import torch.nn as nn
from torch import Tensor
import warnings
from typing import Optional, Union, Any
from numpy import ndarray
from tqdm import tqdm


class SDEOutput(dict):
    r"""
    A container for storing the output of an SDE solver, that behaves like a `dict` but allows access with the attribute syntax.

    Attributes:
    :attr torch.Tensor sample: the final samples of the sampling process, of shape ``(B, C, H, W)``.
    :attr torch.Tensor trajectory: the trajectory of the sampling process, of shape ``(num_steps, B, C, H, W)`` if ``full_trajectory`` is ``True``, otherwise of shape ``(B, C, H, W)``.
    :attr torch.Tensor timesteps: the time steps at which the samples were taken, of shape ``(num_steps,)``.
    :attr int nfe: the number of function evaluations performed during the integration.
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

    .. math::
        d x_{t} = f(x_t, t) dt + g(t) d w_{t}

    where :math:`f` is the drift term, :math:`g` is the diffusion coefficient, and :math:`w_t` is a standard Brownian process.

    Currently only supported for fixed time steps for numerical integration.

    :param torch.Tensor, numpy.ndarray, list timesteps: time steps at which the SDE will be discretized.e.
    :param torch.Generator rng: a random number generator for reproducibility, optional.
    :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.
    """

    def __init__(
        self,
        timesteps: Union[Tensor, ndarray],
        rng: Optional[torch.Generator] = None,
    ):
        super().__init__()
        if isinstance(timesteps, ndarray):
            self.timesteps = torch.from_numpy(timesteps.copy())
        elif isinstance(timesteps, Tensor):
            self.timesteps = timesteps
        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()
            self.timesteps = self.timesteps.to(rng.device)

    def step(self, sde, t0: float, t1: float, x0: Tensor, *args, **kwargs) -> Tensor:
        r"""
        Perform a single step with step size from time `t0` to time `t1`, with current state `x0`.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param float or torch.Tensor t0: Time at the start of the step, of size (,).
        :param float or torch.Tensor t1: Time at the end of the step, of size (,).
        :param torch.Tensor x0: Current state of the system, of size (batch_size, d).
        :return: Updated state of the system after the step.

        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        sde,
        x_init: Tensor,
        seed: int = None,
        *args,
        timesteps: Union[Tensor, ndarray] = None,
        get_trajectory: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> SDEOutput:
        r"""
        Solve the Stochastic Differential Equation (SDE) with given time steps.

        This function iteratively applies the SDE solver step for each time interval
        defined by the provided timesteps.

        :param deepinv.sampling.BaseSDE sde: the SDE to solve.
        :param torch.Tensor x_init: The initial state of the system.
        :param int seed: The seed for the random number generator, if `rng` is provided.
        :param torch.Tensor, numpy.ndarray, list timesteps: A sequence of time points at which to solve the SDE. If None, default timesteps will be used.
        :param bool get_trajectory: whether to return the full trajectory of the SDE or only the last sample, optional. Default to False.
        :param bool verbose: whether to display a progress bar during the sampling process, optional. Default to False.
        :param \*args: Variable length argument list to be passed to the step function.
        :param \*\*kwargs: Arbitrary keyword arguments to be passed to the step function.

        :return: SDEOutput
        """
        self.rng_manual_seed(seed)
        x = x_init
        nfe = 0
        trajectory = [x_init.clone()] if get_trajectory else []

        if timesteps is None:
            timesteps = self.timesteps.to(sde.device, sde.dtype)
        else:
            if isinstance(timesteps, ndarray):
                timesteps = torch.from_numpy(timesteps.copy())
            timesteps = timesteps.to(sde.device, sde.dtype)

        for t_cur, t_next in tqdm(
            zip(timesteps[:-1], timesteps[1:], strict=True),
            total=len(timesteps) - 1,
            disable=not verbose,
        ):
            x, cur_nfe = self.step(sde, t_cur, t_next, x, *args, **kwargs)
            nfe += cur_nfe
            if get_trajectory:
                trajectory.append(x.clone())
        if get_trajectory:
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
        :param int seed: The seed for the random number generator, if `rng` is provided.

        :return: A tensor of the same size as input filled with random numbers from a normal distribution.
        :rtype: torch.Tensor

        This method uses the `rng` attribute of the class, which is a pseudo-random number generator
        for reproducibility. If a seed is provided, it will be used to set the state of `rng` before
        generating the random numbers.

        .. note::
           The `rng` attribute must be initialized for this method to work properly.
        """
        self.rng_manual_seed(seed)
        return torch.empty_like(input).normal_(generator=self.rng)


class EulerSolver(BaseSDESolver):
    r"""
    Euler-Maruyama solver for SDEs.

    This solver uses the Euler-Maruyama method to numerically integrate SDEs. It is a first-order method that
    approximates the solution using the following update rule:

    .. math::

        x_{t+dt} = x_t + f(x_t,t)dt + g(t) W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Generator rng: A random number generator for reproducibility.
    """

    def __init__(self, timesteps, rng: torch.Generator = None):
        super().__init__(timesteps, rng=rng)

    def step(self, sde, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift, diffusion = sde.discretize(x0, t0, *args, **kwargs)
        return x0 + drift * dt + diffusion * dW, 1


class HeunSolver(BaseSDESolver):
    r"""
    Heun solver for SDEs.

    This solver uses the second-order Heun method to numerically integrate SDEs, defined as:

    .. math::
        \tilde{x}_{t+dt} &= x_t + f(x_t,t)dt + g(t) W_{dt} \\
        x_{t+dt} &= x_t + \frac{1}{2}[f(x_t,t) + f(\tilde{x}_{t+dt},t+dt)]dt + \frac{1}{2}[g(t) + g(t+dt)] W_{dt}

    where :math:`W_t` is a Gaussian random variable with mean 0 and variance dt.

    :param torch.Tensor timesteps: The time steps at which to evaluate the solution.
    :param torch.Generator rng: A random number generator for reproducibility.
    """

    def __init__(
        self,
        timesteps,
        rng: torch.Generator = None,
    ):
        super().__init__(timesteps, rng=rng)

    def step(self, sde, t0, t1, x0: Tensor, *args, **kwargs):
        dt = abs(t1 - t0)
        dW = self.randn_like(x0) * dt**0.5
        drift_0, diffusion_0 = sde.discretize(x0, t0, *args, **kwargs)
        x_euler = x0 + drift_0 * dt + diffusion_0 * dW
        drift_1, diffusion_1 = sde.discretize(x_euler, t1, *args, **kwargs)

        return (
            x0
            + 0.5 * (drift_0 + drift_1) * dt
            + 0.5 * (diffusion_0 + diffusion_1) * dW,
            2,
        )
