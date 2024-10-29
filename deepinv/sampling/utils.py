import torch
import numpy as np


class Welford:
    r"""
    Welford's algorithm for calculating mean and variance

    https://doi.org/10.2307/1266577
    """

    def __init__(self, x):
        self.k = 1
        self.M = x.clone()
        self.S = torch.zeros_like(x)

    def update(self, x):
        self.k += 1
        Mnext = self.M + (x - self.M) / self.k
        self.S = self.S + (x - self.M) * (x - Mnext)
        self.M = Mnext

    def mean(self):
        return self.M

    def var(self):
        return self.S / (self.k - 1)


def refl_projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    x = torch.abs(x)
    return torch.clamp(x, min=lower, max=upper)


def projbox(x, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=lower, max=upper)


def exp(input):
    if isinstance(input, torch.Tensor):
        return torch.exp(input)
    elif isinstance(input, [float, int, np.ndarray]):
        return np.exp(input)
    else:
        raise TypeError(f"Invalid type {type(input)} for computing exponential")


def get_edm_parameters(discretization: str = "edm"):
    r"""
    :param str discretization: discretization type for solving the SDE, one of 've', 'vp', 'edm
    """

    # Helper functions for VP
    if discretization == "vp":
        vp_beta_d = 19.9
        vp_beta_min = 0.1

        def sigma(t):
            return (exp(0.5 * vp_beta_d * (t**2) + vp_beta_min * t) - 1) ** 0.5

        vp_sigma = (
            lambda t: (np.e ** (0.5 * vp_beta_d * (t**2) + vp_beta_min * t) - 1) ** 0.5
        )
        vp_sigma_deriv = (
            lambda beta_d, beta_min: lambda t: 0.5
            * (beta_min + beta_d * t)
            * (sigma(t) + 1 / sigma(t))
        )
        vp_sigma_inv = (
            lambda sigma: (
                (vp_beta_min**2 + 2 * vp_beta_d * (sigma**2 + 1).log()) ** 0.5
                - vp_beta_min
            )
            / vp_beta_d
        )

        def vp_timesteps(num_steps):
            pass

        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma**2

    sigma_min = {"vp": vp_sigma(1e-3), "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
        discretization
    ]

    if discretization == "ve":
        pass
        pass
    if discretization == "edm":
        return {
            "sigma_min": 0.002,
            "sigma_max": 80,
        }
