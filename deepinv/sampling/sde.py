# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable
import numpy as np
import warnings


class SDE(nn.Module):
    def __init__(
        self, drift: Callable, diffusion: Callable, t_init : float = 0., t_end : float = 1., rng: torch.Generator = None
    ):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion
        self.t_init = t_init
        self.t_end = t_end

        self.rng = rng
        if rng is not None:
            self.initial_random_state = rng.get_state()

    def euler_step(self, x, t, stepsize):
        return (
            x
            + stepsize * self.drift(x, t)
            + self.diffusion(t) * self.randn_like(x) * stepsize**0.5
        )

    def sample(
        self,
        x_init: Tensor,
        num_steps: int = 100,
        method = 'euler'
    ):
        stepsize = (self.t_end - self.t_init) / num_steps
        x = x_init
        for k in range(num_steps):
            t = self.t_init + k * stepsize
            if method == 'euler':
                x = self.euler_step(x, t, stepsize)
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
        self.drift_forw = lambda x, t: f(x, t)
        self.diff_forw = lambda t: g(t)
        self.forward_sde = SDE(drift=self.drift_forw, diffusion=self.diff_forw, t_end = T, rng=rng)

        self.drift_back = lambda x, t: -f(x, T - t) - (g(T - t) ** 2) * prior.grad(
            x, T - t
        )
        self.diff_back = lambda t: g(T - t)
        self.backward_sde = SDE(drift=self.drift_back, diffusion=self.diff_back, t_end = T, rng=rng)


class EDMSDE(DiffusionSDE):
    def __init__(
        self,
        prior: Callable,
        T: float,
        sigma: Callable =  lambda t: t,
        sigma_prime: Callable =  lambda t: 1,
        s: Callable =  lambda t: 1,
        s_prime: Callable = lambda t : 0,
        beta: Callable = lambda t: 1.0 / t,
        rng: torch.Generator = None,
    ): 
        self.sigma = sigma
        self.beta = beta
        self.drift_forw = lambda x, t: (- sigma_prime(t) * sigma(t) + beta(t) * self.sigma(t) ** 2) * (-prior.grad(x, sigma(t)))
        self.diff_forw = lambda t: self.sigma(t) * (2 * beta(t)) ** 0.5
        self.drift_back = lambda x, t: (sigma_prime(t) * sigma(t) + beta(t) * self.sigma(t) ** 2) * (-prior.grad(x, sigma(t)))
        self.diff_back = self.diff_forw
        super().__init__(prior=prior, T=T, rng=rng)


# %%
if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.utils.demo import load_url_image, get_image_url

    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    #denoiser = dinv.models.DRUNet(device = device)
    denoiser = dinv.models.DiffUNet().to(device)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

    OUSDE = EDMSDE(prior=prior, T=1.0)
    with torch.no_grad():
        sample_noise = OUSDE.forward_sde.sample(x, num_steps = 100)
        noise = torch.randn_like(x)
        sample = OUSDE.backward_sde.sample(noise, num_steps = 100)
    dinv.utils.plot([x, sample_noise, sample])

    # from temp_model import UNetModelWrapper

    # class EDMDenoiser(nn.Module):
    #     def __init__(
    #         self,
    #         image_size=(3, 32, 32),
    #         sigma_data: float = 0.5,
    #     ):
    #         super().__init__()

    #         # Any U-Net network
    #         self.model = UNetModelWrapper(
    #             dim=image_size,
    #             num_channels=64,
    #             num_res_blocks=4,
    #             attention_resolutions="32, 16",
    #         )
    #         self.sigma_data = sigma_data

    #     def forward(self, input: Tensor, sigma: float):
    #         if isinstance(sigma, float):
    #             sigma = torch.tensor([sigma], device=input.device)
    #         skip_scaling, output_scaling, input_scaling, noise_condition = (
    #             self.get_scaling_coefficients(sigma)
    #         )
    #         if isinstance(noise_condition, Tensor):
    #             if noise_condition.ndim == 4:
    #                 t = noise_condition.squeeze((1, 2, 3)).to(input.dtype)
    #             else:
    #                 t = noise_condition.to(input.dtype)
    #         return skip_scaling * input + output_scaling * self.model(
    #             x=input_scaling * input, t=t
    #         )

    #     def get_scaling_coefficients(self, sigma: Tensor):
    #         r"""
    #         Get scaling coefficients for the input and output of the denoiser
    #         """

    #         sum_of_square = (self.sigma_data**2 + sigma**2) ** 0.5
    #         skip_scaling = self.sigma_data**2 / sum_of_square**2
    #         output_scaling = sigma * self.sigma_data / sum_of_square
    #         input_scaling = 1.0 / sum_of_square
    #         noise_condition = 0.25 * torch.log(sigma)

    #         return skip_scaling, output_scaling, input_scaling, noise_condition

    # score_model = EDMDenoiser()
    # score_model.load_state_dict(
    #     torch.load("/home/minhhai/Works/dev/deepinv_folk/weights/cifar10_27M.pth")[
    #         "model_state_dict"
    #     ]
    # )
    # # score_model.to(device)

    # print(
    #     "Number of parameters: ",
    #     sum(p.numel() for p in score_model.parameters() if p.requires_grad),
    # )

    # x = load_url_image(url=url, img_size=32, device=device)
    # # x = x * 2 - 1
    # # denoiser = dinv.models.DiffUNet().to(device)
    # prior = dinv.optim.prior.ScorePrior(denoiser=score_model)

    # OUSDE = EDMSDE(prior=prior, T=1.0)
    # with torch.no_grad():
    #     sample_noise = OUSDE.forward_sde.sample(x, 0, 1.0, 100)
    #     noise = torch.randn_like(x)
    #     sample = OUSDE.backward_sde.sample(noise, 1e-4, 1.0, num_steps=10000)
    # dinv.utils.plot([x, sample_noise, sample])
