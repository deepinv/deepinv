# %%

import torch
import math
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union, Optional, Tuple
import numpy as np
import warnings
from utils import get_edm_parameters
from noisy_datafidelity import NoisyDataFidelity, DPSDataFidelity
from sde_solver import EulerSolver, HeunSolver
from deepinv.optim.prior import ScorePrior
from deepinv.physics import Physics


class DiffusionSDE(nn.Module):
    def __init__(
        self,
        f: Callable = lambda x, t: -x,
        g: Callable = lambda t: math.sqrt(2.0),
        prior: ScorePrior = None,
        solver_name: str = "Euler",
        rng: torch.Generator = None,
        use_backward_ode=False,
    ):
        super().__init__()
        self.prior = prior
        self.use_backward_ode = use_backward_ode
        self.drift_forw = lambda x, t, *args, **kwargs: f(x, t)
        self.diff_forw = lambda t: g(t)
        self.forward_sde = EulerSolver(
            drift=self.drift_forw, diffusion=self.diff_forw, rng=rng
        )
        if self.use_backward_ode:
            self.drift_back = lambda x, t, *args, **kwargs: -f(x, t) + 0.5 * (
                g(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        else:
            self.drift_back = lambda x, t, *args, **kwargs: -f(x, t) + (
                g(t) ** 2
            ) * self.score(x, t, *args, **kwargs)
        self.diff_back = lambda t: g(t)
        self.solver_name = solver_name
        self.rng = rng
        self._define_solvers()

    def _define_solvers(self):
        if self.solver_name.lower() == "euler":
            self.backward_sde = EulerSolver(
                drift=self.drift_back, diffusion=self.diff_back, rng=self.rng
            )
        elif self.solver_name.lower() == "heun":
            self.backward_sde = HeunSolver(
                drift=self.drift_back, diffusion=self.diff_back, rng=self.rng
            )

    def score(self, x: Tensor, sigma: Union[Tensor, float], rescale=False):
        if rescale:
            x = (x + 1) * 0.5
            sigma_in = sigma * 0.5
        else:
            sigma_in = sigma
        score = -self.prior.grad(x, sigma_in)
        if rescale:
            score = score * 2 - 1
        return score

    @torch.no_grad()
    def forward(self, init, timesteps):
        return self.backward_sde.sample(init, timesteps=timesteps)


class EDMSDE(DiffusionSDE):
    def __init__(self, *args, name: str = "ve", use_backward_ode=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_backward_ode = use_backward_ode
        params = get_edm_parameters(name)

        self.timesteps_fn = params["timesteps_fn"]
        self.sigma_max = params["sigma_max"]

        sigma_fn = params["sigma_fn"]
        sigma_deriv_fn = params["sigma_deriv_fn"]
        beta_fn = params["beta_fn"]
        s_fn = params["s_fn"]
        s_deriv_fn = params["s_deriv_fn"]

        # Forward SDE
        self.drift_forw = lambda x, t, *args, **kwargs: (
            -sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
        ) * self.score(x, sigma_fn(t), *args, **kwargs)
        self.diff_forw = lambda t: sigma_fn(t) * (2 * beta_fn(t)) ** 0.5

        # Backward SDE
        if self.use_backward_ode:
            self.diff_back = lambda t: 0.0
            self.drift_back = lambda x, t, *args, **kwargs: -(
                (s_deriv_fn(t) / s_fn(t)) * x
                - (s_fn(t) ** 2)
                * sigma_deriv_fn(t)
                * sigma_fn(t)
                * self.score(x, sigma_fn(t), *args, **kwargs)
            )
        else:
            self.drift_back = lambda x, t, *args, **kwargs: (
                sigma_deriv_fn(t) * sigma_fn(t) + beta_fn(t) * sigma_fn(t) ** 2
            ) * self.score(x, sigma_fn(t), *args, **kwargs)
            self.diff_back = self.diff_forw
        self._define_solvers()

    @torch.no_grad()
    def forward(
        self,
        latents: Optional[Tensor] = None,
        shape: Tuple[int, ...] = None,
        max_iter: int = 100,
    ):
        if latents is None:
            latents = (
                torch.randn(shape, device=device, generator=self.rng) * self.sigma_max
            )
        return self.backward_sde.sample(latents, timesteps=self.timesteps_fn(max_iter))


class PosteriorEDMSDE(EDMSDE):
    def __init__(self, prior: ScorePrior, data_fidelity: Callable, *args, **kwargs):
        super().__init__(prior=prior, *args, **kwargs)
        self.data_fidelity = data_fidelity

    def score(
        self,
        x: Tensor,
        sigma: Union[Tensor, float],
        y: Tensor,
        physics: Physics,
        rescale: bool = False,
    ):
        if rescale:
            x = (x + 1) * 0.5
            sigma_in = sigma * 0.5
        else:
            sigma_in = sigma
        score_prior = -self.prior.grad(x, sigma_in)
        if rescale:
            score_prior = score_prior * 2 - 1
        return score_prior - self.data_fidelity.grad(x, y, physics, sigma)

    torch.no_grad()

    def forward(
        self,
        y: Tensor,
        physics: Physics,
        x_init: Optional[Tensor] = None,
        max_iter: int = 100,
    ):
        if x_init is None:
            x_init = torch.randn_like(y) * self.sigma_max
        return self.backward_sde.sample(
            x_init, y, physics, timesteps=self.timesteps_fn(max_iter)
        )


if __name__ == "__main__":
    from edm import load_model
    import numpy as np
    from deepinv.utils.demo import load_url_image, get_image_url
    import deepinv as dinv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_model("edm-ffhq-64x64-uncond-ve.pkl").to(device)
    url = get_image_url("CBSD_0010.png")
    x = load_url_image(url=url, img_size=64, device=device)
    x_noisy = x + torch.randn_like(x) * 0.3
    dinv.utils.plot(
        [x, x_noisy, denoiser(x_noisy, 0.3)], titles=["sample", "y", "denoised"]
    )

    # denoiser = lambda x, t: model(x.to(torch.float32), t).to(torch.float64)
    # denoiser = dinv.models.DRUNet(device=device)
    prior = dinv.optim.prior.ScorePrior(denoiser=denoiser)

    # EDM generation
    sde = EDMSDE(prior=prior, use_backward_ode=False, name="ve", solver_name="Heun")
    x = sde(shape=(1, 3, 64, 64), max_iter=100)

    # Posterior EDM generation
    physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.5, device=device)
    noisy_data_fidelity = DPSDataFidelity(denoiser=denoiser)
    y = physics(x)
    posterior_sde = PosteriorEDMSDE(
        prior=prior,
        data_fidelity=noisy_data_fidelity,
        name="ve",
        use_backward_ode=True,
        solver_name="Heun",
    )
    posterior_sample = posterior_sde(y, physics, max_iter=100)

    # Plotting the samples
    # dinv.utils.plot([x], titles = ['sample'])
    dinv.utils.plot(
        [x, y, posterior_sample], titles=["sample", "y", "posterior_sample"]
    )
