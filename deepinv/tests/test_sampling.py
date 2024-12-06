import pytest
import torch.nn
import numpy as np

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import ULA, SKRock, DiffPIR, DPS
import os

SAMPLING_ALGOS = ["DDRM", "ULA", "SKRock"]


def choose_algo(algo, likelihood, thresh_conv, sigma, sigma_prior):
    if algo == "ULA":
        out = ULA(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            thinning=1,
            step_size=0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "SKRock":
        out = SKRock(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            inner_iter=5,
            step_size=1 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
            verbose=True,
        )
    elif algo == "DDRM":
        diff = dinv.sampling.DDRM(
            denoiser=GaussianDenoiser(sigma_prior),
            eta=1,
            sigmas=np.linspace(1, 0, 100),
        )
        out = dinv.sampling.DiffusionSampler(diff, clip=(-100, 100), max_iter=500)
    else:
        raise Exception("The sampling algorithm doesnt exist")

    return out


class GaussianScore(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def grad(self, x, sigma):
        return x / self.sigma_prior2


class GaussianDenoiser(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def forward(self, x, sigma):
        return x / (1 + sigma**2 / self.sigma_prior2)


@pytest.mark.parametrize("algo", SAMPLING_ALGOS)
def test_sampling_algo(algo, imsize, device):
    test_sample = torch.ones((1, *imsize))

    sigma = 1
    sigma_prior = 1
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = 0.1  # for fast tests
    likelihood = L2(sigma=sigma)
    f = choose_algo(
        algo,
        likelihood,
        thresh_conv=convergence_crit,
        sigma=sigma,
        sigma_prior=sigma_prior,
    )

    xmean, xvar = f(y, physics, seed=0)

    tol = 5  # can be lowered?
    sigma2 = sigma**2
    sigma_prior2 = sigma_prior**2

    # the posterior of a gaussian likelihood with a gaussian prior is gaussian
    post_var = (sigma2 * sigma_prior2) / (sigma2 + sigma_prior2)
    post_mean = y / (1 + sigma2 / sigma_prior2)

    mean_ok = (
        torch.sum((xmean - post_mean).abs() / post_mean < tol)
        > np.prod(xmean.shape) / 2
    )

    var_ok = (
        torch.sum((xvar - post_var).abs() / post_var < tol) > np.prod(xvar.shape) / 2
    )

    assert f.mean_has_converged() and f.var_has_converged() and mean_ok and var_ok


def test_diffpir(device):
    from deepinv.models import DiffUNet

    x = torch.ones((1, 3, 32, 32)).to(device)

    sigma = 12.75 / 255.0  # noise level

    physics = dinv.physics.BlurFFT(
        img_size=(3, x.shape[-2], x.shape[-1]),
        filter=torch.ones((1, 1, 5, 5), device=device) / 25,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    y = physics(x)

    model = DiffUNet().to(device)
    likelihood = L2()

    algorithm = DiffPIR(model, likelihood, max_iter=5, verbose=False, device=device)

    out = algorithm(y, physics)
    assert out.shape == x.shape


def test_dps(device):
    from deepinv.models import DiffUNet

    x = torch.ones((1, 3, 32, 32)).to(device)

    sigma = 12.75 / 255.0  # noise level

    physics = dinv.physics.BlurFFT(
        img_size=(3, x.shape[-2], x.shape[-1]),
        filter=torch.ones((1, 1, 5, 5), device=device) / 25,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    y = physics(x)

    model = DiffUNet().to(device)
    likelihood = L2()

    algorithm = DPS(model, likelihood, max_iter=5, verbose=False, device=device)

    out = algorithm(y, physics)
    assert out.shape == x.shape


DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.mark.parametrize("device", DEVICES)
def test_sde(device):
    from deepinv.sampling.sde import VESDE
    from deepinv.models import NCSNpp, ADMUNet, EDMPrecond, DRUNet

    # Set up all denoisers
    denoisers = []
    rescales = []
    list_kwargs = []
    denoisers.append(
        EDMPrecond(model=NCSNpp.from_pretrained("edm-ffhq64-uncond-ve")).to(device)
    )
    rescales.append(False)
    list_kwargs.append(dict())

    denoisers.append(
        EDMPrecond(model=ADMUNet.from_pretrained("imagenet64-cond")).to(device)
    )
    rescales.append(False)
    list_kwargs.append(dict(class_labels=torch.eye(1000, device=device)[0:1]))

    denoisers.append(DRUNet(pretrained="download").to(device))
    rescales.append(True)
    list_kwargs.append(dict())

    # Set up the SDE
    sigma_max = 20
    sigma_min = 0.02
    num_steps = 20
    rng = torch.Generator(device)
    for denoiser, rescale, kwargs in zip(denoisers, rescales, list_kwargs):
        sde = VESDE(
            denoiser=denoiser,
            rescale=rescale,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            rng=rng,
            device=device,
            use_backward_ode=False,
        )

        # Test generation
        timesteps = np.linspace(0.001, 1, num_steps)[::-1]
        solution = sde(
            (1, 3, 64, 64),
            timesteps=timesteps,
            method="Euler",
            seed=10,
            full_trajectory=True,
            **kwargs,
        )
        x_init_1 = solution.trajectory[0]
        sample_1 = solution.sample

        assert solution.sample.shape == (1, 3, 64, 64)

        # Test reproducibility
        solution = sde(
            (1, 3, 64, 64),
            timesteps=timesteps,
            method="Euler",
            seed=10,
            full_trajectory=True,
            **kwargs,
        )
        x_init_2 = solution.trajectory[0]
        sample_2 = solution.sample
        # Test reproducibility
        assert torch.allclose(x_init_1, x_init_2, atol=1e-5, rtol=1e-5)
        assert torch.nn.functional.mse_loss(sample_1, sample_2, reduction="mean") < 1e-2
        # TODO: add better error checking
