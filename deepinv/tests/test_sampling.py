import pytest
import torch.nn

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import ULA, SKRock, DiffPIR
from deepinv.models import get_diffpir_model_defaults
import numpy as np


@pytest.fixture
def device():
    return dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imsize():
    h = 2
    w = 2
    c = 1
    return c, h, w


def choose_algo(algo, likelihood, thresh_conv, sigma, sigma_prior):
    if algo == "ULA":
        out = ULA(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            thinning=1,
            verbose=True,
            step_size=0.01 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            thresh_conv=thresh_conv,
            sigma=1,
        )
    elif algo == "SKRock":
        out = SKRock(
            GaussianScore(sigma_prior),
            likelihood,
            max_iter=500,
            verbose=True,
            thresh_conv=thresh_conv,
            inner_iter=5,
            step_size=1 / (1 / sigma**2 + 1 / sigma_prior**2),
            clip=(-100, 100),
            sigma=1,
        )
    elif algo == "DDRM":
        diff = dinv.sampling.DDRM(
            denoiser=GaussianDenoiser(sigma_prior),
            sigma_noise=sigma,
            eta=1,
            sigmas=np.linspace(1, 0, 100),
        )
        out = dinv.sampling.DiffusionSampler(diff, clip=(-100, 100), max_iter=500)
    else:
        raise Exception("The sampling algorithm doesnt exist")

    return out


sampling_algo = ["DDRM", "ULA", "SKRock"]


class GaussianScore(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def forward(self, x, sigma):
        return x / self.sigma_prior2


class GaussianDenoiser(torch.nn.Module):
    def __init__(self, sigma_prior):
        super().__init__()
        self.sigma_prior2 = sigma_prior**2

    def forward(self, x, sigma):
        return x / (1 + sigma**2 / self.sigma_prior2)


@pytest.mark.parametrize("algo", sampling_algo)
def test_sampling_algo(algo, imsize, device):
    test_sample = torch.ones((1, 1, 2, 2))

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


def test_DiffPIR(device):
    from deepinv.models import get_diffpir_model_defaults

    x = torch.ones((1, 3, 128, 128)).to(device)

    sigma = 12.75 / 255.0  # noise level

    physics = dinv.physics.BlurFFT(
        img_size=(3, x.shape[-2], x.shape[-1]),
        filter=torch.ones((1, 1, 5, 5), device=device) / 25,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=sigma),
    )

    y = physics(x)

    model = get_diffpir_model_defaults(device=device)
    likelihood = L2()

    algorithm = DiffPIR(
        model, sigma, likelihood, max_iter=5, verbose=False, device="cpu"
    )

    out = algorithm(y, physics)

    assert out.shape == x.shape
