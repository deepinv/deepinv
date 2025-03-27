import pytest
import torch.nn
import numpy as np

import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import ULA, SKRock, DiffPIR, DPS

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


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS"])
def test_algo(name_algo, device):
    test_sample = torch.ones((1, 3, 64, 64), device=device)

    sigma = 1
    physics = dinv.physics.Denoising(device=device)
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    likelihood = L2(sigma=sigma)

    if name_algo == "DiffPIR":
        f = DiffPIR(
            dinv.models.DiffUNet().to(device),
            likelihood,
            max_iter=5,
            verbose=False,
            device=device,
        )
    elif name_algo == "DPS":
        f = DPS(
            dinv.models.DiffUNet().to(device),
            likelihood,
            max_iter=5,
            verbose=False,
            device=device,
        )
    else:
        raise Exception("The sampling algorithm doesn't exist")

    x = f(y, physics)

    assert x.shape == test_sample.shape


@pytest.mark.parametrize("name_algo", ["DiffPIR", "DPS"])
def test_algo_inpaint(name_algo, device):
    from deepinv.models import DiffUNet

    x = torch.ones((1, 3, 32, 32)).to(device) / 2.0
    x[:, 0, ...] = 0  # create a colored image

    torch.manual_seed(10)

    mask = torch.ones_like(x)
    mask[:, :, 10:20, 10:20] = 0

    physics = dinv.physics.Inpainting(mask=mask, tensor_size=x.shape[1:], device=device)

    y = physics(x)

    model = DiffUNet().to(device)
    likelihood = L2()

    if name_algo == "DiffPIR":
        algorithm = DiffPIR(
            model, likelihood, max_iter=20, verbose=False, device=device, sigma=0.01
        )
    elif name_algo == "DPS":
        algorithm = DPS(model, likelihood, max_iter=100, verbose=False, device=device)

    with torch.no_grad():
        out = algorithm(y, physics)

    assert out.shape == x.shape

    mean_crop = out[:, :, 10:20, 10:20].flatten().mean()

    mask = mask.bool()
    masked_out = out[mask]
    mean_outside_crop = masked_out.mean()

    masked_target = x[mask]
    mean_target_masked = masked_target.mean()
    mean_target_inmask = 1 / 3.0

    assert (mean_target_inmask - mean_crop).abs() < 0.2
    assert (mean_target_masked - mean_outside_crop).abs() < 0.01


def test_sde(device):
    from deepinv.sampling import (
        VarianceExplodingDiffusion,
        PosteriorDiffusion,
        DPSDataFidelity,
        EulerSolver,
        HeunSolver,
    )
    from deepinv.models import NCSNpp, ADMUNet, EDMPrecond, DRUNet

    # Set up all denoisers
    denoisers = []
    rescales = []
    list_kwargs = []
    denoisers.append(EDMPrecond(model=NCSNpp(pretrained="download")).to(device))
    rescales.append(False)
    list_kwargs.append(dict())

    denoisers.append(EDMPrecond(model=ADMUNet(pretrained="download")).to(device))
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

    # Set up solvers
    timesteps = np.linspace(0.001, 1, num_steps)[::-1]
    solvers = [
        EulerSolver(timesteps=timesteps, rng=rng),
        HeunSolver(timesteps=timesteps, rng=rng),
    ]
    for denoiser, rescale, kwargs in zip(denoisers, rescales, list_kwargs):
        for solver in solvers:
            sde = VarianceExplodingDiffusion(
                denoiser=denoiser,
                rescale=rescale,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                solver=solver,
                device=device,
            )

            # Test generation
            sample_1, trajectory = sde.sample(
                (1, 3, 64, 64),
                seed=10,
                get_trajectory=True,
                **kwargs,
            )
            x_init_1 = trajectory[0]

            assert sample_1.shape == (1, 3, 64, 64)

            # Test reproducibility
            sample_2, trajectory = sde.sample(
                (1, 3, 64, 64),
                seed=10,
                get_trajectory=True,
                **kwargs,
            )
            x_init_2 = trajectory[0]
            # Test reproducibility
            assert torch.allclose(x_init_1, x_init_2, atol=1e-5, rtol=1e-5)
            assert (
                torch.nn.functional.mse_loss(sample_1, sample_2, reduction="mean")
                < 1e-2
            )

    # Test posterior sampling
    sde = VarianceExplodingDiffusion(
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        device=device,
    )
    posterior = PosteriorDiffusion(
        data_fidelity=DPSDataFidelity(denoiser=denoisers[0]),
        sde=sde,
        denoiser=denoisers[0],
        solver=solvers[0],
        rescale=rescales[0],
        dtype=torch.float64,
        device=device,
    )
    x = sample_2
    physics = dinv.physics.Inpainting(tensor_size=x.shape[1:], mask=0.5, device=device)
    y = physics(x)

    x_hat = posterior.forward(
        y,
        physics,
        x_init=(1, 3, 64, 64),
        seed=10,
    )

    assert x_hat.shape == (1, 3, 64, 64)
