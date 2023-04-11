import pytest

import deepinv as dinv
from deepinv.models.denoiser import ScoreDenoiser
from deepinv.optim.data_fidelity import L2
from deepinv.sampling import ULA, SKRock
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.plotting import plot_debug, torch2cpu

from torch.utils.data import DataLoader


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w


# TODO: use a DummyCircle as dataset and check convergence of optim algorithms (maybe with TV denoiser)
@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=1, imsize=imsize)

def choose_algo(algo, prior, likelihood, stepsize, thresh_conv, sigma):
    if algo == 'ULA':
        out = ULA(prior, likelihood, max_iter=1000,  verbose=True,
               alpha=.9, step_size=.01*stepsize, clip=(-1, 2), thresh_conv=thresh_conv, sigma=sigma)
    elif algo == 'SKRock':
        out = SKRock(prior, likelihood, max_iter=500, verbose=True, thresh_conv=thresh_conv,
                          alpha=.9, step_size=stepsize, clip=(-1, 2), sigma=sigma)
    else:
        raise Exception('The sampling algorithm doesnt exist')

    return out

sampling_algo = ['ULA', 'SKRock']

@pytest.mark.parametrize("algo", sampling_algo)
def test_sampling_algo(algo, imsize, dummy_dataset, device):

    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader)).to(device)

    sigma = .1
    physics = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)  # 2. Set a physical experiment (here, deblurring)
    physics.noise_model = dinv.physics.GaussianNoise(sigma)
    y = physics(test_sample)

    convergence_crit = .1 # for fast tests
    model_spec = {'name': 'waveletprior', 'args': {'wv': 'db8', 'level': 3, 'device': device}}
    stepsize = (sigma**2)
    likelihood = L2(sigma=sigma)
    prior = ScoreDenoiser(model_spec=model_spec)
    sigma_denoiser = 2/255.
    f = choose_algo(algo, prior, likelihood, stepsize=stepsize, thresh_conv=convergence_crit, sigma=sigma_denoiser)

    xmean, xvar = f(y, physics)

    assert f.mean_has_converged() and f.var_has_converged()
