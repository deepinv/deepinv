import pytest

import deepinv as dinv
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import *
from deepinv.optim.optimizers import *
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


@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=1, imsize=imsize)


def test_denoiser(imsize, dummy_dataset, device):

    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Denoising()  # 2. Set a physical experiment (here, denoising)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    ths = 2.

    model_spec = {'name': 'tgv', 'args': {'n_it_max': 5000, 'verbose': True, 'crit': 1e-4}}
    model = Denoiser(model_spec)

    x = model(y, ths)  # 3. Apply the model we want to test

    plot = False

    if plot:
        imgs = []
        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
        imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

        titles = ['Input', 'Output']
        num_im = 2
        plot_debug(imgs, shape=(1, num_im), titles=titles,
                   row_order=True, save_dir=None)

    assert model.denoiser.has_converged


optim_algos = ['PGD', 'HQS', 'DRS', 'ADMM', 'PD']
# optim_algos = ['PGD']
# optim_algos = ['GD']  # To implement
@pytest.mark.parametrize("pnp_algo", optim_algos)
def test_optim_algo(pnp_algo, imsize, dummy_dataset, device):

    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader)).to(device)

    physics = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)  # 2. Set a physical experiment (here, deblurring)
    y = physics(test_sample)
    max_iter = 1000
    sigma_denoiser = 0.1
    stepsize = 1.
    lamb = 1.

    data_fidelity = L2()

    model_spec = {'name': 'waveletprior', 'args': {'wv': 'db8', 'level': 3, 'device': device}}
    prior = {'prox_g': Denoiser(model_spec)}
    params_algo={'stepsize': stepsize, 'g_param': sigma_denoiser, 'lambda': lamb}
    pnp = Optim(pnp_algo, prior=prior, data_fidelity=data_fidelity, max_iter=max_iter, thres_conv=1e-4, verbose=True, params_algo=params_algo)

    x = pnp(y, physics)

    plot = False
    if plot:
        imgs = []
        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
        imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

        titles = ['Input', 'Output']
        num_im = 2
        plot_debug(imgs, shape=(1, num_im), titles=titles,
                   row_order=True, save_dir=None)

    assert pnp.has_converged()
