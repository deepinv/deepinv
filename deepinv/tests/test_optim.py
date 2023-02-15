import pytest

import deepinv as dinv
from deepinv.pnp.denoiser import Denoiser
from deepinv.optim.data_fidelity import DataFidelity
from deepinv.pnp.pnp import PnP
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils import save_model, AverageMeter, ProgressMeter, get_timestamp, cal_psnr
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
def dummy_dataset(imsize):
    return DummyCircles(samples=1, imsize=imsize)


def test_denoiser(imsize, dummy_dataset, device):

    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Denoising(sigma=.2)  # 2. Set a physical experiment (here, denoising)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    backbone = dinv.models.TGV(reg=2, n_it_max=5000, crit=1e-5, verbose=True)
    model = dinv.models.ArtifactRemoval(backbone)

    x = model(y, physics)  # 3. Apply the model we want to test

    plot_debug = False

    if plot_debug:
        imgs = []
        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
        imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

        titles = ['Input', 'Output']
        num_im = 2
        plot_debug(imgs, shape=(1, num_im), titles=titles,
                   row_order=True, save_dir=None)

    assert model.backbone_net.has_converged


optim_algos = ['PGD', 'HQS', 'DRS']
# optim_algos = ['GD', 'ADMM']  # Test fails!
@pytest.mark.parametrize("pnp_algo", optim_algos)
def test_optim_algo(pnp_algo, imsize, dummy_dataset, device):

    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Blur(dinv.physics.blur.gaussian_blur(sigma=(2, .1), angle=45.), device=dinv.device)  # 2. Set a physical experiment (here, deblurring)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    denoiser_name = 'TGV'
    denoiser = Denoiser(denoiser_name=denoiser_name, device=dinv.device, n_it_max=1000)

    data_fidelity = DataFidelity(type='L2')
    # pnp_algo = 'PGD'
    sigma_denoiser = 1.
    stepsize = 1.
    max_iter = 10000

    pnp = PnP(denoiser=denoiser, sigma_denoiser=sigma_denoiser, algo_name=pnp_algo, data_fidelity=data_fidelity,
              max_iter=max_iter, crit_conv=1e-5, stepsize=stepsize, device=device, verbose=True)

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

    assert pnp.has_converged

