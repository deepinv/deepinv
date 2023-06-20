import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles
import torch


@pytest.fixture
def device():
    return dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w


def test_dip(imsize, device):
    torch.manual_seed(0)
    channels = 64
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.2))
    f = dinv.models.DeepImagePrior(
        generator=dinv.models.ConvDecoder(imsize, layers=3, channels=channels).to(
            device
        ),
        input_size=(channels, imsize[1], imsize[2]),
        iterations=30,
    )
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)
    mse_in = (y - x).pow(2).mean()
    x_net = f(y, physics)
    mse_out = (x_net - x).pow(2).mean()

    assert mse_out < mse_in
