import pytest
import deepinv as dinv
import torch


@pytest.fixture
def device():
    return dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imsize():
    h = 37
    w = 31
    c = 3
    return c, h, w


model_list = [
    "unet",
    "drunet",
    "dncnn",
    "waveletprior",
    "waveletdict",
    "tgv",
    "median",
    "autoencoder",
    "gsdrunet",
]


def choose_denoiser(name, imsize):
    if name == "unet":
        out = dinv.models.UNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "drunet":
        out = dinv.models.DRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "gsdrunet":
        out = dinv.models.GSDRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "dncnn":
        out = dinv.models.DnCNN(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "waveletprior":
        out = dinv.models.WaveletPrior()
    elif name == "waveletdict":
        out = dinv.models.WaveletDict()
    elif name == "tgv":
        out = dinv.models.TGV(n_it_max=10)
    elif name == "median":
        out = dinv.models.MedianFilter()
    elif name == "autoencoder":
        out = dinv.models.AutoEncoder(dim_input=imsize[0] * imsize[1] * imsize[2])
    else:
        raise Exception("Unknown denoiser")

    return out


@pytest.mark.parametrize("denoiser", model_list)
def test_denoiser(imsize, device, denoiser):
    torch.manual_seed(0)
    sigma = 0.2
    physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(sigma))
    x = torch.ones(imsize, device=device).unsqueeze(0)
    y = physics(x)

    f = choose_denoiser(denoiser, imsize).to(device)

    x_hat = f(y, sigma)

    assert x_hat.shape == x.shape


def test_dip(imsize, device):
    device = 'cpu'
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
