import pytest
import torch
import numpy as np
from deepinv.physics.forward import adjoint_function
import deepinv as dinv

# Noise model which has a `rng` attribute
NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "LogPoisson",
]
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

DTYPES = [torch.float32, torch.float64]


def choose_noise(noise_type, rng):
    gain = 0.1
    sigma = 0.1
    mu = 0.2
    N0 = 1024.0
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain, rng=rng)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma, rng=rng)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise(rng=rng)
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain, rng=rng)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain, rng=rng)
    elif noise_type == "LogPoisson":
        noise_model = dinv.physics.LogPoissonNoise(N0, mu, rng=rng)
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_concatenation(device, rng, dtype):
    imsize = (1, 3, 7, 16)
    for name_1 in NOISES:
        for name_2 in NOISES:
            noise_model_1 = choose_noise(name_1, rng=rng)
            noise_model_2 = choose_noise(name_2, rng=rng)
            noise_model = noise_model_1 * noise_model_2
            x = torch.rand(imsize, device=device, dtype=dtype)
            y = noise_model(x)
            assert y.shape == torch.Size(imsize)


@pytest.mark.parametrize("name", NOISES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rng(name, device, rng, dtype):
    imsize = (1, 3, 7, 16)
    x = torch.rand(imsize, device=device, dtype=dtype)
    noise_model = choose_noise(name, rng=rng)
    y_1 = noise_model(x, seed=0)
    y_2 = noise_model(x, seed=1)
    y_3 = noise_model(x, seed=0)
    assert torch.allclose(y_1, y_3)
    assert not torch.allclose(y_1, y_2)
