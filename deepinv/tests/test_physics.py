import pytest
import torch
import numpy as np

import deepinv as dinv


# Linear forward operators to test (make sure they appear in find_operator as well)
OPERATORS = [
    "CS",
    "fastCS",
    "inpainting",
    "denoising",
    "deblur_fft",
    "deblur",
    "singlepixel",
    "fast_singlepixel",
    "super_resolution",
    "MRI",
    "pansharpen",
]
NONLINEAR_OPERATORS = ["haze", "blind_deblur", "lidar"]

NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "Neighbor2Neighbor",
]


def find_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    img_size = (3, 16, 8)
    norm = 1
    if name == "CS":
        m = 30
        p = dinv.physics.CompressedSensing(m=m, img_shape=img_size, device=device)
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 0.75  # Marcenko-Pastur law, second term is a small n correction
    elif name == "fastCS":
        p = dinv.physics.CompressedSensing(
            m=20, fast=True, channelwise=True, img_shape=img_size, device=device
        )
    elif name == "inpainting":
        p = dinv.physics.Inpainting(tensor_size=img_size, mask=0.5, device=device)
    elif name == "MRI":
        img_size = (2, 16, 8)
        p = dinv.physics.MRI(mask=torch.ones(img_size[-2], img_size[-1]), device=device)
    elif name == "Tomography":
        img_size = (1, 16, 16)
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=img_size[-1], device=device
        )
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1))
    elif name == "pansharpen":
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(img_size=img_size, device=device)
        norm = 0.4
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_shape=img_size, device=device
        )
    elif name == "singlepixel":
        m = 20
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=False, img_shape=img_size, device=device
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 3.7  # Marcenko-Pastur law, second term is a small n correction
    elif name == "deblur":
        img_size = (3, 17, 19)
        p = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
        )
    elif name == "deblur_fft":
        img_size = (3, 17, 19)
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.1, 0.5), angle=45.0),
            device=device,
        )
    elif name == "super_resolution":
        img_size = (1, 32, 32)
        factor = 2
        norm = 1 / factor**2
        p = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size, norm


def find_nonlinear_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "blind_deblur":
        x = dinv.utils.TensorList(
            [
                torch.randn(1, 3, 16, 16, device=device),
                torch.randn(1, 1, 3, 3, device=device),
            ]
        )
        p = dinv.physics.BlindBlur(kernel_size=3)
    elif name == "haze":
        x = dinv.utils.TensorList(
            [
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, device=device),
            ]
        )
        p = dinv.physics.Haze()
    elif name == "lidar":
        x = torch.rand(1, 3, 16, 16, device=device)
        p = dinv.physics.SinglePhotonLidar(device=device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, x


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_adjointness(name, device):
    r"""
    Tests if a linear forward operator has a well defined adjoint.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts adjointness
    """
    physics, imsize, _ = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    error = physics.adjointness_test(x).abs()
    assert error < 1e-3


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_norm(name, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.8,1.2)
    """
    if name == "singlepixel" or name == "CS":
        device = torch.device("cpu")

    torch.manual_seed(0)
    physics, imsize, norm_ref = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    norm = physics.compute_norm(x)
    assert torch.abs(norm - norm_ref) < 0.2


@pytest.mark.parametrize("name", NONLINEAR_OPERATORS)
def test_nonlinear_operators(name, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts correct shapes
    """
    physics, x = find_nonlinear_operator(name, device)
    y = physics(x)
    xhat = physics.A_dagger(y)
    assert x.shape == xhat.shape


@pytest.mark.parametrize("name", OPERATORS)
def test_pseudo_inverse(name, device):
    r"""
    Tests if a linear physics operator has a well defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize, _ = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)

    r = physics.A_adjoint(physics.A(x))
    y = physics.A(r)
    error = (physics.A_dagger(y) - r).flatten().mean().abs()
    assert error < 0.01


def test_MRI(device):
    r"""
    Test MRI function

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics = dinv.physics.MRI(mask=None, device=device, acceleration_factor=4)
    x = torch.randn((2, 320, 320), device=device).unsqueeze(0)
    x2 = physics.A_adjoint(physics.A(x))
    assert x2.shape == x.shape

    physics = dinv.physics.MRI(mask=None, device=device, acceleration_factor=8, seed=0)
    y1 = physics.A(x)
    physics.reset()
    y2 = physics.A(x)
    if y1.shape == y2.shape:
        error = (y1.abs() - y2.abs()).flatten().mean().abs()
        assert error > 0.0


def choose_noise(noise_type):
    gain = 0.1
    sigma = 0.1
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise(
            sigma=sigma
        )  # This is equivalent to GaussianNoise when sigma is fixed
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "Neighbor2Neighbor":
        noise_model = dinv.physics.PoissonNoise(gain)
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("noise_type", NOISES)
def test_noise(device, noise_type):
    r"""
    Tests noise models.
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = choose_noise(noise_type)
    x = torch.ones((1, 12, 7), device=device).unsqueeze(0)

    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    assert y1.shape == x.shape

    if noise_type == "UniformGaussian":
        physics.reset()
        y2 = physics(x)
        error = (y1 - y2).flatten().abs().sum()
        assert error > 0.0


def test_reset_noise(device):
    r"""
    Tests that the reset function works.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is > 0
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = dinv.physics.UniformGaussianNoise(
        sigma=None
    )  # Should be 20/255 (to check)
    x = torch.ones((1, 12, 7), device=device).unsqueeze(0)

    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    physics.reset()
    y2 = physics(x)
    error = (y1 - y2).flatten().abs().sum()
    assert error > 0.0


def test_tomography(device):
    r"""
    Tests tomography operator which does not have a numerically precise adjoint.

    :param device: (torch.device) cpu or cuda:x
    """
    for circle in [True, False]:
        imsize = (1, 16, 16)
        physics = dinv.physics.Tomography(
            img_width=imsize[-1], angles=imsize[-1], device=device, circle=circle
        )

        x = torch.randn(imsize, device=device).unsqueeze(0)
        r = physics.A_adjoint(physics.A(x))
        y = physics.A(r)
        error = (physics.A_dagger(y) - r).flatten().mean().abs()
        assert error < 0.2
