import pytest
import torch
import deepinv as dinv


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 16
    w = 8
    c = 2
    return c, h, w


# Linear forward operators to test (make sure they appear in find_operator as well)
operators = [
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
]  #'CT'


def find_operator(name, img_size, device):
    r"""
    Chooses operator

    :param name: operator name
    :param img_size: size of the input signals
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "CS":
        p = dinv.physics.CompressedSensing(m=30, img_shape=img_size, device=device)
    elif name == "fastCS":
        p = dinv.physics.CompressedSensing(
            m=20, fast=True, channelwise=True, img_shape=img_size, device=device
        )
    elif name == "inpainting":
        p = dinv.physics.Inpainting(tensor_size=img_size, mask=0.5, device=device)
    elif name == "MRI":
        p = dinv.physics.MRI(mask=torch.ones(img_size[-2], img_size[-1]), device=device)
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1))
    elif name == "blind_deblur":
        p = dinv.physics.BlindBlur(kernel_size=3)
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_shape=img_size, device=device
        )
    elif name == "singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=False, img_shape=img_size, device=device
        )
    elif name == "deblur":
        p = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
        )
    elif name == "deblur_fft":
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.1, 0.5), angle=45.0),
            device=device,
        )
    elif name == "super_resolution":
        p = dinv.physics.Downsampling(img_size=img_size, factor=2)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p


@pytest.mark.parametrize("name", operators)
def test_operators_adjointness(name, imsize, device):
    r"""
    Tests if a linear forward operator has a well defined adjoint.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts adjointness
    """
    physics = find_operator(name, imsize, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    assert physics.adjointness_test(x).abs() < 1e-3


@pytest.mark.parametrize("name", operators)
def test_operators_norm(name, imsize, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.5,1.5)
    """
    physics = find_operator(name, imsize, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    norm = physics.compute_norm(x)
    assert 1.5 > norm > 0.5


@pytest.mark.parametrize("name", operators)
def test_pseudo_inverse(name, imsize, device):
    r"""
    Tests if a linear physics operator has a well defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.5,1.5)
    """
    physics = find_operator(name, imsize, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)

    r = physics.A_adjoint(physics.A(x))
    y = physics.A(r)
    error = (physics.A_dagger(y) - r).flatten().mean().abs()
    assert error < 0.01
