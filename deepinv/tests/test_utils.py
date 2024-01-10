import deepinv
import torch
import pytest


@pytest.fixture
def tensorlist():
    x = torch.ones((1, 1, 2, 2))
    y = torch.ones((1, 1, 2, 2))
    x = deepinv.utils.TensorList([x, x])
    y = deepinv.utils.TensorList([y, y])
    return x, y


def test_tensordict_sum(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2)) * 2
    z1 = deepinv.utils.TensorList([z, z])
    z = x + y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_mul(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_div(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x / y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_sub(tensorlist):
    x, y = tensorlist
    z = torch.zeros((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x - y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_neg(tensorlist):
    x, y = tensorlist
    z = -torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = -x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_append(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z, z, z])
    z = x.append(y)
    assert (z1[0] == z[0]).all() and (z1[-1] == z[-1]).all()


def test_plot():
    x = torch.ones((1, 1, 2, 2))
    imgs = [x, x]
    deepinv.utils.plot(imgs, titles=["a", "b"])
    deepinv.utils.plot(x, titles="a")
    deepinv.utils.plot(imgs)


import torch.nn as nn
import torch


# OPTIMIZATION

from deepinv.utils.optimization import (
    NeuralIteration,
    GradientDescent,
    ProximalGradientDescent,
)


# Mock Physics Class for Testing
class MockPhysics:
    def __init__(self, device="cpu"):
        self.device = device

    def A(self, x):
        # Mock implementation of A
        return x

    def A_adjoint(self, y):
        # Mock implementation of A_adjoint
        return y


## Neural Iteration


def test_neural_iteration_initialization():
    model = NeuralIteration()
    # Pass multiple identical blocks to avoid the single block issue
    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    model.init(backbone_blocks, step_size=0.5, iterations=2)
    assert model.iterations == 2
    assert model.step_size.size() == torch.Size([2])
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 2  # Ensure there are 2 blocks


def test_neural_iteration_forward():
    model = NeuralIteration()
    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    model.init(backbone_blocks, iterations=2)
    physics = MockPhysics()
    y = torch.randn(10, 10)
    output = model.forward(y, physics)
    assert torch.equal(output, y)  # Assuming forward returns physics.A_adjoint(y)


## Gradient Descent

## Proximal Gradient Descent


# In the test fixture
@pytest.fixture
def setup_proximal_gradient_descent():
    physics = MockPhysics()
    backbone_blocks = [
        torch.nn.Linear(10, 10) for _ in range(1)
    ]  # List of nn.Module modules
    step_size = 1.0
    iterations = 1
    return ProximalGradientDescent(backbone_blocks, step_size, iterations), physics


# METRIC

from deepinv.utils.metric import cal_angle, cal_mse, cal_psnr, cal_psnr_complex, norm


def test_norm():
    a = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    expected_norm = torch.tensor([[[[5.4772]]]])
    assert torch.allclose(norm(a), expected_norm, atol=1e-4)


def test_cal_angle():
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    expected_normalized_angle = 0.5  # 90 degrees normalized (pi/2 radians / pi)
    assert cal_angle(a, b) == pytest.approx(expected_normalized_angle, rel=1e-3)


def test_cal_psnr():
    a = torch.ones((1, 1, 256, 256))
    b = torch.zeros((1, 1, 256, 256))
    max_pixel = 1.0
    expected_psnr = 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(1.0)))
    assert cal_psnr(a, b, max_pixel) == pytest.approx(expected_psnr.item(), rel=1e-3)


def test_cal_mse():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected_mse = 0.0
    assert cal_mse(a, b) == expected_mse


def test_cal_psnr_complex():
    a = torch.randn((1, 2, 10, 10))  # Simulated complex data
    b = torch.randn((1, 2, 10, 10))
    # The test will check if the function executes without errors
    # and returns a reasonable result, but cannot predict the exact value
    psnr_complex = cal_psnr_complex(a, b)
    assert psnr_complex > 0


# PARAMETERS

import numpy as np
import pytest
from deepinv.utils.parameters import get_DPIR_params, get_GSPnP_params


def test_get_DPIR_params():
    noise_level_img = 0.05
    lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(noise_level_img)

    assert lamb == pytest.approx(1 / 0.23)
    assert len(sigma_denoiser) == 8
    assert len(stepsize) == 8
    assert max_iter == 8
    assert all(s >= 0 for s in sigma_denoiser)
    assert all(s >= 0 for s in stepsize)


def test_get_GSPnP_params_deblur():
    problem = "deblur"
    noise_level_img = 0.05
    lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
        problem, noise_level_img
    )

    assert max_iter == 500
    assert sigma_denoiser == pytest.approx(1.8 * noise_level_img)
    assert lamb == pytest.approx(1 / 0.1)
    assert stepsize == 1.0


def test_get_GSPnP_params_super_resolution():
    problem = "super-resolution"
    noise_level_img = 0.05
    lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
        problem, noise_level_img
    )

    assert max_iter == 500
    assert sigma_denoiser == pytest.approx(2.0 * noise_level_img)
    assert lamb == pytest.approx(1 / 0.065)
    assert stepsize == 1.0


def test_get_GSPnP_params_inpaint():
    problem = "inpaint"
    noise_level_img = 0.05
    lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
        problem, noise_level_img
    )

    assert max_iter == 100
    assert sigma_denoiser == pytest.approx(10.0 / 255)
    assert lamb == pytest.approx(1 / 0.1)
    assert stepsize == 1.0


def test_get_GSPnP_params_invalid():
    with pytest.raises(ValueError):
        get_GSPnP_params("invalid_problem", 0.05)
