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


from deepinv.utils.optimization import (
    NeuralIteration,
)


class MockPhysics:
    def __init__(self, device="cpu"):
        self.device = device

    def A(self, x):
        return x

    def A_adjoint(self, y):
        return y


def test_neural_iteration_initialization():
    r"""
    Test the initialization of the NeuralIteration model.

    :param model: An instance of the NeuralIteration model.
    :param backbone_blocks: A list of neural network blocks used in the model.
    :param step_size: The step size for each iteration.
    :param iterations: The number of iterations the model should run.
    :return: Asserts the correct number of iterations, the correct size of the step size array, and the correct type and number of blocks in the model.
    """

    model = NeuralIteration()
    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    model.init(backbone_blocks, step_size=0.5, iterations=2)
    assert model.iterations == 2
    assert model.step_size.size() == torch.Size([2])
    assert isinstance(model.blocks, nn.ModuleList)
    assert len(model.blocks) == 2


def test_neural_iteration_forward():
    r"""
    Test the forward pass of the NeuralIteration model.

    :param model: An instance of the NeuralIteration model initialized with backbone blocks.
    :param physics: A mock physics model used for the forward pass.
    :param y: A sample input tensor for the forward pass.
    :return: Asserts that the output of the forward pass is as expected, matching the input processed by the mock physics model's adjoint operator.
    """

    model = NeuralIteration()
    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    model.init(backbone_blocks, iterations=2)
    physics = MockPhysics()
    y = torch.randn(10, 10)
    output = model.forward(y, physics)
    assert torch.equal(output, y)


from deepinv.utils.metric import cal_angle, cal_mse, cal_psnr, cal_psnr_complex, norm


def test_norm():
    r"""
    Test the `norm` function from the utility metrics.

    :param a: A sample input tensor.
    :return: Asserts that the calculated norm is close to the expected value within a specified tolerance.
    """

    a = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    expected_norm = torch.tensor([[[[5.4772]]]])
    assert torch.allclose(norm(a), expected_norm, atol=1e-4)


def test_cal_angle():
    r"""
    Test the `cal_angle` function from the utility metrics.


    :param a: The first input vector.
    :param b: The second input vector.
    :param expected_normalized_angle: The expected normalized angle between the two vectors.
    :return: Asserts that the calculated angle matches the expected value within a given relative tolerance.
    """

    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    expected_normalized_angle = 0.5  # 90 degrees normalized (pi/2 radians / pi)
    assert cal_angle(a, b) == pytest.approx(expected_normalized_angle, rel=1e-3)


def test_cal_psnr():
    r"""
    Test the `cal_psnr` function from the utility metrics.


    :param a: The original image tensor.
    :param b: The modified image tensor.
    :param max_pixel: The maximum pixel value in the image.
    :param expected_psnr: The expected PSNR value for the given tensors.
    :return: Asserts that the calculated PSNR matches the expected value within a specified tolerance.
    """

    a = torch.ones((1, 1, 16, 16))
    b = torch.zeros((1, 1, 16, 16))
    max_pixel = 1.0
    expected_psnr = 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(1.0)))
    assert cal_psnr(a, b, max_pixel) == pytest.approx(expected_psnr.item(), rel=1e-3)


def test_cal_mse():
    r"""
    Test the `cal_mse` function from the utility metrics.


    :param a: The first tensor for MSE calculation.
    :param b: The second tensor for MSE calculation, identical to the first in this test.
    :param expected_mse: The expected MSE value, which is zero in this case.
    :return: Asserts that the calculated MSE is equal to the expected value, validating the function's correctness.
    """

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected_mse = 0.0
    assert cal_mse(a, b) == expected_mse
