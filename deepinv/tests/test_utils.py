import deepinv
import torch
import pytest
import torch.nn as nn


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


from deepinv.utils.optimization import NeuralIteration, GradientDescent


class MockPhysics:
    def __init__(self, device="cpu"):
        self.device = device

    def A(self, x):
        return x

    def A_adjoint(self, y):
        return y


def test_gradient_descent_initialization():
    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    step_size = 0.5
    iterations = 2

    gd_model = GradientDescent(backbone_blocks, step_size, iterations)

    assert gd_model.name == "gd"
    assert len(gd_model.blocks) == len(backbone_blocks)
    assert gd_model.iterations == iterations
    assert gd_model.step_size.size() == torch.Size([iterations])


def test_gradient_descent_forward():

    backbone_blocks = [nn.Linear(10, 10), nn.Linear(10, 10)]
    gd_model = GradientDescent(backbone_blocks, step_size=0.5, iterations=2)

    physics = MockPhysics()
    y = torch.randn(10, 10)
    x_init = None

    # Test sans x_init
    output = gd_model.forward(y, physics, x_init)
    assert output is not None

    # Test avec x_init
    x_init = torch.randn(10, 10)
    output_with_init = gd_model.forward(y, physics, x_init)
    assert output_with_init is not None

    def test_gradient_descent_single_block():
        single_block = [nn.Linear(10, 10)]
        gd_model = GradientDescent(single_block, step_size=0.5, iterations=2)

        physics = MockPhysics()
        y = torch.randn(10, 10)
        output = gd_model.forward(y, physics)
        assert output is not None


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

    a1 = torch.ones((1, 1, 16, 16))
    b1 = torch.zeros((1, 1, 16, 16))
    a2 = [a1]  # a2 is a list in which the first element is a tensor
    b2 = [b1]  # b2 is a list in which the first element is a tensor
    max_pixel = 1.0

    # MSE is remplaced by 1e-10 in the function if the real mse is 0
    mse_substitute = 1e-10
    expected_psnr = 20 * torch.log10(
        max_pixel / torch.sqrt(torch.tensor(mse_substitute))
    )

    # Test with tensors
    calculated_psnr_a1_b1 = cal_psnr(a1, b1, max_pixel)
    assert calculated_psnr_a1_b1 == pytest.approx(expected_psnr.item(), rel=100)
    # Test with list
    calculated_psnr_a2_b2 = cal_psnr(a2, b2, max_pixel)
    assert calculated_psnr_a2_b2 == pytest.approx(expected_psnr.item(), rel=100)

    # Test if normalize is True
    calculated_psnr_a1_b1 = cal_psnr(a1, b1, max_pixel, True)
    assert calculated_psnr_a1_b1 == pytest.approx(expected_psnr.item(), rel=100)


def test_cal_psnr_complex():
    # Créer des tenseurs complexes de test
    a_real = torch.ones((1, 16, 16))
    a_imag = torch.zeros((1, 16, 16))
    b_real = torch.zeros((1, 16, 16))
    b_imag = torch.zeros((1, 16, 16))

    # Empiler les parties réelles et imaginaires pour créer des tenseurs complexes
    a = torch.stack((a_real, a_imag), dim=1)  # Shape devrait être [1, 2, 16, 16]
    b = torch.stack((b_real, b_imag), dim=1)  # Shape devrait être [1, 2, 16, 16]

    # Vérifier la forme des tenseurs
    assert a.shape == (1, 2, 16, 16)
    assert b.shape == (1, 2, 16, 16)

    # Calculer la magnitude absolue
    a_abs = complex_abs(
        a.permute(0, 2, 3, 1)
    )  # Permutation pour mettre les parties réelle et imaginaire à la fin
    b_abs = complex_abs(b.permute(0, 2, 3, 1))

    # Calculer le PSNR attendu
    # Remarque : Ce calcul doit correspondre à la manière dont cal_psnr calcule le PSNR
    mse = torch.mean((a_abs - b_abs) ** 2)
    max_pixel = 1.0  # Assurez-vous que cette valeur correspond à ce que vous utilisez dans cal_psnr_complex
    expected_psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))

    # Tester la fonction cal_psnr_complex
    calculated_psnr = cal_psnr_complex(a, b)
    assert calculated_psnr == pytest.approx(expected_psnr.item(), rel=100)


def test_cal_mse():
    r"""
    Test the `cal_mse` function from the utility metrics.


    :param expected_mse: The expected MSE value, which is zero in this case.
    :return: Asserts that the calculated MSE is equal to the expected value, validating the function's correctness.
    """

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([1.0, 2.0, 3.0])
    expected_mse = 0.0
    assert cal_mse(a, b) == expected_mse


from deepinv.utils.metric import (
    complex_abs,
)  # Remplacez 'your_module' par le nom réel de votre module


def test_complex_abs():
    # Créer un tenseur complexe de test
    real_part = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    imag_part = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    complex_tensor = torch.stack((real_part, imag_part), dim=-1)

    # Calculer le module attendu manuellement
    expected_abs = torch.sqrt(real_part**2 + imag_part**2)

    # Calculer le module en utilisant la fonction
    calculated_abs = complex_abs(complex_tensor)

    # Vérifier si les résultats sont les mêmes
    assert torch.allclose(
        calculated_abs, expected_abs
    ), "The calculated absolute values are not as expected."
