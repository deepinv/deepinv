from types import SimpleNamespace

import pytest
import torch

import deepinv as dinv
from deepinv.loss.codesign import BinaryRegularization, CodesignRegularization


def make_physics(matrix):
    """Create the smallest physics-like object needed by the regularizer."""
    return SimpleNamespace(_A=matrix)


def test_binary_regularization_matches_penalty():
    """The implementation must match the binary penalty from the papers."""
    m = 4
    weight = 2.0
    target = 1.0 / (m**0.5)
    matrix = torch.tensor(
        [[-target, 0.0, target, 1.0], [0.25, -0.75, 0.5, -0.5]],
        dtype=torch.float64,
    )
    physics = make_physics(matrix)
    regularization = BinaryRegularization(m=m, weight=weight)

    value = regularization(physics=physics)
    expected = weight * ((matrix - target).square() * (matrix + target).square()).mean(
        dim=1
    )

    assert torch.allclose(value, expected)


def test_binary_regularization_is_zero_at_binary_targets():
    """Both positive and negative target values must be minimizers."""
    m = 4
    target = 1.0 / (m**0.5)
    matrix = torch.tensor([[-target, target, -target, target]])

    value = BinaryRegularization(m=m)(physics=make_physics(matrix))

    assert torch.allclose(value, torch.zeros_like(value))


def test_binary_regularization_has_the_expected_gradient():
    """The regularizer must provide a finite gradient for the sensing matrix."""
    m = 4
    weight = 3.0
    target = 1.0 / (m**0.5)
    matrix = torch.tensor([[0.0, 0.25, -0.75, 1.0]], requires_grad=True)
    regularization = BinaryRegularization(m=m, weight=weight)

    value = regularization(physics=make_physics(matrix))
    value.sum().backward()

    expected_gradient = (
        weight
        * 4.0
        * matrix.detach()
        * (matrix.detach().square() - target**2)
        / matrix.shape[1]
    )

    assert matrix.grad is not None
    assert torch.isfinite(matrix.grad).all()
    assert torch.allclose(matrix.grad, expected_gradient)


def test_binary_regularization_weight_schedule_and_clipping():
    """The learning schedule increases the weight and respects max_weight."""
    matrix = torch.ones(1, 4)
    physics = make_physics(matrix)
    regularization = BinaryRegularization(
        m=4,
        weight=1.0,
        weight_increase_factor=2.0,
        max_weight=3.0,
    )

    regularization(physics=physics)
    assert regularization.weight == 2.0

    regularization(physics=physics)
    assert regularization.weight == 3.0

    regularization(physics=physics)
    assert regularization.weight == 3.0


def test_codesign_regularization_requires_physics():
    regularization = CodesignRegularization()

    with pytest.raises(ValueError, match="requires a physics operator"):
        regularization()


def test_codesign_regularization_base_is_abstract_in_practice():
    regularization = CodesignRegularization()

    with pytest.raises(NotImplementedError):
        regularization(physics=make_physics(torch.zeros(1, 4)))


def test_binary_regularization_validates_constructor_arguments():
    with pytest.raises(ValueError, match="m must be a positive integer"):
        BinaryRegularization(m=0)

    with pytest.raises(ValueError, match="parameter_name"):
        BinaryRegularization(m=4, parameter_name="")

    with pytest.raises(ValueError, match="weight must be non-negative"):
        BinaryRegularization(m=4, weight=-1.0)

    with pytest.raises(ValueError, match="weight_increase_factor"):
        BinaryRegularization(m=4, weight_increase_factor=-1.0)

    with pytest.raises(ValueError, match="max_weight"):
        BinaryRegularization(m=4, max_weight=-1.0)


def test_binary_regularization_validates_physics_parameter():
    regularization = BinaryRegularization(m=4)

    with pytest.raises(AttributeError, match="has no parameter '_A'"):
        regularization(physics=SimpleNamespace())

    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        regularization(physics=SimpleNamespace(_A=[1.0, -1.0]))


def test_binary_regularization_is_publicly_exported():
    assert dinv.loss.BinaryRegularization is BinaryRegularization
    assert dinv.loss.CodesignRegularization is CodesignRegularization
