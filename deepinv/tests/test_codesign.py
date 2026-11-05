from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO

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


def test_codesign_trainer_updates_physics_and_reconstructor(device):
    """Online joint training must keep working after the first matrix update."""
    physics = dinv.physics.CompressedSensing(
        m=4, img_size=(1, 4, 4), device=device, trainable=True
    )
    physics.update(_A=physics._A * 0.5)
    model = dinv.models.ArtifactRemoval(
        dinv.models.DnCNN(
            in_channels=1, out_channels=1, depth=2, nf=4, pretrained=None, device=device
        ),
        mode="adjoint",
        device=device,
    )
    parameters = [physics._A, *model.parameters()]
    initial_parameters = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.SGD(parameters, lr=1e-2)
    images = torch.ones(2, 1, 4, 4)
    measurements = physics(images.to(device))
    expected_adjoint = (measurements @ physics._A).reshape_as(images)
    assert torch.allclose(physics.A_adjoint(measurements), expected_adjoint)
    trainer = dinv.Trainer(
        model=model,
        physics=physics,
        optimizer=optimizer,
        train_dataloader=torch.utils.data.DataLoader(
            [{"x": image} for image in images], batch_size=1
        ),
        online_measurements=True,
        epochs=1,
        losses=[dinv.loss.SupLoss(), BinaryRegularization(m=4)],
        metrics=[],
        device=device,
        save_path=None,
        plot_images=False,
        non_blocking_transfers=False,
        verbose=False,
        show_progress_bar=False,
    )

    with patch("torch.linalg.pinv", side_effect=AssertionError("Unnecessary inverse")):
        trainer.train()

    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert not torch.equal(physics._A, initial_parameters[0])
    assert any(
        not torch.equal(parameter, initial)
        for parameter, initial in zip(
            parameters[1:], initial_parameters[1:], strict=True
        )
    )


@pytest.mark.parametrize("channelwise", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_compressed_sensing_legacy_checkpoint_and_cached_operators(
    device, channelwise, dtype
):
    physics = dinv.physics.CompressedSensing(
        m=4, img_size=(2, 4, 4), channelwise=channelwise, dtype=dtype, device=device
    )
    matrix = physics._A * 0.5
    # The exact buffer names written by CompressedSensing on main.
    legacy_state = {
        "initial_random_state": physics.initial_random_state.clone(),
        "_A": matrix,
        "_A_dagger": torch.linalg.pinv(matrix),
        "_A_adjoint": matrix.conj().T,
    }
    physics.load_state_dict(legacy_state, strict=True)
    assert set(physics.state_dict()) == set(legacy_state)
    assert not physics._A.requires_grad
    trainable_physics = dinv.physics.CompressedSensing(
        m=4,
        img_size=(2, 4, 4),
        channelwise=channelwise,
        dtype=dtype,
        device=device,
        trainable=True,
    )
    trainable_physics.load_state_dict(legacy_state, strict=True)
    assert trainable_physics._A.requires_grad
    assert torch.equal(trainable_physics._A, matrix)

    y_shape = (3, 2, 4) if channelwise else (3, 4)
    y = torch.randn(y_shape, device=device, dtype=dtype)
    flat_y = y.reshape(-1, 4)

    for scale in (1.0, 2.0):
        if scale != 1.0:
            physics.update(_A=matrix * scale)
        current_matrix = matrix * scale
        expected_adjoint = (flat_y @ current_matrix.conj()).reshape(3, 2, 4, 4)
        expected_inverse = (flat_y @ torch.linalg.pinv(current_matrix).T).reshape(
            3, 2, 4, 4
        )
        assert torch.allclose(physics._A_adjoint, current_matrix.conj().T)
        assert torch.allclose(physics._A_dagger, torch.linalg.pinv(current_matrix))
        with patch("torch.linalg.pinv", side_effect=AssertionError("Cache not reused")):
            physics.state_dict()
            for _ in range(2):
                assert torch.allclose(physics.A_adjoint(y), expected_adjoint)
                assert torch.allclose(physics.A_dagger(y), expected_inverse)


@pytest.mark.parametrize("operation", ["A_adjoint", "A_dagger"])
@pytest.mark.parametrize("channelwise", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_trainable_compressed_sensing_gradients_and_checkpoint(
    device, operation, channelwise, dtype
):
    kwargs = dict(
        m=4, img_size=(2, 4, 4), channelwise=channelwise, dtype=dtype, device=device
    )
    physics = dinv.physics.CompressedSensing(**kwargs, trainable=True)
    physics.update(_A=physics._A * 0.5)
    assert physics._A.is_leaf and physics._A.requires_grad
    optimizer = torch.optim.SGD([physics._A], lr=1e-2)
    y_shape = (3, 2, 4) if channelwise else (3, 4)
    y = torch.randn(y_shape, device=device, dtype=dtype)

    for _ in range(2):
        optimizer.zero_grad()
        reference_matrix = physics._A.detach().clone().requires_grad_(True)
        operator = (
            reference_matrix.conj().T
            if operation == "A_adjoint"
            else torch.linalg.pinv(reference_matrix)
        )
        expected = (y.reshape(-1, 4) @ operator.T).reshape(3, 2, 4, 4)
        expected_gradient = torch.autograd.grad(
            expected.abs().square().mean(), reference_matrix
        )[0]

        output = getattr(physics, operation)(y)
        assert torch.allclose(output, expected)
        output.abs().square().mean().backward()
        assert torch.isfinite(physics._A.grad).all()
        assert torch.allclose(physics._A.grad, expected_gradient)
        optimizer.step()

    # Saving through a parent module must also refresh the nested operator's caches.
    checkpoint = BytesIO()
    torch.save(torch.nn.ModuleList([physics]).state_dict(), checkpoint)
    checkpoint.seek(0)
    state = torch.load(checkpoint, weights_only=True)
    assert torch.allclose(state["0._A_dagger"], torch.linalg.pinv(physics._A))
    assert torch.allclose(state["0._A_adjoint"], physics._A.conj().T)
    assert physics._A_dagger.grad_fn is None
    assert not physics._A_dagger.requires_grad
    assert not physics._A_adjoint.requires_grad

    fixed_physics = dinv.physics.CompressedSensing(**kwargs)
    torch.nn.ModuleList([fixed_physics]).load_state_dict(state, strict=True)
    assert not fixed_physics._A.requires_grad
    assert torch.allclose(fixed_physics.A_adjoint(y), physics.A_adjoint(y))
    assert torch.allclose(fixed_physics.A_dagger(y), physics.A_dagger(y))


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_trainable_compressed_sensing_device_move(device, grad_enabled):
    physics = dinv.physics.CompressedSensing(m=4, img_size=(1, 4, 4), trainable=True)
    with torch.set_grad_enabled(grad_enabled):
        physics.to(device=device, dtype=torch.float64)
    assert physics._A.is_leaf and physics._A.requires_grad
    assert physics.dtype == physics._A.dtype == torch.float64
    assert all(buffer.device == device for buffer in physics.buffers())
    optimizer = torch.optim.SGD([physics._A], lr=1e-2)
    initial = physics._A.detach().clone()
    physics.A_adjoint(torch.ones(1, 4, device=device)).square().mean().backward()
    optimizer.step()
    assert not torch.equal(physics._A, initial)
