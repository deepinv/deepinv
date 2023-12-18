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



#Test jules

from deepinv.utils.optimization import NeuralIteration, GradientDescent, ProximalGradientDescent
import deepinv as dinv  # You may need to adjust this import based on your actual module structure

@pytest.fixture
def setup_data():
    net = dinv.models.unet().to(dinv.device)
    physics = dinv.physics.Inpainting([32, 32], device=dinv.device)
    x = torch.randn(10, 1, 32, 32).to(dinv.device)
    y = physics.A(x)
    fbp = physics.A_dagger(y)
    return net, physics, x, y, fbp

def test_neural_iteration_forward(setup_data):
    net, physics, _, y, _ = setup_data
    neural_iteration = NeuralIteration()
    result = neural_iteration.forward(y, physics)
    assert torch.equal(result, physics.A_adjoint(y))


# Add more tests as needed




