import math
import pytest

import deepinv as dinv
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2, IndicatorL2, L1
from deepinv.optim.optimizers import *
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.plotting import plot_debug, torch2cpu

from torch.utils.data import DataLoader


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w


@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=1, imsize=imsize)


def test_data_fidelity_l2():
    data_fidelity = L2()

    # 1. Testing value of the loss for a simple case
    # Define two points
    x = torch.Tensor([1, 4])
    y = torch.Tensor([1, 1])

    # Create a measurement operator
    A = torch.Tensor([[2, 0], [0, 0.5]])
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    assert data_fidelity(x, y, physics) == 1.0

    # Compute the gradient of f
    grad_fA = data_fidelity.grad(x, y, physics)  # print grad_f gives [2.0000, 0.5000]

    # Compute the proximity operator of f
    prox_fA = data_fidelity.prox(
        x, y, physics, gamma=1.0
    )  # print prox_fA gives [0.6000, 3.6000]

    # 2. Testing trivial operations on f and not f\circ A
    gamma = 1.0
    assert torch.allclose(
        data_fidelity.prox_f(x, y, gamma), (x + gamma * y) / (1 + gamma)
    )
    assert torch.allclose(data_fidelity.grad_f(x, y), x - y)

    # 3. Testing the value of the proximity operator for a nonsymmetric linear operator
    # Create a measurement operator
    B = torch.Tensor([[2, 1], [-1, 0.5]])
    B_forward = lambda v: B @ v
    B_adjoint = lambda v: B.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=B_forward, A_adjoint=B_adjoint)

    # Compute the proximity operator manually (closed form formula)
    Id = torch.eye(2)
    manual_prox = (Id + gamma * B.transpose(0, 1) @ B).inverse() @ (
        x + gamma * B.transpose(0, 1) @ y
    )

    # Compute the deepinv proximity operator
    deepinv_prox = data_fidelity.prox(x, y, physics, gamma)

    assert torch.allclose(deepinv_prox, manual_prox)

    # 4. Testing the gradient of the loss
    grad_deepinv = data_fidelity.grad(x, y, physics)
    grad_manual = B.transpose(0, 1) @ (B @ x - y)

    assert torch.allclose(grad_deepinv, grad_manual)


def test_data_fidelity_indicator():
    # Define two points
    x = torch.Tensor([1, 4])
    y = torch.Tensor([1, 1])

    # Redefine the data fidelity with a different radius
    radius = 0.5
    data_fidelity = IndicatorL2(radius=radius)

    # Create a measurement operator
    A = torch.Tensor([[2, 0], [0, 0.5]])
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Test values of the loss for points inside and outside the l2 ball
    assert data_fidelity(x, y, physics) == 1e16
    assert data_fidelity(x / 2, y, physics) == 0
    assert data_fidelity.f(x, y, radius=1) == 1e16
    assert data_fidelity.f(x, y, radius=3.1) == 0

    # 2. Testing trivial operations on f (and not f \circ A)
    x_proj = torch.Tensor([1.0, 1 + radius])
    assert torch.allclose(data_fidelity.prox_f(x, y, gamma=None), x_proj)

    # 3. Testing the proximity operator of the f \circ A
    data_fidelity = IndicatorL2(radius=0.5)

    x = torch.Tensor([1, 4])
    y = torch.Tensor([1, 1])

    A = torch.Tensor([[2, 0], [0, 0.5]])
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Define the physics model associated to this operator
    x_proj = torch.Tensor([0.5290, 2.9917])
    dfb_proj = data_fidelity.prox(x, y, physics)
    assert torch.allclose(x_proj, dfb_proj)
    assert torch.norm(A_forward(dfb_proj) - y) <= radius


def test_data_fidelity_l1():
    # Define two points
    x = torch.Tensor([1, 4, -0.5])
    y = torch.Tensor([1, 1, 1])

    data_fidelity = L1()
    assert torch.allclose(data_fidelity.f(x, y), (x - y).abs().sum())

    A = torch.Tensor([[2, 0, 0], [0, -0.5, 0], [0, 0, 1]])
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    Ax = A_forward(x)
    assert data_fidelity(x, y, physics) == (Ax - y).abs().sum()

    # Check subdifferential
    grad_manual = torch.sign(x - y)
    assert torch.allclose(data_fidelity.grad_f(x, y), grad_manual)

    # Check prox
    threshold = 0.5
    prox_manual = torch.Tensor([1.0, 3.5, 0.0])
    assert torch.allclose(data_fidelity.prox_f(x, y, threshold), prox_manual)


def test_denoiser(imsize, dummy_dataset, device):
    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Denoising()  # 2. Set a physical experiment (here, denoising)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    ths = 2.0

    model_spec = {
        "name": "tgv",
        "args": {"n_it_max": 5000, "verbose": True, "crit": 1e-4},
    }
    model = Denoiser(model_spec)

    x = model(y, ths)  # 3. Apply the model we want to test

    plot = False

    if plot:
        imgs = []
        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
        imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

        titles = ["Input", "Output"]
        num_im = 2
        plot_debug(
            imgs, shape=(1, num_im), titles=titles, row_order=True, save_dir=None
        )

    assert model.denoiser.has_converged


optim_algos = ["PGD", "HQS", "DRS", "ADMM", "PD"]


# optim_algos = ['PGD']
# optim_algos = ['GD']  # To implement
@pytest.mark.parametrize("pnp_algo", optim_algos)
def test_optim_algo(pnp_algo, imsize, dummy_dataset, device):
    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader)).to(device)

    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=dinv.device
    )  # 2. Set a physical experiment (here, deblurring)
    y = physics(test_sample)
    max_iter = 1000
    sigma_denoiser = 0.1
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    model_spec = {
        "name": "waveletprior",
        "args": {"wv": "db8", "level": 3, "device": device},
    }
    prior = {"prox_g": Denoiser(model_spec)}
    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
    pnp = optimbuilder(
        pnp_algo,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        thres_conv=1e-4,
        verbose=True,
        params_algo=params_algo,
    )

    x = pnp(y, physics)

    plot = False
    if plot:
        imgs = []
        imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
        imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))

        titles = ["Input", "Output"]
        num_im = 2
        plot_debug(
            imgs, shape=(1, num_im), titles=titles, row_order=True, save_dir=None
        )

    assert pnp.has_converged()
