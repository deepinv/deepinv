import math
import pytest

import deepinv as dinv
from deepinv.models.denoiser import Denoiser
from deepinv.models.basic_prox_models import ProxL1Prior
from deepinv.optim.data_fidelity import L2, IndicatorL2, L1
from deepinv.optim.prior import Prior, PnP
from deepinv.optim.optimizers import *
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.plotting import plot, torch2cpu

from torch.utils.data import DataLoader


@pytest.fixture
def device():
    return dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w


@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=1, imsize=imsize)


def test_data_fidelity_l2(device):
    data_fidelity = L2()

    # 1. Testing value of the loss for a simple case
    # Define two points
    x = torch.Tensor([[1], [4]]).unsqueeze(0).to(device)
    y = torch.Tensor([[1], [1]]).unsqueeze(0).to(device)

    # Create a measurement operator
    A = torch.Tensor([[2, 0], [0, 0.5]]).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    assert torch.allclose(data_fidelity(x, y, physics), torch.Tensor([1.0]).to(device))

    # Compute the gradient of f
    grad_dA = data_fidelity.grad(
        x, y, physics
    )  # print(grad_dA) gives [[[2.0000], [0.5000]]]

    # Compute the proximity operator of f
    prox_dA = data_fidelity.prox(
        x, y, physics, gamma=1.0
    )  # print(prox_dA) gives [[[0.6000], [3.6000]]]

    # 2. Testing trivial operations on f and not f\circ A
    gamma = 1.0
    assert torch.allclose(
        data_fidelity.prox_d(x, y, gamma), (x + gamma * y) / (1 + gamma)
    )
    assert torch.allclose(data_fidelity.grad_d(x, y), x - y)

    # 3. Testing the value of the proximity operator for a nonsymmetric linear operator
    # Create a measurement operator
    B = torch.Tensor([[2, 1], [-1, 0.5]]).to(device)
    B_forward = lambda v: B @ v
    B_adjoint = lambda v: B.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=B_forward, A_adjoint=B_adjoint)

    # Compute the proximity operator manually (closed form formula)
    Id = torch.eye(2).to(device)
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


def test_data_fidelity_indicator(device):
    # Define two points
    x = torch.Tensor([[1], [4]]).unsqueeze(0).to(device)
    y = torch.Tensor([[1], [1]]).unsqueeze(0).to(device)

    # Redefine the data fidelity with a different radius
    radius = 0.5
    data_fidelity = IndicatorL2(radius=radius)

    # Create a measurement operator
    A = torch.Tensor([[2, 0], [0, 0.5]]).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Test values of the loss for points inside and outside the l2 ball
    assert data_fidelity(x, y, physics) == 1e16
    assert data_fidelity(x / 2, y, physics) == 0
    assert data_fidelity.d(x, y, radius=1) == 1e16
    assert data_fidelity.d(x, y, radius=3.1) == 0

    # 2. Testing trivial operations on f (and not f \circ A)
    x_proj = torch.Tensor([[[1.0], [1 + radius]]]).to(device)
    assert torch.allclose(data_fidelity.prox_d(x, y, gamma=None), x_proj)

    # 3. Testing the proximity operator of the f \circ A
    data_fidelity = IndicatorL2(radius=0.5)

    x = torch.Tensor([[1], [4]]).unsqueeze(0).to(device)
    y = torch.Tensor([[1], [1]]).unsqueeze(0).to(device)

    A = torch.Tensor([[2, 0], [0, 0.5]]).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Define the physics model associated to this operator
    x_proj = torch.Tensor([[[0.5290], [2.9917]]]).to(device)
    dfb_proj = data_fidelity.prox(x, y, physics)
    assert torch.allclose(x_proj, dfb_proj)
    assert torch.norm(A_forward(dfb_proj) - y) <= radius


def test_data_fidelity_l1(device):
    # Define two points
    x = torch.Tensor([[[1], [4], [-0.5]]]).to(device)
    y = torch.Tensor([[[1], [1], [1]]]).to(device)

    data_fidelity = L1()
    assert torch.allclose(data_fidelity.d(x, y), (x - y).abs().sum())

    A = torch.Tensor([[2, 0, 0], [0, -0.5, 0], [0, 0, 1]]).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    Ax = A_forward(x)
    assert data_fidelity(x, y, physics) == (Ax - y).abs().sum()

    # Check subdifferential
    grad_manual = torch.sign(x - y)
    assert torch.allclose(data_fidelity.grad_d(x, y), grad_manual)

    # Check prox
    threshold = 0.5
    prox_manual = torch.Tensor([[[1.0], [3.5], [0.0]]]).to(device)
    assert torch.allclose(data_fidelity.prox_d(x, y, threshold), prox_manual)


optim_algos = ["PGD", "ADMM", "DRS", "CP", "HQS"]


# other algos: check constraints on the stepsize
@pytest.mark.parametrize("name_algo", optim_algos)
def test_optim_algo(name_algo, imsize, dummy_dataset, device):
    for g_first in [True, False]:  # Test both g first and f first
        if not g_first or (g_first and not ("HQS" in name_algo or "PGD" in name_algo)):
            # Define two points
            x = torch.tensor([[[10], [10]]], dtype=torch.float64)

            # Create a measurement operator
            B = torch.tensor([[2, 1], [-1, 0.5]], dtype=torch.float64)
            B_forward = lambda v: B @ v
            B_adjoint = lambda v: B.transpose(0, 1) @ v

            # Define the physics model associated to this operator
            physics = dinv.physics.LinearPhysics(A=B_forward, A_adjoint=B_adjoint)
            y = physics(x)

            data_fidelity = L2()  # The data fidelity term

            def prior_g(x, *args):
                ths = 0.1
                return ths * torch.norm(x.view(x.shape[0], -1), p=1, dim=-1)

            prior = Prior(g=prior_g)  # The prior term

            # reg = L1()  # The regularization term
            #
            # def prox_g(x, ths=0.1):
            #     return reg.prox_d(x, 0, ths)

            # old
            # prior = {"prox_g": prox_g}

            # dirty hack, temporary
            # # TODO: clarify
            # prior = Prior(g=prox_g)  # here the prior model is common for all iterations

            if (
                name_algo == "CP"
            ):  # In the case of primal-dual, stepsizes need to be bounded as reg_param*stepsize < 1/physics.compute_norm(x, tol=1e-4).item()
                stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
                reg_param = 1.0
            else:  # Note that not all other algos need such constraints on parameters, but we use these to check that the computations are correct
                stepsize = 1.0 / physics.compute_norm(x, tol=1e-4).item()
                reg_param = 1.0 * stepsize

            lamb = 1.5
            max_iter = 1000
            params_algo = {"stepsize": stepsize, "g_param": reg_param, "lambda": lamb}

            optimalgo = optim_builder(
                name_algo,
                prior=prior,
                data_fidelity=data_fidelity,
                max_iter=max_iter,
                crit_conv="residual",
                thres_conv=1e-11,
                verbose=True,
                params_algo=params_algo,
                early_stop=True,
                g_first=g_first,
            )

            # Run the optimisation algorithm
            x = optimalgo(y, physics)

            assert optimalgo.has_converged

            # Compute the subdifferential of the regularisation at the limit point of the algorithm.
            subdiff = prior.grad(x, 0)

            if name_algo == "HQS":
                # In this case, the algorithm does not converge to the minimum of :math:`\lambda f+g` but to that of
                # :math:`\lambda \gamma_1 ^1(f)+\gamma_2 g` where :math:`^1(f)` denotes the Moreau envelope of :math:`f`,
                # and :math:`\gamma_1` and :math:`\gamma_2` are the stepsizes in the proximity operators. Beware, these are
                # not fetch automatically here but handwritten in the test.
                # The optimality condition is then :math:`0 \in \gamma_1 \nabla ^1(f)(x)+\gamma_2 \partial g(x)`
                stepsize_f = lamb * stepsize
                stepsize_g = reg_param

                moreau_grad = (
                    x - data_fidelity.prox(x, y, physics, stepsize_f)
                ) / stepsize_f  # Gradient of the moreau envelope
                assert torch.allclose(
                    moreau_grad * stepsize_f, -subdiff * stepsize_g, atol=1e-12
                )  # Optimality condition
            else:
                # In this case, the algorithm converges to the minimum of :math:`\lambda f+g`.
                # The optimality condition is then :math:`0 \in \lambda \nabla f(x)+\partial g(x)`
                grad_deepinv = data_fidelity.grad(x, y, physics)
                assert torch.allclose(
                    lamb * grad_deepinv, -subdiff, atol=1e-12
                )  # Optimality condition


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

    # For debugging
    # plot = False
    # if plot:
    #     imgs = []
    #     imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
    #     imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))
    #
    #     titles = ["Input", "Output"]
    #     num_im = 2
    #     plot_debug(
    #         imgs, shape=(1, num_im), titles=titles, row_order=True, save_dir=None
    #     )

    assert model.denoiser.has_converged


optim_algos = ["PGD", "HQS", "DRS", "ADMM", "CP"]  # GD not implemented for this one


@pytest.mark.parametrize("pnp_algo", optim_algos)
def test_pnp_algo(pnp_algo, imsize, dummy_dataset, device):
    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader)).to(device)

    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
    )  # 2. Set a physical experiment (here, deblurring)
    y = physics(test_sample)
    max_iter = 1000
    sigma_denoiser = 1.0  # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    model_spec = {
        "name": "waveletprior",
        "args": {"wv": "db8", "level": 3, "device": device},
    }

    prior = PnP(
        denoiser=Denoiser(model_spec)
    )  # here the prior model is common for all iterations

    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}
    pnp = optim_builder(
        pnp_algo,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        thres_conv=1e-4,
        verbose=True,
        params_algo=params_algo,
        early_stop=True,
    )

    x = pnp(y, physics)

    # # For debugging  # Remark: to get nice results, lower sigma_denoiser to 0.001
    # plot = True
    # if plot:
    #     imgs = []
    #     imgs.append(torch2cpu(y[0, :, :, :].unsqueeze(0)))
    #     imgs.append(torch2cpu(x[0, :, :, :].unsqueeze(0)))
    #     imgs.append(torch2cpu(test_sample[0, :, :, :].unsqueeze(0)))
    #
    #     titles = ["Input", "Output", "Groundtruth"]
    #     num_im = 3
    #     plot_debug(
    #         imgs, shape=(1, num_im), titles=titles, row_order=True, save_dir=None
    #     )

    assert pnp.has_converged
