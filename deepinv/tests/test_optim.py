import pytest

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.optim import DataFidelity
from deepinv.optim.data_fidelity import L2, IndicatorL2, L1
from deepinv.optim.prior import Prior, PnP, RED
from deepinv.optim.optimizers import optim_builder


def custom_init_CP(y, physics):
    x_init = physics.A_adjoint(y)
    u_init = y
    return {"est": (x_init, x_init, u_init)}


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
    deepinv_prox = data_fidelity.prox(x, y, physics, gamma=gamma)

    assert torch.allclose(deepinv_prox, manual_prox)

    # 4. Testing the gradient of the loss
    grad_deepinv = data_fidelity.grad(x, y, physics)
    grad_manual = B.transpose(0, 1) @ (B @ x - y)

    assert torch.allclose(grad_deepinv, grad_manual)

    # 5. Testing the torch autograd implementation of the gradient
    def dummy_torch_l2(x, y):
        return 0.5 * torch.norm((B @ (x - y)).flatten(), p=2, dim=-1) ** 2

    torch_loss = DataFidelity(d=dummy_torch_l2)
    torch_loss_grad = torch_loss.grad_d(x, y)
    grad_manual = B.transpose(0, 1) @ (B @ (x - y))
    assert torch.allclose(torch_loss_grad, grad_manual)

    # 6. Testing the torch autograd implementation of the prox

    torch_loss = DataFidelity(d=dummy_torch_l2)
    torch_loss_prox = torch_loss.prox_d(
        x, y, gamma=gamma, stepsize_inter=0.1, max_iter_inter=1000, tol_inter=1e-6
    )

    manual_prox = (Id + gamma * B.transpose(0, 1) @ B).inverse() @ (
        x + gamma * B.transpose(0, 1) @ B @ y
    )

    assert torch.allclose(torch_loss_prox, manual_prox)


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
    assert torch.allclose(data_fidelity.prox_d(x, y), x_proj)

    # 3. Testing the proximity operator of the f \circ A
    data_fidelity = IndicatorL2(radius=0.5)

    x = torch.Tensor([[1], [4]]).unsqueeze(0).to(device)
    y = torch.Tensor([[1], [1]]).unsqueeze(0).to(device)

    A = torch.Tensor([[2, 0], [0, 0.5]]).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)

    # Define the physics model associated to this operator
    x_proj = torch.Tensor([[[0.5290], [2.9932]]]).to(device)
    dfb_proj = data_fidelity.prox(x, y, physics, max_iter=1000, crit_conv=1e-12)
    assert torch.allclose(x_proj, dfb_proj, atol=1e-4)
    assert torch.norm(A_forward(dfb_proj) - y) <= radius + 1e-06


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


# we do not test CP (Chambolle-Pock) as we have a dedicated test (due to more specific optimality conditions)
@pytest.mark.parametrize("name_algo", ["PGD", "ADMM", "DRS", "HQS"])
def test_optim_algo(name_algo, imsize, dummy_dataset, device):
    for g_first in [True, False]:
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

        if (
            name_algo == "CP"
        ):  # In the case of primal-dual, stepsizes need to be bounded as reg_param*stepsize < 1/physics.compute_norm(x, tol=1e-4).item()
            stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
            sigma = 1.0
        else:  # Note that not all other algos need such constraints on parameters, but we use these to check that the computations are correct
            stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
            sigma = None

        lamb = 1.1
        max_iter = 1000
        params_algo = {"stepsize": stepsize, "lambda": lamb, "sigma": sigma}

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

        # Run the optimization algorithm
        x = optimalgo(y, physics)

        assert optimalgo.has_converged

        # Compute the subdifferential of the regularisation at the limit point of the algorithm.

        if name_algo == "HQS":
            # In this case, the algorithm does not converge to the minimum of :math:`\lambda f+g` but to that of
            # :math:`\lambda M_{\lambda \tau f}+g` where :math:` M_{\lambda \tau f}` denotes the Moreau envelope of :math:`f` with parameter :math:`\lambda \tau`.
            # Beware, these are not fetch automatically here but handwritten in the test.
            # The optimality condition is then :math:`0 \in \lambda M_{\lambda \tau f}(x)+\partial g(x)`
            if not g_first:
                subdiff = prior.grad(x)
                moreau_grad = (
                    x - data_fidelity.prox(x, y, physics, gamma=lamb * stepsize)
                ) / (
                    lamb * stepsize
                )  # Gradient of the moreau envelope
                assert torch.allclose(
                    lamb * moreau_grad, -subdiff, atol=1e-8
                )  # Optimality condition
            else:
                subdiff = lamb * data_fidelity.grad(x, y, physics)
                moreau_grad = (
                    x - prior.prox(x, gamma=stepsize)
                ) / stepsize  # Gradient of the moreau envelope
                assert torch.allclose(
                    moreau_grad, -subdiff, atol=1e-8
                )  # Optimality condition
        else:
            subdiff = prior.grad(x)
            # In this case, the algorithm converges to the minimum of :math:`\lambda f+g`.
            # The optimality condition is then :math:`0 \in \lambda \nabla f(x)+\partial g(x)`
            grad_deepinv = data_fidelity.grad(x, y, physics)
            assert torch.allclose(
                lamb * grad_deepinv, -subdiff, atol=1e-8
            )  # Optimality condition


def test_denoiser(imsize, dummy_dataset, device):
    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Denoising()  # 2. Set a physical experiment (here, denoising)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    ths = 2.0

    model = dinv.models.TGV(n_it_max=5000, verbose=True, crit=1e-4)

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

    assert model.has_converged


# GD not implemented for this one
@pytest.mark.parametrize("pnp_algo", ["PGD", "HQS", "DRS", "ADMM", "CP"])
def test_pnp_algo(pnp_algo, imsize, dummy_dataset, device):
    pytest.importorskip("pytorch_wavelets")

    # 1. Generate a dummy dataset
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_sample = next(iter(dataloader)).to(device)

    # 2. Set a physical experiment (here, deblurring)
    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
    )
    y = physics(test_sample)
    max_iter = 1000
    # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
    sigma_denoiser = torch.tensor([[0.1]])
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    # here the prior model is common for all iterations
    prior = PnP(denoiser=dinv.models.WaveletPrior(wv="db8", level=3, device=device))

    stepsize_dual = 1.0 if pnp_algo == "CP" else None
    params_algo = {
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
        "stepsize_dual": stepsize_dual,
    }

    custom_init = custom_init_CP if pnp_algo == "CP" else None

    pnp = optim_builder(
        pnp_algo,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        thres_conv=1e-4,
        verbose=True,
        params_algo=params_algo,
        early_stop=True,
        custom_init=custom_init,
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


@pytest.mark.parametrize("pnp_algo", ["PGD", "HQS", "DRS", "ADMM", "CP"])
def test_priors_algo(pnp_algo, imsize, dummy_dataset, device):
    # for prior_name in ['L1Prior', 'Tikhonov']:
    for prior_name in ["L1Prior", "Tikhonov"]:
        # 1. Generate a dummy dataset
        dataloader = DataLoader(
            dummy_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        test_sample = next(iter(dataloader)).to(device)

        # 2. Set a physical experiment (here, deblurring)
        physics = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
        )
        y = physics(test_sample)
        max_iter = 1000
        # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
        # sigma_denoiser = torch.tensor([[0.1]])
        sigma_denoiser = torch.tensor([[1.0]], device=device)
        stepsize = 1.0
        lamb = 1.0

        data_fidelity = L2()

        # here the prior model is common for all iterations
        if prior_name == "L1Prior":
            prior = dinv.optim.prior.L1Prior()
        elif prior_name == "Tikhonov":
            prior = dinv.optim.prior.Tikhonov()

        stepsize_dual = 1.0 if pnp_algo == "CP" else None
        params_algo = {
            "stepsize": stepsize,
            "g_param": sigma_denoiser,
            "lambda": lamb,
            "stepsize_dual": stepsize_dual,
        }

        custom_init = custom_init_CP if pnp_algo == "CP" else None

        opt_algo = optim_builder(
            pnp_algo,
            prior=prior,
            data_fidelity=data_fidelity,
            max_iter=max_iter,
            thres_conv=1e-4,
            verbose=True,
            params_algo=params_algo,
            early_stop=True,
            custom_init=custom_init,
        )

        x = opt_algo(y, physics)

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

        assert opt_algo.has_converged


@pytest.mark.parametrize("red_algo", ["GD", "PGD"])
def test_red_algo(red_algo, imsize, dummy_dataset, device):
    # This test uses WaveletPrior, which requires pytorch_wavelets
    # TODO: we could use a dummy trainable denoiser with a linear layer instead
    pytest.importorskip("pytorch_wavelets")

    # 1. Generate a dummy dataset
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_sample = next(iter(dataloader)).to(device)

    # 2. Set a physical experiment (here, deblurring)
    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
    )
    y = physics(test_sample)
    max_iter = 1000
    sigma_denoiser = 1.0  # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    prior = RED(denoiser=dinv.models.WaveletPrior(wv="db8", level=3, device=device))

    params_algo = {"stepsize": stepsize, "g_param": sigma_denoiser, "lambda": lamb}

    red = optim_builder(
        red_algo,
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        thres_conv=1e-4,
        verbose=True,
        params_algo=params_algo,
        early_stop=True,
        g_first=True,
    )

    red(y, physics)

    assert red.has_converged


def test_CP_K(imsize, dummy_dataset, device):
    r"""
    This test checks that the CP algorithm converges to the solution of the following problem:

    .. math::

        \min_x \lambda a(x) + b(Kx)


    where :math:`a` and :math:`b` are functions and :math:`K` is a linear operator. In this setting, we test both for
    :math:`a(x) = d(Ax-y)` and :math:`b(z) = g(z)`, and for :math:`a(x) = g(x)` and :math:`b(z) = f(z-y)`.
    """

    for g_first in [True, False]:
        # Define two points
        x = torch.tensor([[[10], [10]]], dtype=torch.float64).to(device)

        # Create a measurement operator
        Id_forward = lambda v: v
        Id_adjoint = lambda v: v

        # Define the physics model associated to this operator
        physics = dinv.physics.LinearPhysics(A=Id_forward, A_adjoint=Id_adjoint)
        y = physics(x)

        data_fidelity = L2()  # The data fidelity term

        def prior_g(x, *args):
            ths = 1.0
            return ths * torch.norm(x.view(x.shape[0], -1), p=1, dim=-1)

        prior = Prior(g=prior_g)  # The prior term

        # Define a linear operator
        K = torch.tensor([[2, 1], [-1, 0.5]], dtype=torch.float64).to(device)
        K_forward = lambda v: K @ v
        K_adjoint = lambda v: K.transpose(0, 1) @ v

        # stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
        stepsize = 0.9 / torch.linalg.norm(K, ord=2).item() ** 2
        reg_param = 1.0
        stepsize_dual = 1.0

        lamb = 1.5
        max_iter = 1000

        params_algo = {
            "stepsize": stepsize,
            "g_param": reg_param,
            "lambda": lamb,
            "stepsize_dual": stepsize_dual,
            "K": K_forward,
            "K_adjoint": K_adjoint,
        }

        optimalgo = optim_builder(
            "CP",
            prior=prior,
            data_fidelity=data_fidelity,
            max_iter=max_iter,
            crit_conv="residual",
            thres_conv=1e-11,
            verbose=True,
            params_algo=params_algo,
            early_stop=True,
            g_first=g_first,
            custom_init=custom_init_CP,
        )

        # Run the optimization algorithm
        x = optimalgo(y, physics)

        print("g_first: ", g_first)
        assert optimalgo.has_converged

        # Compute the subdifferential of the regularisation at the limit point of the algorithm.
        if not g_first:
            subdiff = prior.grad(x, 0)

            grad_deepinv = K_adjoint(
                data_fidelity.grad(K_forward(x), y, physics)
            )  # This test is only valid for differentiable data fidelity terms.
            assert torch.allclose(
                lamb * grad_deepinv, -subdiff, atol=1e-12
            )  # Optimality condition

        else:
            subdiff = K_adjoint(prior.grad(K_forward(x), 0))

            grad_deepinv = data_fidelity.grad(x, y, physics)
            assert torch.allclose(
                lamb * grad_deepinv, -subdiff, atol=1e-12
            )  # Optimality condition


def test_CP_datafidsplit(imsize, dummy_dataset, device):
    r"""
    This test checks that the CP algorithm converges to the solution of the following problem:

    .. math::

        \min_x \lambda d(Ax,y) + g(x)


    where :math:`d` is a distance function and :math:`g` is a prior term.
    """

    g_first = False
    # Define two points
    x = torch.tensor([[[10], [10]]], dtype=torch.float64).to(device)

    # Create a measurement operator
    A = torch.tensor([[2, 1], [-1, 0.5]], dtype=torch.float64).to(device)
    A_forward = lambda v: A @ v
    A_adjoint = lambda v: A.transpose(0, 1) @ v

    # Define the physics model associated to this operator
    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    y = physics(x)

    data_fidelity = L2()  # The data fidelity term

    def prior_g(x, *args):
        ths = 1.0
        return ths * torch.norm(x.view(x.shape[0], -1), p=1, dim=-1)

    prior = Prior(g=prior_g)  # The prior term

    # stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
    stepsize = 0.9 / torch.linalg.norm(A, ord=2).item() ** 2
    reg_param = 1.0
    stepsize_dual = 1.0

    lamb = 1.5
    max_iter = 1000

    params_algo = {
        "stepsize": stepsize,
        "g_param": reg_param,
        "lambda": lamb,
        "stepsize_dual": stepsize_dual,
        "K": A_forward,
        "K_adjoint": A_adjoint,
    }

    optimalgo = optim_builder(
        "CP",
        prior=prior,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        crit_conv="residual",
        thres_conv=1e-11,
        verbose=True,
        params_algo=params_algo,
        early_stop=True,
        g_first=g_first,
        custom_init=custom_init_CP,
    )

    # Run the optimization algorithm
    x = optimalgo(y, physics)

    assert optimalgo.has_converged

    # Compute the subdifferential of the regularisation at the limit point of the algorithm.
    subdiff = prior.grad(x, 0)

    grad_deepinv = A_adjoint(
        data_fidelity.grad_d(A_forward(x), y)
    )  # This test is only valid for differentiable data fidelity terms.
    assert torch.allclose(
        lamb * grad_deepinv, -subdiff, atol=1e-12
    )  # Optimality condition
