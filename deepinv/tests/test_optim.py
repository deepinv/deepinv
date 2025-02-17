import pytest

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.optim import DataFidelity
from deepinv.optim.data_fidelity import L2, IndicatorL2, L1, AmplitudeLoss
from deepinv.optim.prior import Prior, PnP, RED
from deepinv.optim.optimizers import optim_builder
from deepinv.optim.optim_iterators import GDIteration
from deepinv.tests.test_physics import find_operator


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
        data_fidelity.d.prox(x, y, gamma), (x + gamma * y) / (1 + gamma)
    )
    assert torch.allclose(data_fidelity.d.grad(x, y), x - y)

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
    torch_loss_grad = torch_loss.d.grad(x, y)
    grad_manual = B.transpose(0, 1) @ (B @ (x - y))
    assert torch.allclose(torch_loss_grad, grad_manual)

    # 6. Testing the torch autograd implementation of the prox

    torch_loss = DataFidelity(d=dummy_torch_l2)
    torch_loss_prox = torch_loss.d.prox(
        x, y, gamma=gamma, stepsize_inter=0.1, max_iter_inter=1000, tol_inter=1e-6
    )

    manual_prox = (Id + gamma * B.transpose(0, 1) @ B).inverse() @ (
        x + gamma * B.transpose(0, 1) @ B @ y
    )

    assert torch.allclose(torch_loss_prox, manual_prox)

    # 7. Testing that d.prox / d.grad and prox_d / grad_d are consistent
    assert torch.allclose(
        data_fidelity.d.prox(x, y, gamma=1.0),
        data_fidelity.prox_d(x, y, physics, gamma=1.0),
    )
    assert torch.allclose(
        data_fidelity.d.grad(x, y),
        data_fidelity.grad_d(
            x,
            y,
        ),
    )


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
    assert torch.allclose(data_fidelity.d.prox(x, y), x_proj)

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

    # 4. Testing that d.prox / d.grad and prox_d / grad_d are consistent
    assert torch.allclose(
        data_fidelity.d.prox(x, y, gamma=1.0),
        data_fidelity.prox_d(x, y, physics, gamma=1.0),
    )


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

    # Check sub-differential
    grad_manual = torch.sign(x - y)
    assert torch.allclose(data_fidelity.d.grad(x, y), grad_manual)

    # Check prox
    threshold = 0.5
    prox_manual = torch.Tensor([[[1.0], [3.5], [0.0]]]).to(device)
    assert torch.allclose(data_fidelity.d.prox(x, y, gamma=threshold), prox_manual)

    # Testing that d.prox / d.grad and prox_d / grad_d are consistent
    assert torch.allclose(
        data_fidelity.d.prox(x, y, gamma=1.0),
        data_fidelity.prox_d(x, y, physics, gamma=1.0),
    )
    assert torch.allclose(
        data_fidelity.d.grad(x, y),
        data_fidelity.grad_d(
            x,
            y,
        ),
    )


def test_zero_prior():
    A = torch.eye(3, dtype=torch.float64)

    def A_forward(v):
        return A @ v

    def A_adjoint(v):
        return A.T @ v

    physics = dinv.physics.LinearPhysics(A=A_forward, A_adjoint=A_adjoint)
    data_fidelity = dinv.optim.data_fidelity.L2()
    params_algo = {"stepsize": 0.5, "lambda": 1.0}
    iterator = GDIteration()
    optimalgo = dinv.optim.BaseOptim(
        iterator,
        data_fidelity=data_fidelity,
        params_algo=params_algo,
    )
    for _ in range(10):
        x = torch.randn((3,), dtype=torch.float64)
        xhat = optimalgo(x, physics)
        assert torch.allclose(xhat, x)


def test_data_fidelity_amplitude_loss(device):
    r"""
    Tests if the gradient computed with grad_d method of amplitude loss is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    with torch.enable_grad():
        x = torch.randn(
            (1, 1, 3, 3), dtype=torch.cfloat, device=device, requires_grad=True
        )
        physics = dinv.physics.RandomPhaseRetrieval(
            m=10, img_shape=(1, 3, 3), device=device
        )
        loss = AmplitudeLoss()
        func = lambda x: loss(x, torch.ones_like(physics(x)), physics)[0]
        grad_value = torch.func.grad(func)(x)
        jvp_value = loss.grad(x, torch.ones_like(physics(x)), physics)
    assert torch.isclose(grad_value[0], jvp_value, rtol=1e-5).all()


# we do not test CP (Chambolle-Pock) as we have a dedicated test (due to more specific optimality conditions)
@pytest.mark.parametrize("name_algo", ["GD", "PGD", "ADMM", "DRS", "HQS", "FISTA"])
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

        def prior_g(x, *args, **kwargs):
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

        lamb = 0.9
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
            # In this case, the algorithm does not converge to the minimum of :math:`f+\lambda g` but to that of
            # :math:` M_{\tau f}+ \lambda g` where :math:` M_{\tau f}` denotes the Moreau envelope of :math:`f` with parameter :math:`\tau`.
            # Beware, these are not fetch automatically here but handwritten in the test.
            # The optimality condition is then :math:`0 \in M_{\tau f}(x)+ \lambda \partial g(x)`
            if not g_first:
                subdiff = prior.grad(x)
                moreau_grad = (
                    x - data_fidelity.prox(x, y, physics, gamma=stepsize)
                ) / (
                    stepsize
                )  # Gradient of the moreau envelope
                assert torch.allclose(
                    moreau_grad, -lamb * subdiff, atol=1e-8
                )  # Optimality condition
            else:
                subdiff = data_fidelity.grad(x, y, physics)
                moreau_grad = (x - prior.prox(x, gamma=lamb * stepsize)) / (
                    lamb * stepsize
                )  # Gradient of the moreau envelope
                assert torch.allclose(
                    lamb * moreau_grad, -subdiff, atol=1e-8
                )  # Optimality condition
        else:
            subdiff = prior.grad(x)
            # In this case, the algorithm converges to the minimum of :math:`f+\lambda g`.
            # The optimality condition is then :math:`0 \in  \nabla f(x)+ \lambda \partial g(x)`
            grad_deepinv = data_fidelity.grad(x, y, physics)
            assert torch.allclose(
                grad_deepinv, -lamb * subdiff, atol=1e-8
            )  # Optimality condition


def test_denoiser(imsize, dummy_dataset, device):
    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    test_sample = next(iter(dataloader))

    physics = dinv.physics.Denoising()  # 2. Set a physical experiment (here, denoising)
    y = physics(test_sample).type(test_sample.dtype).to(device)

    ths = 2.0

    model = dinv.models.TGVDenoiser(n_it_max=5000, verbose=True, crit=1e-4)

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
@pytest.mark.parametrize("pnp_algo", ["PGD", "HQS", "DRS", "ADMM", "CP", "FISTA"])
def test_pnp_algo(pnp_algo, imsize, dummy_dataset, device):
    pytest.importorskip("ptwt")

    # 1. Generate a dummy dataset
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_sample = next(iter(dataloader)).to(device)

    # 2. Set a physical experiment (here, deblurring)
    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0),
        device=device,
        padding="circular",
    )
    y = physics(test_sample)
    max_iter = 1000
    # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
    sigma_denoiser = torch.tensor([[0.1]])
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    # here the prior model is common for all iterations
    prior = PnP(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=3, device=device))

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


def get_prior(prior_name, device="cpu"):
    if prior_name == "L1Prior":
        prior = dinv.optim.prior.L1Prior()
    elif prior_name == "L12Prior":
        prior = dinv.optim.prior.L12Prior(l2_axis=1)  # l2 on channels
    elif prior_name == "Tikhonov":
        prior = dinv.optim.prior.Tikhonov()
    elif prior_name == "TVPrior":
        prior = dinv.optim.prior.TVPrior()
    elif "wavelet" in prior_name.lower():
        pytest.importorskip(
            "ptwt",
            reason="This test requires pytorch_wavelets. It should be "
            "installed with `pip install "
            "git+https://github.com/fbcotter/pytorch_wavelets.git`",
        )
        if prior_name == "WaveletPrior":
            prior = dinv.optim.prior.WaveletPrior(wv="db8", level=3, device=device)
        elif prior_name == "WaveletDictPrior":
            prior = dinv.optim.prior.WaveletPrior(
                wv=["db1", "db4", "db8"], level=3, device=device
            )
    return prior


@pytest.mark.parametrize("pnp_algo", ["PGD", "HQS", "DRS", "ADMM", "CP", "FISTA"])
def test_priors_algo(pnp_algo, imsize, dummy_dataset, device):
    for prior_name in [
        "L1Prior",
        "L12Prior",
        "Tikhonov",
        "TVPrior",
        "WaveletPrior",
        "WaveletDictPrior",
    ]:
        # 1. Generate a dummy dataset
        dataloader = DataLoader(
            dummy_dataset, batch_size=1, shuffle=False, num_workers=0
        )
        test_sample = next(iter(dataloader)).to(device)

        # 2. Set a physical experiment (here, deblurring)
        physics = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0),
            padding="circular",
            device=device,
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
        prior = get_prior(prior_name, device=device)

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


@pytest.mark.parametrize("red_algo", ["GD", "PGD", "FISTA"])
def test_red_algo(red_algo, imsize, dummy_dataset, device):
    # This test uses WaveletDenoiser, which requires pytorch_wavelets
    # TODO: we could use a dummy trainable denoiser with a linear layer instead
    pytest.importorskip("ptwt")

    # 1. Generate a dummy dataset
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_sample = next(iter(dataloader)).to(device)

    # 2. Set a physical experiment (here, deblurring)
    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0),
        device=device,
    )
    y = physics(test_sample)
    max_iter = 1000
    sigma_denoiser = 1.0  # Note: results are better for sigma_denoiser=0.001, but it takes longer to run.
    stepsize = 1.0
    lamb = 1.0

    data_fidelity = L2()

    prior = RED(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=3, device=device))

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


def test_dpir(imsize, dummy_dataset, device):
    # 1. Generate a dummy dataset
    dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_sample = next(iter(dataloader)).to(device)

    # 2. Set a physical experiment (here, deblurring)
    physics = dinv.physics.Blur(
        dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0),
        device=device,
        noise_model=dinv.physics.GaussianNoise(0.1),
        padding="circular",
    )
    y = physics(test_sample)
    model = dinv.optim.DPIR(0.1, device=device)
    out = model(y, physics)

    in_psnr = dinv.metric.PSNR()(test_sample, y)
    out_psnr = dinv.metric.PSNR()(out, test_sample)

    assert out_psnr > in_psnr


def test_CP_K(imsize, dummy_dataset, device):
    r"""
    This test checks that the CP algorithm converges to the solution of the following problem:

    .. math::

        \min_x  a(x) + \lambda b(Kx)


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

        def prior_g(x, *args, **kwargs):
            ths = 1.0
            return ths * torch.norm(x.view(x.shape[0], -1), p=1, dim=-1)

        prior = Prior(g=prior_g)  # The prior term

        # Define a linear operator
        K = torch.tensor([[2, 1], [-1, 0.5]], dtype=torch.float64).to(device)
        K_forward = lambda v: K @ v
        K_adjoint = lambda v: K.transpose(0, 1) @ v

        stepsize = 0.9 / torch.linalg.norm(K, ord=2).item() ** 2
        reg_param = 1.0
        stepsize_dual = 1.0

        lamb = 0.6
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
            subdiff = prior.grad(x)
            grad_deepinv = K_adjoint(
                data_fidelity.grad(K_forward(x), y, physics)
            )  # This test is only valid for differentiable data fidelity terms.
            assert torch.allclose(
                grad_deepinv, -lamb * subdiff, atol=1e-12
            )  # Optimality condition

        else:
            subdiff = K_adjoint(prior.grad(K_forward(x)))
            grad_deepinv = data_fidelity.grad(x, y, physics)
            assert torch.allclose(
                grad_deepinv, -lamb * subdiff, atol=1e-12
            )  # Optimality condition


def test_CP_datafidsplit(imsize, dummy_dataset, device):
    r"""
    This test checks that the CP algorithm converges to the solution of the following problem:

    .. math::

        \min_x d(Ax,y) + \lambda g(x)


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
    physics = dinv.physics.LinearPhysics()
    y = physics(x)

    data_fidelity = L2()  # The data fidelity term

    def prior_g(x, *args, **kwargs):
        ths = 1.0
        return ths * torch.norm(x.view(x.shape[0], -1), p=1, dim=-1)

    prior = Prior(g=prior_g)  # The prior term

    # stepsize = 0.9 / physics.compute_norm(x, tol=1e-4).item()
    stepsize = 0.9 / torch.linalg.norm(A, ord=2).item() ** 2
    reg_param = 1.0
    stepsize_dual = 1.0

    lamb = 0.6
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
    subdiff = prior.grad(x)

    grad_deepinv = A_adjoint(
        data_fidelity.d.grad(A_forward(x), y)
    )  # This test is only valid for differentiable data fidelity terms.
    assert torch.allclose(
        grad_deepinv, -lamb * subdiff, atol=1e-12
    )  # Optimality condition


def test_patch_prior(imsize, dummy_dataset, device):
    pytest.importorskip(
        "FrEIA",
        reason="This test requires FrEIA. It should be "
        "installed with `pip install FrEIA",
    )
    torch.manual_seed(0)

    dataloader = DataLoader(
        dummy_dataset, batch_size=1, shuffle=False, num_workers=0
    )  # 1. Generate a dummy dataset
    # gray-valued
    test_sample = next(iter(dataloader)).mean(1, keepdim=True).to(device)

    with torch.enable_grad():
        physics = dinv.physics.Denoising(
            noise_model=dinv.physics.GaussianNoise(0.1)
        )  # 2. Set a physical experiment (here, denoising)
        y = physics(test_sample).type(test_sample.dtype).to(device)

        epll = dinv.optim.EPLL(channels=test_sample.shape[1], device=device)
        patchnr = dinv.optim.PatchNR(channels=test_sample.shape[1], device=device)
        prior1 = dinv.optim.PatchPrior(epll.negative_log_likelihood)
        prior2 = dinv.optim.PatchPrior(patchnr)
        data_fidelity = L2()

        lam = 1.0
        x_out = []
        for prior in [prior1, prior2]:
            x = y.detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=0.01)
            for i in range(10):
                optimizer.zero_grad()
                loss = data_fidelity(x, y, physics) + prior(x, lam)
                loss.backward()
                optimizer.step()
            x_out.append(x.detach())

    assert torch.sum((x_out[0] - test_sample) ** 2) < torch.sum((y - test_sample) ** 2)


def test_datafid_stacking(imsize, device):
    physics = dinv.physics.StackedLinearPhysics(
        [dinv.physics.Denoising(), dinv.physics.Denoising()]
    )
    data_fid = dinv.optim.StackedPhysicsDataFidelity(
        [dinv.optim.L2(2.0), dinv.optim.L2(1.0)]
    )

    x = torch.ones((1, 1, 1, 1), device=device)
    y = physics.A(x)
    y2 = dinv.utils.TensorList([3 * y[0], 2 * y[1]])

    assert (
        data_fid(x, y2, physics)
        == (y2[0] - y[0]) ** 2 / (4 * 2) + (y2[1] - y[1]) ** 2 / 2
    )

    assert data_fid.grad(x, y2, physics) == -(y2[0] - y[0]) / 4 - (y2[1] - y[1])


solvers = ["CG", "BiCGStab", "lsqr"]
least_squares_physics = ["fftdeblur", "inpainting", "MRI", "super_resolution_circular"]


@pytest.mark.parametrize("physics_name", least_squares_physics)
@pytest.mark.parametrize("solver", solvers)
def test_least_square_solvers(device, solver, physics_name):
    batch_size = 4

    physics, img_size, _, _ = find_operator(physics_name, device=device)

    x = torch.randn((batch_size, *img_size), device=device)

    tol = 0.01
    y = physics(x)
    x_hat = physics.A_dagger(y, solver=solver, tol=tol)
    assert (
        (physics.A(x_hat) - y).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        / y.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        < tol
    ).all()

    z = x.clone()
    gamma = 1.0

    x_hat = physics.prox_l2(z, y, gamma=gamma, solver=solver, tol=tol)

    assert (
        (x_hat - x).abs().pow(2).mean(dim=(1, 2, 3), keepdim=True)
        / x.pow(2).mean(dim=(1, 2, 3), keepdim=True)
        < 3 * tol
    ).all()

    # test backprop
    y.requires_grad = True
    x_hat = physics.A_dagger(y, solver=solver, tol=tol)
    loss = (x_hat - x).pow(2).mean()
    loss.backward()
    if not "inpainting" in physics_name:
        assert y.grad.norm() > 0


def test_condition_number(device):
    imsize = (2, 1, 32, 32)

    c = torch.rand(imsize, device=device) * 0.95 + 0.05

    class DummyPhysics(dinv.physics.LinearPhysics):
        def A(self, x, **kwargs):
            return x * c

        def A_adjoint(self, y, **kwargs):
            return y * c

    physics = DummyPhysics()
    x = torch.randn(imsize, device=device)
    cond = physics.condition_number(x)
    gt_cond = c.max() / c.min()
    rel_error = (cond - gt_cond).abs() / gt_cond
    assert rel_error < 0.1
