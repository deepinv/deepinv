import pytest
import torch

import deepinv as dinv
from deepinv.optim.prior import PnP, RED
from deepinv.optim.data_fidelity import L2
from deepinv.optim import DEQ_builder
from deepinv import optim

UNFOLDED_ALGO = [
    "ProximalGradientDescent",
    "HQS",
    "GradientDescent",
    "ADMM",
    "FISTA",
]


@pytest.mark.parametrize("unfolded_algo", UNFOLDED_ALGO)
def test_unfolded(unfolded_algo, imsize, dummy_dataset, device):
    pytest.importorskip("ptwt")

    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
    level = 3
    if not unfolded_algo == "GradientDescent":
        prior = [
            PnP(
                denoiser=dinv.models.WaveletDenoiser(
                    wv="db8", level=level, device=device
                )
            )
            for i in range(max_iter)
        ]
    else:
        prior = [
            RED(
                denoiser=dinv.models.WaveletDenoiser(
                    wv="db8", level=level, device=device
                )
            )
            for i in range(max_iter)
        ]

    # Unrolled optimization algorithm parameters
    lamb = [
        1.0
    ] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
    stepsize = [
        1.0
    ] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.

    sigma_denoiser_init = 0.01
    sigma_denoiser = [sigma_denoiser_init * torch.ones(1, level, 3)] * max_iter
    # sigma_denoiser = [torch.Tensor([sigma_denoiser_init])]*max_iter
    trainable_params = [
        "g_param",
        "stepsize",
    ]  # define which parameters  are trainable

    # Define the unfolded trainable model.
    model = getattr(optim, unfolded_algo)(
        unfold=True,
        stepsize=stepsize,
        g_param=sigma_denoiser,
        lambda_reg=lamb,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
    )

    for idx, (name, param) in enumerate(model.named_parameters()):
        assert param.requires_grad
        assert (trainable_params[0] in name) or (trainable_params[1] in name)

    # batch_size, n_channels, img_size_w, img_size_h = 5, imsize
    batch_size = 5
    n_channels, img_size_w, img_size_h = imsize
    noise_level = 0.01

    torch.manual_seed(0)
    test_sample = torch.randn(batch_size, n_channels, img_size_w, img_size_h).to(device)
    groundtruth_sample = torch.randn(batch_size, n_channels, img_size_w, img_size_h).to(
        device
    )

    physics = dinv.physics.BlurFFT(
        img_size=(n_channels, img_size_w, img_size_h),
        filter=dinv.physics.blur.gaussian_blur(),
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
    )

    y = physics(test_sample).type(test_sample.dtype).to(device)

    out = model(y, physics=physics)

    assert out.shape == test_sample.shape

    loss_fn = dinv.loss.SupLoss(metric=torch.nn.MSELoss())
    loss = loss_fn(groundtruth_sample, out)
    loss.backward()


DEQ_ALGO = ["ProximalGradientDescent", "HQS"]


@pytest.mark.parametrize("unfolded_algo", DEQ_ALGO)
def test_DEQ(unfolded_algo, imsize, dummy_dataset, device):
    pytest.importorskip("ptwt")
    torch.set_grad_enabled(
        True
    )  # Disabled somewhere in previous test files, necessary for this test to pass

    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
    level = 3
    prior = [
        PnP(denoiser=dinv.models.WaveletDenoiser(wv="db8", level=level, device=device))
        for i in range(max_iter)
    ]

    # Unrolled optimization algorithm parameters
    lamb = [
        1.0
    ] * max_iter  # initialization of the regularization parameter. A distinct lamb is trained for each iteration.
    stepsize = [
        1.0
    ] * max_iter  # initialization of the stepsizes. A distinct stepsize is trained for each iteration.

    sigma_denoiser_init = 0.01
    sigma_denoiser = [sigma_denoiser_init * torch.ones(1, level, 3)] * max_iter

    trainable_params = [
        "g_param",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.
    for and_acc in [False, True]:
        for jac_free in [False, True]:
            # DRS, ADMM and CP algorithms are not real fixed-point algorithms on the primal variable

            model = getattr(optim, unfolded_algo)(
                stepsize=stepsize,
                g_param=sigma_denoiser,
                lambda_reg=lamb,
                DEQ=True,
                trainable_params=trainable_params,
                data_fidelity=data_fidelity,
                max_iter=max_iter,
                prior=prior,
                anderson_acceleration=and_acc,
                DEQ_anderson_acceleration_backward=and_acc,
                DEQ_jacobian_free=jac_free,
            )

            for idx, (name, param) in enumerate(model.named_parameters()):
                assert param.requires_grad
                assert (trainable_params[0] in name) or (trainable_params[1] in name)

            # batch_size, n_channels, img_size_w, img_size_h = 5, imsize
            batch_size = 5
            n_channels, img_size_w, img_size_h = imsize
            noise_level = 0.01

            torch.manual_seed(0)
            test_sample = torch.randn(
                batch_size, n_channels, img_size_w, img_size_h
            ).to(device)
            groundtruth_sample = torch.randn(
                batch_size, n_channels, img_size_w, img_size_h
            ).to(device)

            physics = dinv.physics.BlurFFT(
                img_size=(n_channels, img_size_w, img_size_h),
                filter=dinv.physics.blur.gaussian_blur(),
                device=device,
                noise_model=dinv.physics.GaussianNoise(sigma=noise_level),
            )

            y = physics(test_sample).type(test_sample.dtype).to(device)

            out = model(y, physics=physics)

            assert out.shape == test_sample.shape

            loss_fn = dinv.loss.SupLoss(metric=torch.nn.MSELoss())
            loss = loss_fn(groundtruth_sample, out)
            loss.backward()
