import math
import pytest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import deepinv as dinv
from deepinv.optim import DataFidelity
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import Prior, PnP
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.plotting import plot, torch2cpu
from deepinv.unfolded import unfolded_builder, DEQ_builder
from deepinv.utils import investigate_model


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


optim_algos = ["PGD", "HQS", "DRS", "ADMM", "CP"]


@pytest.mark.parametrize("unfolded_algo", optim_algos)
def test_unfolded(unfolded_algo, imsize, dummy_dataset, device):
    # pths
    BASE_DIR = Path(".")
    ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
    # DATA_DIR = BASE_DIR / "measurements"
    # RESULTS_DIR = BASE_DIR / "results"
    # DEG_DIR = BASE_DIR / "degradations"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
    level = 3
    prior = [
        PnP(denoiser=dinv.models.WaveletPrior(wv="db8", level=level, device=device))
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
    sigma_denoiser = [sigma_denoiser_init * torch.ones(level, 3)] * max_iter
    # sigma_denoiser = [torch.Tensor([sigma_denoiser_init])]*max_iter
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
    }

    trainable_params = [
        "g_param",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.

    # Because the CP algorithm uses more than 2 variables, we need to define a custom initialization.
    if unfolded_algo == "CP":

        def custom_init(y, physics):
            x_init = physics.A_adjoint(y)
            u_init = y
            return {"est": (x_init, x_init, u_init)}

        params_algo["sigma"] = 1.0
    else:
        custom_init = None

    model = unfolded_builder(
        unfolded_algo,
        params_algo=params_algo,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        custom_init=custom_init,
    )

    for idx, (name, param) in enumerate(model.named_parameters()):
        assert param.requires_grad
        assert (trainable_params[0] in name) or (trainable_params[1] in name)


@pytest.mark.parametrize("unfolded_algo", optim_algos)
def test_DEQ(unfolded_algo, imsize, dummy_dataset, device):
    # pths
    BASE_DIR = Path(".")
    ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
    # DATA_DIR = BASE_DIR / "measurements"
    # RESULTS_DIR = BASE_DIR / "results"
    # DEG_DIR = BASE_DIR / "degradations"
    CKPT_DIR = BASE_DIR / "ckpts"

    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
    level = 3
    prior = [
        PnP(denoiser=dinv.models.WaveletPrior(wv="db8", level=level, device=device))
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
    sigma_denoiser = [sigma_denoiser_init * torch.ones(level, 3)] * max_iter
    # sigma_denoiser = [torch.Tensor([sigma_denoiser_init])]*max_iter
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
    }

    trainable_params = [
        "g_param",
        "stepsize",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.

    # Because the CP algorithm uses more than 2 variables, we need to define a custom initialization.
    if unfolded_algo == "CP":

        def custom_init(y, physics):
            x_init = physics.A_adjoint(y)
            u_init = y
            return {"est": (x_init, x_init, u_init)}

        params_algo["sigma"] = 1.0
    else:
        custom_init = None

    model = DEQ_builder(
        unfolded_algo,
        params_algo=params_algo,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
        custom_init=custom_init,
        anderson_acceleration=True,
        anderson_acceleration_backward=True,
    )

    for idx, (name, param) in enumerate(model.named_parameters()):
        assert param.requires_grad
        assert (trainable_params[0] in name) or (trainable_params[1] in name)
