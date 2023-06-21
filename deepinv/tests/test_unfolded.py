import math
import pytest

import torch

import deepinv as dinv
from deepinv.optim import DataFidelity
from deepinv.models.denoiser import Denoiser
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import Prior, PnP
from deepinv.tests.dummy_datasets.datasets import DummyCircles
from deepinv.utils.plotting import plot, torch2cpu
from deepinv.unfolded import Unfolded
from deepinv.utils import investigate_model
from deepinv.training_utils import train, test

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


optim_algos = [
    "PGD",
    "HQS",
    "DRS",
    "ADMM",
    "CP",
]


# we do not test CP (Chambolle-Pock) as we have a dedicated test (due to more specific optimality conditions)
@pytest.mark.parametrize("name_algo", optim_algos)
def test_optim_algo(name_algo, imsize, dummy_dataset, device):
    # Select the data fidelity term
    data_fidelity = L2()

    # Set up the trainable denoising prior; here, the soft-threshold in a wavelet basis.
    level = 3
    model_spec = {
        "name": "waveletprior",
        "args": {"wv": "db8", "level": level, "device": device},
    }
    # If the prior is initialized with a list of length max_iter,
    # then a distinct weight is trained for each PGD iteration.
    # For fixed trained model prior across iterations, initialize with a single model.
    max_iter = 30 if torch.cuda.is_available() else 20  # Number of unrolled iterations
    prior = [PnP(denoiser=Denoiser(model_spec)) for i in range(max_iter)]

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
    model = Unfolded(
        name_algo,
        params_algo=params_algo,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
    )

    for idx, (name, param) in enumerate(model.named_parameters()):
        assert param.requires_grad
        assert (trainable_params[0] in name) or (trainable_params[1] in name)

    N = 10
    max_N = 10
    train_dataset = DummyCircles(samples=N, imsize=imsize)
    test_dataset = DummyCircles(samples=N, imsize=imsize)

    physics = dinv.physics.Inpainting(mask=0.5, tensor_size=imsize, device=device)

    tmp_pth = '.'
    dinv.datasets.generate_dataset(
        train_dataset,
        physics,
        tmp_pth,
        test_dataset=test_dataset,
        device=device,
        dataset_filename="dinv_dataset",
        train_datapoints=max_N,
    )

    dataset = dinv.datasets.HDF5Dataset(path=f"{tmp_pth}/dinv_dataset0.h5", train=True)

    train_dataloader = DataLoader(
        train_dataset, batch_size=2, num_workers=0, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=2, num_workers=0, shuffle=False
    )

