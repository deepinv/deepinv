import pytest

import torch

import deepinv as dinv
from dummy import DummyCircles

import matplotlib
import importlib
from contextlib import contextmanager


@pytest.fixture
def device():
    return (
        dinv.utils.get_freer_gpu() if torch.cuda.is_available() else torch.device("cpu")
    )


@pytest.fixture
def toymatrix():
    w = 50
    A = torch.diag(torch.Tensor(range(1, w + 1)))
    return A


@pytest.fixture
def dummy_dataset(imsize, device):
    return DummyCircles(samples=2, imsize=imsize)


@pytest.fixture
def imsize():
    h = 37
    w = 31
    c = 3
    return c, h, w


@pytest.fixture
def imsize_1_channel():
    h = 37
    w = 31
    c = 1
    return c, h, w


@pytest.fixture
def imsize_2_channel():
    h = 37
    w = 31
    c = 2
    return c, h, w


@pytest.fixture
def rng(device):
    return torch.Generator(device).manual_seed(0)


@contextmanager
def no_plot():
    """Wrap any statement to send matplotlib calls to not display plots."""
    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.close("all")
        importlib.reload(plt)
        yield
    finally:
        plt.close("all")
        matplotlib.use(original_backend, force=True)
        importlib.reload(plt)
