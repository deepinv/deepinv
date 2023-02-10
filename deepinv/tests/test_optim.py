import pytest
import deepinv as dinv
from deepinv.tests.dummy_datasets.datasets import DummyCircles


@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def imsize():
    h = 28
    w = 32
    c = 3
    return c, h, w

# TODO: use a DummyCircle as dataset and check convergence of optim algorithms (maybe with TV denoiser)
