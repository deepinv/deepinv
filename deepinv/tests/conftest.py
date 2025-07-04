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


# Certain tests are particularly slow and make for a large part of
# the time it takes for the entire test suite to run. For this reason, we make
# them run in parallel of the rest of the tests thereby reducing the overall
# test time drastically.
# NOTE: The decorator `pytest.hookimpl̀` is needed to make sure that the group
# marks are set before xdist reads them.
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Set the xdist group of the test items based on their markers."""
    next_slow_idx = 1
    for item in items:
        slow_marker = item.get_closest_marker("slow")
        if slow_marker is not None:
            # Slow tests can't share the same group to make sure they run in
            # parallel. This is why we use a counter to create unique group
            # names.
            group_name = f"slow_{next_slow_idx}"
            item.add_marker(pytest.mark.xdist_group(group_name))
            next_slow_idx += 1
        else:
            # All other tests are grouped under "main" and run one at a time
            # but in parallel of the slow tests.
            item.add_marker(pytest.mark.xdist_group("main"))
