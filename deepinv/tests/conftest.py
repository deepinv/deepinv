import pytest

import torch

import deepinv as dinv
from dummy import DummyCircles

import importlib

import os, socket, subprocess, time, sys


@pytest.fixture(
    params=[torch.device("cpu")]
    + ([dinv.utils.get_freer_gpu()] if torch.cuda.is_available() else [])
)
def device(request):
    return request.param


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


@pytest.fixture
def non_blocking_plots():
    """Make plots in a test non-blocking"""
    import matplotlib
    import matplotlib.pyplot as plt

    original_backend = matplotlib.get_backend()
    try:
        # Use a non-interactive backend to avoid blocking the tests
        matplotlib.use("Agg", force=True)
        plt.close("all")
        # Reload matplotlib.pyplot to force usage
        importlib.reload(plt)
        yield
    finally:
        plt.close("all")
        # Restore the original backend
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


@pytest.fixture(params=[1])
def world_size(request):
    """Parametrize tests with different world sizes (number of processes)."""
    return request.param


@pytest.fixture
def dist_config(world_size):
    """
    Configuration for distributed testing with torch.distributed.
    
    This fixture provides:
    - world_size: number of processes to spawn
    - backend: 'gloo' (CPU-compatible) or 'nccl' (GPU-only)
    - master_addr/port: for process group initialization
    
    The fixture is parametrized to run tests with 1, 2, and 3 processes.
    For multi-process tests, the actual distributed initialization happens
    within the test via DistributedContext when RANK/WORLD_SIZE env vars are set.
    """
    # Use unique port based on world_size to avoid conflicts
    base_port = 29500
    port = base_port + world_size
    
    config = {
        "world_size": world_size,
        "backend": "gloo",  # CPU-compatible backend
        "master_addr": "127.0.0.1",
        "master_port": str(port),
        "device_mode": "cpu",  # Force CPU for CI compatibility
    }
    return config
