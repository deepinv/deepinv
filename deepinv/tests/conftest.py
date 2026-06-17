import pytest

import torch

import deepinv as dinv
from dummy import DummyCircles

import importlib

# Keywords (case-insensitive) that indicate a test failure was caused by a
# transient network problem rather than a real bug. When any of these appear
# in the exception text or traceback, the test is converted to a skip so CI
# is not failed by flaky HTTPS connections.
_NETWORK_ERROR_KEYWORDS = (
    "httperror",
    "httpserror",
    "http.client",
    "httpsconnectionpool",
    "httpconnectionpool",
    "connectionerror",
    "connectionreseterror",
    "connectionaborted",
    "connectionrefused",
    "connectiontimeout",
    "sslerror",
    "PytorchStreamReader",
    "urlerror",
    "urllib",
)

_NETWORK_SKIPS_KEY = pytest.StashKey[list]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert failures caused by transient network errors into skips.

    The CI of deepinv occasionally fails because of HTTPS connection issues
    when downloading datasets, models or example images. Any test that raises
    an exception whose type or message mentions a network-related keyword is
    reported as skipped and recorded in a session-scoped list that is printed
    as its own section in the terminal summary at the end of the run.
    """
    outcome = yield
    report = outcome.get_result()

    # `call` covers test-body failures; `setup` covers fixture failures (which
    # pytest reports as ERROR, not FAILED). We need to intercept both because
    # downloads typically happen inside session-scoped fixtures.
    if (
        report.when not in ("call", "setup")
        or not report.failed
        or call.excinfo is None
    ):
        return

    try:
        excinfo_text = (
            f"{call.excinfo.typename}\n"
            f"{call.excinfo.value!r}\n"
            f"{call.excinfo.getrepr(style='short')!s}"
        ).lower()
    except Exception:
        excinfo_text = f"{call.excinfo.typename}\n{call.excinfo.value!r}".lower()

    if not any(kw in excinfo_text for kw in _NETWORK_ERROR_KEYWORDS):
        return

    record = (item.nodeid, call.excinfo.typename, str(call.excinfo.value))
    item.config.stash.setdefault(_NETWORK_SKIPS_KEY, []).append(record)

    report.outcome = "skipped"
    report.longrepr = (
        f"Skipped due to network error ({call.excinfo.typename}): "
        f"{call.excinfo.value}"
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print tests skipped due to network errors as their own summary section."""
    records = config.stash.get(_NETWORK_SKIPS_KEY, None)
    if not records:
        return

    terminalreporter.write_sep("=", "Failed examples due to download errors")
    for nodeid, exc_type, exc_msg in records:
        terminalreporter.write_line(f"{nodeid}")
        terminalreporter.write_line(f"    {exc_type}: {exc_msg}")
    terminalreporter.write_line(
        f"({len(records)} test(s) skipped because of transient network errors)"
    )


@pytest.fixture(
    params=list(
        dict.fromkeys([torch.device("cpu"), dinv.utils.get_device(verbose=False)])
    )
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


@pytest.fixture(scope="session")
def _example_image_cache():
    """Session-scoped cache so each (name, img_size) pair is downloaded at most once."""
    return {}


@pytest.fixture
def load_example_image(_example_image_cache):
    """Return a loader that caches images by (name, img_size, extra kwargs) for the full test session.

    Usage::

        img = load_example_image("butterfly.png", img_size=64, resize_mode="resize")
        img = load_example_image("celeba_example.jpg")   # default img_size

    The returned tensor is on CPU; call ``.to(device)`` yourself when needed.
    ``device`` is intentionally excluded from the cache key so one cached copy
    serves all devices.
    """

    def _load(name, img_size=None, **kwargs):
        key = (name, img_size, tuple(sorted(kwargs.items())))
        if key not in _example_image_cache:
            _example_image_cache[key] = dinv.utils.load_example(
                name, img_size=img_size, **kwargs
            )
        return _example_image_cache[key]

    return _load


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
