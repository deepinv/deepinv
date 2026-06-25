import pytest
import doctest

import torch

import deepinv as dinv
from deepinv.utils import DownloadError
from dummy import DummyCircles

import importlib

# Tag stored on a TestReport's ``user_properties`` when we reclassify a
# download failure as a skip. We attach it to the report (rather than to
# ``config.stash``) so it survives the worker → controller serialization
# performed by ``pytest-xdist``: ``pytest_runtest_makereport`` runs on the
# worker, but ``pytest_terminal_summary`` runs on the controller, and only
# data living on the report itself crosses that boundary.
_DEEPINV_DOWNLOAD_ERROR_PROP = "deepinv_download_error"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Convert failures caused by transient network errors into skips.

    Detects any test (or fixture) failure whose exception is a
    :class:`deepinv.utils.DownloadError`. Those are raised explicitly by the
    deepinv download helpers (:func:`deepinv.utils.load_url`,
    :func:`deepinv.datasets.utils.download_archive`, …) when a remote server
    returns a network-level error (e.g. a HuggingFace 429 rate-limit). The
    test is reported as skipped instead of failed, and tagged on the report
    object so the terminal-summary hook can print it as its own section —
    even under pytest-xdist where the hook runs on the controller and the
    classification runs on the worker.
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

    if not call.excinfo.errisinstance(DownloadError):
        return

    typename = call.excinfo.typename
    msg = str(call.excinfo.value)

    # user_properties is part of TestReport and is preserved by
    # pytest-xdist's report (de)serialization, so the controller sees it.
    report.user_properties.append((_DEEPINV_DOWNLOAD_ERROR_PROP, (typename, msg)))

    report.outcome = "skipped"
    report.longrepr = f"Skipped due to network error ({typename}): {call.excinfo.value}"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print tests skipped due to network errors as their own summary section.

    Reads tags off TestReport.user_properties so the section works both
    with and without pytest-xdist: under xdist, the controller's
    ``terminalreporter.stats`` is populated from worker reports.
    """
    records = []
    for report in terminalreporter.stats.get("skipped", []):
        # Reports from pytest-xdist or pytest internals may not always carry
        # user_properties (e.g. synthetic reports for collection-time skips).
        for name, value in getattr(report, "user_properties", None) or ():
            if name == _DEEPINV_DOWNLOAD_ERROR_PROP:
                # JSON roundtrip via xdist turns tuples into lists; unpack
                # either form.
                typename, msg = value
                records.append((report.nodeid, typename, msg))
                break

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


# A pytest hook to ignore certain outputs in doctests
# NOTE: The expected output is ignored when IGNORE_OUTPUT is set and it should
# ideally be left empty to avoid confusion.
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    excinfo = call.excinfo
    # A DocTestFailure is raised if and only if the doctest failed due to a
    # mismatch between the actual and expected outputs.
    if (
        report.when == "call"
        and report.failed
        and excinfo is not None
        and issubclass(excinfo.type, doctest.DocTestFailure)
    ):
        err = excinfo.value
        example = err.example
        source = example.source  # The failing doctest source line
        comment = (
            "".join(source.split("#")[1:]) if "#" in source else ""
        )  # Comment part
        # NOTE: The syntax is quite strict for the sake of simplicity but
        # it can be made more flexible if needed.
        if comment.strip() == "deepinv: +IGNORE_OUTPUT":
            report.outcome = "passed"
            report.longrepr = None
