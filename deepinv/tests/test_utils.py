import deepinv
import torch
import pytest
from deepinv.utils.decorators import _deprecated_alias
import warnings
import numpy as np
from contextlib import nullcontext
import matplotlib
import random


@pytest.fixture
def non_interactive_matplotlib():
    # Use a non-interactive backend to avoid blocking tests
    current_backend = matplotlib.get_backend()
    matplotlib.use("agg")
    yield
    # Restore the original backend
    matplotlib.use(current_backend)


@pytest.fixture
def tensorlist():
    x = torch.ones((1, 1, 2, 2))
    y = torch.ones((1, 1, 2, 2))
    x = deepinv.utils.TensorList([x, x])
    y = deepinv.utils.TensorList([y, y])
    return x, y


def test_tensordict_sum(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2)) * 2
    z1 = deepinv.utils.TensorList([z, z])
    z = x + y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = y + x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_mul(tensorlist):
    x, y = tensorlist
    alpha = 1.0
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * alpha
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()
    z = alpha * x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_scalar_mul(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x * y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_div(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x / y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_sub(tensorlist):
    x, y = tensorlist
    z = torch.zeros((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = x - y
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_neg(tensorlist):
    x, y = tensorlist
    z = -torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z])
    z = -x
    assert (z1[0] == z[0]).all() and (z1[1] == z[1]).all()


def test_tensordict_append(tensorlist):
    x, y = tensorlist
    z = torch.ones((1, 1, 2, 2))
    z1 = deepinv.utils.TensorList([z, z, z, z])
    z = x.append(y)
    assert (z1[0] == z[0]).all() and (z1[-1] == z[-1]).all()


@pytest.mark.parametrize("shape", [(1, 1, 3, 3), (1, 1, 5, 5)])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_dirac_like(shape, length):
    rng = torch.Generator().manual_seed(0)
    x = [torch.randn(shape, generator=rng) for _ in range(length)]
    h = deepinv.utils.dirac_like(x)
    y = deepinv.utils.TensorList(
        [
            deepinv.physics.functional.conv2d(xi, hi, padding="circular")
            for hi, xi in zip(h, x)
        ]
    )

    for xi, hi, yi in zip(x, h, y):
        assert (
            hi.shape == xi.shape
        ), "Dirac delta should have the same shape as the input tensor."

        if hi.shape[-2] % 2 == 1 and hi.shape[-1] % 2 == 1:
            assert torch.allclose(
                xi, yi
            ), "Convolution with Dirac delta should return the original tensor."


@pytest.mark.parametrize("C", list(range(1, 5)))
@pytest.mark.parametrize("save_plot", [False, True])
def test_plot(tmpdir, C, save_plot):
    x = torch.ones((1, C, 2, 2))
    titles, imgs = ["a", "b"], [x, x]
    save_dir = tmpdir if save_plot else None
    deepinv.utils.plot(imgs, titles=titles, show=False, save_dir=save_dir)
    deepinv.utils.plot(x, titles="a", show=False, save_dir=save_dir)
    deepinv.utils.plot(imgs, show=False, save_dir=save_dir)
    deepinv.utils.plot(
        {k: v for k, v in zip(titles, imgs)}, show=False, save_dir=save_dir
    )


@pytest.mark.parametrize("n_plots", [1, 2, 3])
@pytest.mark.parametrize("titles", [None, "Dummy plot"])
@pytest.mark.parametrize("save_plot", [False, True])
def test_scatter_plot(tmpdir, n_plots, titles, save_plot):
    xy_list = torch.randn(100, 2, generator=torch.Generator().manual_seed(0))
    xy_list = [xy_list] * n_plots if n_plots > 1 else xy_list
    titles = [titles] * n_plots if n_plots > 1 else titles
    save_dir = tmpdir if save_plot else None
    deepinv.utils.scatter_plot(xy_list, titles=titles)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("n_metrics", [1, 2])
@pytest.mark.parametrize("n_batches", [1, 2])
@pytest.mark.parametrize("n_iterations", [1, 2])
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("show", [False, True])
def test_plot_curves(tmpdir, seed, n_metrics, n_batches, n_iterations, save_plot, show):
    rng = random.Random(seed)
    metrics = {
        str(k): [[rng.random() for _ in range(n_iterations)] for _ in range(n_batches)]
        for k in range(n_metrics)
    }
    save_dir = tmpdir if save_plot else None
    deepinv.utils.plot_curves(metrics, save_dir=save_dir, show=show)


def test_plot_inset():
    # Plots a batch of images with a checkboard pattern, with different inset locations
    x = torch.ones(2, 1, 100, 100)

    for i in range(0, 100, 10):
        x[:, :, :, i : i + 5] = 0
        x[:, :, i : i + 5, :] = 0

    deepinv.utils.plot_inset(
        [x],
        titles=["a"],
        labels=["a"],
        inset_loc=((0, 0.5), (0.5, 0.5)),
        show=False,
        save_fn="temp.png",
    )


def test_plot_videos():
    x = torch.rand((1, 3, 5, 8, 8))  # B,C,T,H,W image sequence
    y = torch.rand((1, 3, 5, 16, 16))
    deepinv.utils.plot_videos(
        [x, y], display=True
    )  # this should generate warning without IPython installed
    deepinv.utils.plot_videos([x, y], save_fn="vid.gif")


def test_save_videos():
    x = torch.rand((1, 3, 5, 8, 8))  # B,C,T,H,W image sequence
    y = torch.rand((1, 3, 5, 16, 16))
    deepinv.utils.save_videos([x, y], time_dim=2, save_fn="vid.gif")


def test_plot_ortho3D():
    for c in range(1, 5):
        x = torch.ones((1, c, 2, 2, 2))
        imgs = [x, x]
        deepinv.utils.plot_ortho3D(imgs, titles=["a", "b"], show=False)
        deepinv.utils.plot_ortho3D(x, titles="a", show=False)
        deepinv.utils.plot_ortho3D(imgs, show=False)


# -------------- Test deprecated_alias --------------
class DummyModule(torch.nn.Module):
    @_deprecated_alias(old_lr="lr")
    def __init__(self, lr=0.1):
        super().__init__()
        self.lr = lr


@_deprecated_alias(old_arg="new_arg")
def dummy_function(new_arg=0.1):
    return new_arg**2


def test_deprecated_alias():
    # --- Class (torch.nn.Module) tests ---
    with pytest.warns(DeprecationWarning, match="old_lr.*deprecated"):
        m1 = DummyModule(old_lr=0.01)
        assert m1.lr == 0.01

    m2 = DummyModule(lr=0.02)
    assert m2.lr == 0.02

    with pytest.raises(TypeError, match="Cannot specify both 'old_lr' and 'lr'"):
        DummyModule(old_lr=0.01, lr=0.02)

    # Test no warning with correct parameter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        DummyModule(lr=0.3)
        assert len(record) == 0

    # --- Function tests ---
    with pytest.warns(DeprecationWarning, match="old_arg.*deprecated"):
        result1 = dummy_function(old_arg=0.1)
        assert result1 == 0.1**2

    result2 = dummy_function(new_arg=0.2)
    assert result2 == 0.2**2
    with pytest.raises(TypeError, match="Cannot specify both 'old_arg' and 'new_arg'"):
        dummy_function(old_arg=0.1, new_arg=0.2)
    # Test no warning with correct parameter
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        dummy_function(new_arg=0.3)
        assert len(record) == 0


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("n_data", [1, 2, 3])
@pytest.mark.parametrize("transform", [None, lambda x: x])
@pytest.mark.parametrize("length", [1, 2, 10, np.inf])
def test_random_phantom_dataset(size, n_data, transform, length):
    # Although it is the default value for the parameter length, the current
    # implementation fails when it is used. We simply verify this behavior
    # but it will probably need to be changed in the future.
    dataset = None
    with pytest.raises(ValueError) if length == np.inf else nullcontext():
        dataset = deepinv.utils.RandomPhantomDataset(
            size=size, n_data=n_data, transform=transform, length=length
        )
        assert dataset is not None, "Dataset should not be None when length is finite."

    if dataset is not None:
        x, y = dataset[0]

        assert (
            len(dataset) == length
        ), "Length of dataset should match the specified length."

        assert x.shape == (
            n_data,
            size,
            size,
        ), "Shape of phantom should match (n_data, size, size)."


@pytest.mark.parametrize("size", [64, 128])
@pytest.mark.parametrize("n_data", [1, 2, 3])
@pytest.mark.parametrize("transform", [None, lambda x: x])
def test_shepp_logan_dataset(size, n_data, transform):
    dataset = deepinv.utils.SheppLoganDataset(
        size=size, n_data=n_data, transform=transform
    )
    x, y = dataset[0]

    assert len(dataset) == 1, "Length of dataset should be 1 for Shepp-Logan phantom."

    assert x.shape == (
        n_data,
        size,
        size,
    ), "Shape of phantom should match (n_data, size, size)."


# Module-level fixtures
pytestmark = [pytest.mark.usefixtures("non_interactive_matplotlib")]
