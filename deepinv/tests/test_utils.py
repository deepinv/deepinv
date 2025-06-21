import deepinv
import torch
import pytest
from deepinv.utils.decorators import _deprecated_alias
import warnings
import numpy as np
from contextlib import nullcontext
import matplotlib
import random
import unittest.mock as mock
from unittest.mock import patch
import subprocess
import os
import inspect
import itertools
import pathlib
import torchvision
import torchvision.transforms as transforms
import PIL


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


@pytest.fixture
def model():
    physics = deepinv.physics.Denoising()
    model = deepinv.optim.optimizers.optim_builder(
        iteration="PGD",
        prior=deepinv.optim.prior.TVPrior(n_it_max=20),
        data_fidelity=deepinv.optim.data_fidelity.L2(),
        early_stop=True,
        max_iter=10,
        verbose=False,
        params_algo={"stepsize": 1.0, "lambda": 1e-2},
    )
    x = torch.randn(1, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    # NOTE: It is needed for attribute params_algo to be initialized.
    _ = model(x, physics=physics)
    yield model


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


@pytest.mark.parametrize("C", [1, 3])
@pytest.mark.parametrize("n_images", range(1, 3))
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("cbar", [False, True])
@pytest.mark.parametrize("with_titles", [False, True])
@pytest.mark.parametrize("dict_img_list", [False, True])
@pytest.mark.parametrize("suptitle", [None, "dummy_title"])
def test_plot(
    tmpdir,
    C,
    n_images,
    save_plot,
    cbar,
    with_titles,
    dict_img_list,
    suptitle,
):
    shape = (1, C, 2, 2)
    img_list = torch.ones(shape)
    img_list = [img_list] * n_images if isinstance(img_list, torch.Tensor) else img_list
    titles = "0" if n_images == 1 else [str(i) for i in range(n_images)]
    img_list = {k: v for k, v in zip(titles, img_list)}
    if not with_titles:
        titles = None
    if not dict_img_list:
        img_list = list(img_list.values())
    save_dir = tmpdir if save_plot else None
    with (
        pytest.raises(AssertionError)
        if titles is not None and isinstance(img_list, dict)
        else nullcontext()
    ):
        deepinv.utils.plot(
            img_list,
            titles=titles,
            save_dir=save_dir,
            cbar=cbar,
            suptitle=suptitle,
        )


@pytest.mark.parametrize("n_plots", [1, 2, 3])
@pytest.mark.parametrize("titles", [None, "Dummy plot"])
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("show", [False, True])
@pytest.mark.parametrize("suptitle", [None, "dummy_title"])
def test_scatter_plot(tmpdir, n_plots, titles, save_plot, show, suptitle):
    xy_list = torch.randn(100, 2, generator=torch.Generator().manual_seed(0))
    xy_list = [xy_list] * n_plots if n_plots > 1 else xy_list
    if titles is not None:
        titles = [titles] * n_plots if n_plots > 1 else titles
    save_dir = tmpdir if save_plot else None
    deepinv.utils.scatter_plot(
        xy_list, titles=titles, suptitle=suptitle, save_dir=save_dir, show=show
    )


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


@pytest.mark.parametrize("init_params", [None])
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("show", [False, True])
def test_plot_parameters(tmpdir, model, init_params, save_plot, show):
    save_dir = tmpdir if save_plot else None
    deepinv.utils.plot_parameters(model, init_params, save_dir=save_dir, show=show)


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


@pytest.mark.parametrize("input_shape", [(1, 3, 32, 64)])
@pytest.mark.parametrize("size", [32, 128, 129])
def test_resize_pad_square_tensor(input_shape, size):
    tensor = torch.rand(input_shape, generator=torch.Generator().manual_seed(0))
    output = deepinv.utils.resize_pad_square_tensor(tensor, size)

    assert isinstance(output, torch.Tensor), "Output should be a tensor."
    assert output.dim() == 4, "Output tensor should be 4D."
    assert output.shape[-2] == output.shape[-1], "Output tensor should be square."
    assert output.shape[-2] == size, "Output tensor should have the specified size."

    # The frequency of black pixels should increase as long as the input tensor is not square
    def black_pixels_frequency(im, bin_size=10 / 255):
        return torch.sum(2 * im.abs() < bin_size) / im.numel()

    if input_shape[-2] != input_shape[-1]:
        assert black_pixels_frequency(output) > black_pixels_frequency(
            tensor
        ), "Black pixels frequency should increase after resizing and padding."


@pytest.mark.parametrize("input_shape", [(4, 3, 32, 32), (4, 2, 32, 32)])
def test_torch2cpu(input_shape):
    tensor = torch.randn(input_shape, generator=torch.Generator().manual_seed(0))
    output = deepinv.utils.torch2cpu(tensor)

    assert isinstance(output, np.ndarray), "Output should be a numpy array."

    # Grayscale, complex and color images are treated differently:
    # (B, C, H, W) -> (H, W, C) if C is not in { 1, 2 }
    #              -> (H, W)    otherwise
    assert output.shape[0] == tensor.shape[2]
    assert output.shape[1] == tensor.shape[3]
    if input_shape[1] not in [1, 2]:
        assert output.ndim == 3, "Output should be 3D for color images."
        assert output.shape[2] == (tensor.shape[1] if input_shape[1] != 2 else 1)
    else:
        assert output.ndim == 2, "Output should be 2D for grayscale or complex images."

    # Values clamped to [0, 1]
    assert np.all(output >= 0) and np.all(
        output <= 1
    ), "Output values should be in the range [0, 1]."


# A list of tuples: (command_runs, n_gpus, freer_gpu_index)
@pytest.mark.parametrize(
    "test_case",
    [
        # Case 1: The nvidia-smi command succeeds.
        (True, 2, 1),
        # Case 2: The command fails and torch.cuda.device_count() is zero.
        (False, 0, None),
        # Case 3: The command fails and torch.cuda.device_count() is positive.
        (False, 2, 1),
    ],
)
# NOTE: The nvidia-smi command is executed differently depending on the OS, we
# make sure that the logic is right.
@pytest.mark.parametrize("os_name", ["posix", "nt"])
@pytest.mark.parametrize("verbose", [False, True])
def test_get_freer_gpu(test_case, os_name, verbose):
    # The function get_freer_gpu is meant to return the torch.device associated
    # to the available GPU with the most free memory if there is one and
    # torch.device("cuda") as a fallback.
    # It works by first trying to run a `nvidia-smi` command and if it fails
    # it falls back to using the functions `torch.cuda.device_count` and
    # `torch.cuda.mem_get_info`. We mock the three components to control the
    # expected return value. In this scenario, we consider is a machine with n
    # GPUs, all of which have 1 MiB of total memory, and all of which have 0
    # MiB of free memory except for a single GPU (the freer GPU). We also
    # consider that the command
    # nvidia-smi might or might not be present in the system, resulting in
    # a failure of the function `subprocess.run` used in the implementation.
    command_runs, n_gpus, freer_gpu_index = test_case

    if freer_gpu_index is not None:
        assert (
            0 <= freer_gpu_index and freer_gpu_index < n_gpus
        ), "freer_gpu_index should be a valid index within the range of available GPUs."
    else:
        assert n_gpus == 0, "freer_gpu_index should be None only when n_gpus is 0."

    with (
        patch.object(subprocess, "run") as mock_subprocess_run,
        patch.object(torch.cuda, "device_count") as mock_device_count,
        patch.object(torch.cuda, "mem_get_info") as mock_mem_get_info,
        patch.object(os, "name", os_name),
    ):
        if command_runs:
            # Mock the standard output reported by subprocess.run by simulating
            # the output of the command used in the implementation:
            # nvidia-smi -q -d Memory | grep -A5 GPU | grep Free
            mock_subprocess_run.return_value.stdout = "\n".join(
                [
                    f"Free : 1 MiB" if idx == freer_gpu_index else "Free : 0 MiB"
                    for idx in range(n_gpus)
                ]
            )
        else:
            mock_subprocess_run.side_effect = FileNotFoundError
        mock_device_count.return_value = n_gpus

        def mem_info_mock(idx):
            total_mem = 1048576  # 1 MiB
            if idx == freer_gpu_index:
                free_mem = total_mem
            elif idx < n_gpus:
                free_mem = 0
            else:
                raise ValueError("Invalid GPU index")
            return (free_mem, total_mem)

        mock_mem_get_info.side_effect = mem_info_mock

        device = deepinv.utils.get_freer_gpu(verbose)
        assert isinstance(device, torch.device), "Device should be a torch device."
        assert device.type == "cuda", "Device should be a CUDA device."
        if n_gpus == 0:
            assert (
                device.index is None
            ), "Selected GPU index should be None when no GPUs are available."
        else:
            assert (
                device.index == freer_gpu_index
            ), f"Selected GPU index should be {freer_gpu_index}."


@pytest.mark.parametrize(
    "fn_name", ["norm", "cal_angle", "cal_mse", "complex_abs", "norm_psnr"]
)
def test_deprecated_metric_functions(fn_name):
    f = getattr(deepinv.utils.metric, fn_name)
    with pytest.raises(NotImplementedError, match="deprecated"):
        # The functions take a variable number of required arguments so we
        # use reflection to get their number and pass in None for each of them.
        sig = inspect.signature(f)
        args = [
            None
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        f(*args)


@pytest.mark.parametrize("with_data_dir", [False, True])
@pytest.mark.parametrize("data_dir_type", [str, pathlib.Path])
@pytest.mark.parametrize("name", ["Levin09.npy"])
@pytest.mark.parametrize("index", [1])
@pytest.mark.parametrize("download", [False, True])
def test_load_degradation(tmpdir, with_data_dir, data_dir_type, name, index, download):
    if with_data_dir:
        assert data_dir_type in [
            str,
            pathlib.Path,
        ], "data_dir_type should be str or pathlib.Path."
        data_dir = data_dir_type(tmpdir)
    else:
        data_dir = None

    args = [name, data_dir]
    kwargs = {"index": index}

    # We make sure the degradation is present on disk if download is False.
    if not download:
        _ = deepinv.utils.load_degradation(*args, **kwargs, download=True)

    kernel_torch = deepinv.utils.load_degradation(*args, **kwargs, download=download)
    assert isinstance(kernel_torch, torch.Tensor), "Kernel should be a torch tensor."


@pytest.mark.parametrize("n_retrievals", [1, 2])
@pytest.mark.parametrize("dataset_name", ["set3c"])
@pytest.mark.parametrize(
    "transform",
    [None, transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()])],
)
def test_load_dataset(n_retrievals, dataset_name, transform):
    # If a transform function is provided, mock it for further testing
    if transform is not None:
        transform = mock.Mock(wraps=transform)

    dataset = deepinv.utils.load_dataset(dataset_name, transform=transform)

    assert isinstance(dataset, torchvision.datasets.ImageFolder)

    for k in range(n_retrievals):
        x, y = dataset[k]
        # NOTE: We assume that the transform always converts the image to a
        # tensor if it is provided.
        if transform is not None:
            assert isinstance(x, torch.Tensor), "Dataset image should be a tensor."
        else:
            assert isinstance(
                x, PIL.Image.Image
            ), "Dataset image should be a PIL Image."
        assert isinstance(y, int), "Dataset label should be an integer."

    if transform is not None:
        assert (
            transform.call_count == n_retrievals
        ), "Transform should be called once for each dataset item."


@pytest.mark.parametrize(
    "operation", ["super-resolution", "deblur", "inpaint", "dummy"]
)
@pytest.mark.parametrize("noise_level_img", [0, 0.03])
def test_get_GSPnP_params(operation, noise_level_img):
    supported_operations = ["super-resolution", "deblur", "inpaint"]
    with (
        pytest.raises(ValueError)
        if operation not in supported_operations
        else nullcontext()
    ):
        lamb, sigma_denoiser, stepsize, max_iter = deepinv.utils.get_GSPnP_params(
            operation, noise_level_img
        )
        assert isinstance(lamb, float), "Lambda should be a float."
        assert lamb > 0, "Lambda should be positive."
        assert isinstance(sigma_denoiser, float), "Sigma denoiser should be a float."
        assert sigma_denoiser >= 0, "Sigma denoiser should be non-negative."
        assert isinstance(stepsize, float), "Stepsize should be a float."
        assert stepsize > 0, "Stepsize should be positive."
        assert isinstance(max_iter, int), "Max iterations should be an integer."
        assert max_iter > 0, "Max iterations should be positive."


# Module-level fixtures
pytestmark = [pytest.mark.usefixtures("non_interactive_matplotlib")]
