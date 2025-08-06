import deepinv
import torch
import pytest
from deepinv.utils.decorators import _deprecated_alias
import warnings
import numpy as np
import contextlib
from contextlib import nullcontext
import random
import unittest.mock as mock
from unittest.mock import patch
import subprocess
import os
import inspect
import pathlib
import torchvision
import torchvision.transforms as transforms
import PIL
import io
import copy
import math

# NOTE: It's used as a fixture.
from conftest import non_blocking_plots  # noqa: F401


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


# The class TensorList features many utility methods that we do not test in
# depth but verify that they do not raise any exception when called. To do
# that, we get a tensor list instance, we iterate over its methods and try to call
# them filling in the required parameters in the process. We use the name of
# the parameter to determine what to pass in, e.g., if the parameter is named
# shape we use the input tensor list shape as the value for that parameter. By
# default we use the tensor list itself for every optional parameter.
def test_tensorlist_methods(tensorlist):
    x, y = tensorlist
    parameter_map = {
        "shape": x.shape,
        "dim": 0,
        "device": x[0].device,
        "dtype": x[0].dtype,
    }

    for method_name, method in inspect.getmembers(x, predicate=inspect.ismethod):
        # Ignore dunder methods
        if method_name.startswith("__") and method_name.endswith("__"):
            continue

        # Ignore methods that assume a GPU is available if there is none
        if method_name == "cuda" and not torch.cuda.is_available():
            continue

        sig = inspect.signature(method)

        # Use the tensor list itself for every required argument
        args = [
            parameter_map[p.name] if p.name in parameter_map else x
            for p in sig.parameters.values()
            # Both conditions are needed to deal with *args and **kwargs
            # but also not to include parameters that are not required
            if p.default is p.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]

        # Test that the method does not raise any exception
        # NOTE: We run the method on a copy of the object to avoid side effects
        x_copy = copy.deepcopy(x)
        _ = getattr(x_copy, method_name)(*args)


@pytest.mark.parametrize("shape", [(1, 1, 3, 3), (1, 1, 5, 5)])
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5])
def test_dirac_like(shape, length):
    rng = torch.Generator().manual_seed(0)
    x = [torch.randn(shape, generator=rng) for _ in range(length)]
    h = deepinv.utils.dirac_like(x)
    y = deepinv.utils.TensorList(
        [
            deepinv.physics.functional.conv2d(xi, hi, padding="circular")
            for hi, xi in zip(h, x, strict=True)
        ]
    )

    for xi, hi, yi in zip(x, h, y, strict=True):
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
    tmp_path,
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
    img_list = {k: v for k, v in zip(titles, img_list, strict=True)}
    if not with_titles:
        titles = None
    if not dict_img_list:
        img_list = list(img_list.values())
    save_dir = tmp_path if save_plot else None
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
def test_scatter_plot(tmp_path, n_plots, titles, save_plot, show, suptitle):
    xy_list = torch.randn(100, 2, generator=torch.Generator().manual_seed(0))
    xy_list = [xy_list] * n_plots if n_plots > 1 else xy_list
    if titles is not None:
        titles = [titles] * n_plots if n_plots > 1 else titles
    save_dir = tmp_path if save_plot else None
    deepinv.utils.scatter_plot(
        xy_list, titles=titles, suptitle=suptitle, save_dir=save_dir, show=show
    )


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("n_metrics", [1, 2])
@pytest.mark.parametrize("n_batches", [1, 2])
@pytest.mark.parametrize("n_iterations", [1, 2])
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("show", [False, True])
def test_plot_curves(
    tmp_path, seed, n_metrics, n_batches, n_iterations, save_plot, show
):
    rng = random.Random(seed)
    metrics = {
        str(k): [[rng.random() for _ in range(n_iterations)] for _ in range(n_batches)]
        for k in range(n_metrics)
    }
    save_dir = tmp_path if save_plot else None
    deepinv.utils.plot_curves(metrics, save_dir=save_dir, show=show)


@pytest.mark.parametrize("init_params", [None])
@pytest.mark.parametrize("save_plot", [False, True])
@pytest.mark.parametrize("show", [False, True])
def test_plot_parameters(tmp_path, model, init_params, save_plot, show):
    save_dir = tmp_path if save_plot else None
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
@pytest.mark.parametrize("use_torch_api", [False, True])
@pytest.mark.parametrize("hide_warnings", [False, True])
def test_get_freer_gpu(test_case, os_name, verbose, use_torch_api, hide_warnings):
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

        device = deepinv.utils.get_freer_gpu(
            verbose=verbose, use_torch_api=use_torch_api, hide_warnings=hide_warnings
        )
        if n_gpus == 0:
            assert device is None, "The output should be None when no GPU is available."
        else:
            assert isinstance(device, torch.device), "Device should be a torch device."
            assert device.type == "cuda", "Device should be a CUDA device."
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
def test_load_degradation(
    tmp_path, with_data_dir, data_dir_type, name, index, download
):
    if with_data_dir:
        assert data_dir_type in [
            str,
            pathlib.Path,
        ], "data_dir_type should be str or pathlib.Path."
        data_dir = data_dir_type(tmp_path)
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


@pytest.mark.parametrize("rng", [random.Random(0)])
@pytest.mark.parametrize("n_meters", [1, 2])
@pytest.mark.parametrize("n_updates", [10])
@pytest.mark.parametrize("fmt", ["f"])
@pytest.mark.parametrize(
    "epoch, num_epochs",
    [
        (0, 5),
        (5, 5),
        (5, 10),
        (10, 15),
        (37, 100),
        (150, 150),
    ],
)
@pytest.mark.parametrize("surfix", ["", "dummy_suffix"])
@pytest.mark.parametrize("prefix", ["", "dummy_prefix"])
def test_ProgressMeter(
    rng, n_meters, n_updates, fmt, epoch, num_epochs, surfix, prefix
):
    meters = [
        deepinv.utils.AverageMeter(f"dummy_meter{i + 1}", fmt=f":{fmt}")
        for i in range(n_meters)
    ]

    for meter in meters:
        for _ in range(n_updates):
            meter.update(rng.random())

    progress = deepinv.utils.ProgressMeter(
        num_epochs, meters, surfix=surfix, prefix=prefix
    )

    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        progress.display(epoch)

    stdout = stdout_buf.getvalue()
    stdout = stdout.strip()
    # NOTE: For some reason prefix is appended at the end and surfix at the
    # beginning. Is it a bug?
    assert stdout.endswith(prefix), "Prefix should be at the end of the output."
    assert stdout.startswith(surfix), "Surfix should be at the beginning of the output."

    assert str(epoch) in stdout, "Epoch number should be in the output."
    assert str(num_epochs) in stdout, "Number of epochs should be in the output."

    for meter in meters:
        assert (
            meter.name in stdout
        ), f"Meter name '{meter.name}' should be in the output."
        assert (
            f"{meter.avg:{fmt}}" in stdout
        ), f"Meter average '{meter.avg}' should be in the output."


@pytest.mark.parametrize("original_size", [(16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("grayscale", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("img_size", [None, 32, (32, 32)])
@pytest.mark.parametrize("resize_mode", ["crop", "resize"])
def test_load_image(
    tmp_path, device, original_size, grayscale, dtype, img_size, resize_mode
):
    # We use a mocked PIL image to test the load_image function.
    im = PIL.Image.new("RGB", original_size, color=(0, 0, 0))
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    buffer.seek(0)
    im = PIL.PngImagePlugin.PngImageFile(buffer)

    with patch.object(PIL.Image, "open", return_value=im):
        x = deepinv.utils.load_image(
            f"{tmp_path}/im.png",
            grayscale=grayscale,
            device=device,
            dtype=dtype,
            img_size=img_size,
            resize_mode=resize_mode,
        )
        assert isinstance(x, torch.Tensor), "Loaded image should be a tensor."
        if img_size is not None:
            img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
            assert (
                x.shape[-2:] == img_size
            ), f"Image shape should be {img_size}, got {x.shape[-2:]}"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "signal_shape",
    [(3, 16, 16), (1, 16, 16), (1, 16), (1, 16, 16, 16), (1, 16, 8), (16, 16), (16,)],
)
@pytest.mark.parametrize("mode", ["min_max", "clip"])
@pytest.mark.parametrize("seed", [0])
def test_normalize_signals(batch_size, signal_shape, mode, seed):
    shape = (batch_size, *signal_shape)
    rng = torch.Generator().manual_seed(seed)

    # Generate a batch of random signals, half constant and half not
    # NOTE: Constant signals are the main edge case to test.
    inp = torch.empty(shape, device="cpu", dtype=torch.float32)
    indices = torch.randperm(batch_size, generator=rng)
    N_const_idx = math.ceil(batch_size / 2)
    const_idx, var_idx = indices[:N_const_idx], indices[N_const_idx:]
    const_values = torch.randn(
        N_const_idx, generator=rng, device=inp.device, dtype=inp.dtype
    )
    inp[const_idx] = const_values.view((-1,) + ((1,) * len(signal_shape)))
    if var_idx.numel() != 0:
        inp[var_idx] = torch.randn(
            inp[const_idx].shape, generator=rng, device=inp.device, dtype=inp.dtype
        )

    # Sanity check
    assert inp.shape == shape, "Input tensor should have the specified shape."

    # Apply the tested function
    out = deepinv.utils.normalize_signal(inp, mode=mode)

    # Check the tensor attributes
    assert out.dtype == inp.dtype, "Output dtype should match input dtype."
    assert out.device == inp.device, "Output device should match input device."
    assert out.shape == inp.shape, "Output shape should match input shape."

    # Check that the output entries are between zero and one
    assert torch.all(0 <= out) and torch.all(
        out <= 1
    ), "Output entries should be in [0, 1]."

    # Tests specific to min-max normalization
    if mode == "min_max":
        # Test the edge case of constant signals
        for inp_s, out_s in zip(inp, out, strict=True):
            inp_unique = torch.unique(inp_s)
            is_inp_constant = inp_unique.numel() == 1
            if is_inp_constant:
                # Verify that constant signals remain constant after normalization
                out_unique = torch.unique(out_s)
                is_out_constant = out_unique.numel() == 1
                assert (
                    is_out_constant
                ), "Output should be constant if input is constant."

                # Input and output constant values
                inp_c = inp_unique.item()
                out_c = out_unique.item()

                # Verify that the rescaling is the smallest possible
                target_c = max(0, min(1, inp_c))
                assert (
                    out_c == target_c
                ), "The distance between the input and ouput constants is not minimal."
    elif mode == "clip":
        # Check that the input is clipped between zero and one
        assert torch.all(
            out == torch.clamp(inp, 0, 1)
        ), "Output should be clipped between 0 and 1."
    else:
        raise ValueError(
            f"Unknown mode '{mode}'. Supported modes are 'min_max' and 'clip'."
        )


# Module-level fixtures
pytestmark = [pytest.mark.usefixtures("non_blocking_plots")]
