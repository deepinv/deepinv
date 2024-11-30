from typing import Union, Callable
import os, shutil, zipfile, requests
from io import BytesIO

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

from deepinv.datasets.fastmri import SimpleFastMRISliceDataset


def get_git_root():
    import git

    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_image_dataset_url(dataset_name, file_type="zip"):
    return (
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        + dataset_name
        + "."
        + file_type
        + "?download=true"
    )


def get_degradation_url(file_name):
    return (
        "https://huggingface.co/datasets/deepinv/degradations/resolve/main/"
        + file_name
        + "?download=true"
    )


def get_image_url(file_name):
    return (
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        + file_name
        + "?download=true"
    )


def get_data_home():
    """Return a folder to store deepinv datasets.

    This folder can be specified by setting the environment variable``DEEPINV_DATA``,
    or ``XDG_DATA_HOME``. By default, it is ``./datasets``.
    """
    data_home = os.environ.get("DEEPINV_DATA", None)
    if data_home is not None:
        return Path(data_home)

    data_home = os.environ.get("XDG_DATA_HOME", None)
    if data_home is not None:
        return Path(data_home) / "deepinv"

    return Path(".") / "datasets"


def load_dataset(
    dataset_name: Union[str, Path],
    transform: Callable,
    data_dir: Path = None,
    download: bool = True,
    url: str = None,
    train: bool = True,
) -> Dataset:
    # TODO add load_dataset to docs
    # TODO docstring
    # TODO robust checking of dataset_name especially for mri
    if data_dir is None:
        data_dir = get_data_home()

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    dataset_dir = data_dir / dataset_name
    if dataset_name in ("fastmri_knee_singlecoil"):
        file_type = "pt"
    else:
        file_type = "zip"
    if download and not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if url is None:
            url = get_image_dataset_url(dataset_name, file_type)
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        print("Downloading " + str(dataset_dir) + f".{file_type}")
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(str(dataset_dir) + f".{file_type}", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if file_type == "zip":
            with zipfile.ZipFile(str(dataset_dir) + ".zip") as zip_ref:
                zip_ref.extractall(str(data_dir))
            # remove temp file
            os.remove(str(dataset_dir) + f".{file_type}")
            print(f"{dataset_name} dataset downloaded in {data_dir}")
        else:
            shutil.move(
                str(dataset_dir) + f".{file_type}",
                str(dataset_dir / dataset_name) + f".{file_type}",
            )
    if dataset_name in ("fastmri_knee_singlecoil", "fastmri_brain"):
        dataset = SimpleFastMRISliceDataset(
            root_dir=dataset_dir / dataset_name, transform=transform, train=train
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir, transform=transform
        )
    return dataset


def load_degradation(name, data_dir=None, index=0, download=True):
    if data_dir is None:
        data_dir = get_data_home()
    path = data_dir / name
    if download and not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = get_degradation_url(name)
        with requests.get(url, stream=True) as r:
            with open(str(data_dir / name), "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print(f"{name} degradation downloaded in {data_dir}")
    deg = np.load(path, allow_pickle=True)
    deg_torch = torch.from_numpy(deg[index])  # .unsqueeze(0).unsqueeze(0)
    return deg_torch


def load_image(
    path,
    img_size=None,
    grayscale=False,
    resize_mode="crop",
    device="cpu",
    dtype=torch.float32,
):
    r"""
    Load an image from a file and return a torch.Tensor.

    :param str path: Path to the image file.
    :param int, tuple[int] img_size: Size of the image to return.
    :param bool grayscale: Whether to convert the image to grayscale.
    :param str resize_mode: If ``img_size`` is not None, options are ``"crop"`` or ``"resize"``.
    :param str device: Device on which to load the image (gpu or cpu).
    :return: :class:`torch.Tensor` containing the image.
    """
    img = Image.open(path)
    transform_list = []
    if img_size is not None:
        if resize_mode == "crop":
            transform_list.append(transforms.CenterCrop(img_size))
        elif resize_mode == "resize":
            transform_list.append(transforms.Resize(img_size))
        else:
            raise ValueError(
                f"resize_mode must be either 'crop' or 'resize', got {resize_mode}"
            )
    if grayscale:
        transform_list.append(transforms.Grayscale())
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    x = transform(img).unsqueeze(0).to(device=device, dtype=dtype)
    return x


def load_url_image(
    url=None,
    img_size=None,
    grayscale=False,
    resize_mode="crop",
    device="cpu",
    dtype=torch.float32,
):
    r"""
    Load an image from a URL and return a torch.Tensor.

    :param str url: URL of the image file.
    :param int, tuple[int] img_size: Size of the image to return.
    :param bool grayscale: Whether to convert the image to grayscale.
    :param str resize_mode: If ``img_size`` is not None, options are ``"crop"`` or ``"resize"``.
    :param str device: Device on which to load the image (gpu or cpu).
    :return: :class:`torch.Tensor` containing the image.
    """

    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    transform_list = []
    if img_size is not None:
        if resize_mode == "crop":
            transform_list.append(transforms.CenterCrop(img_size))
        elif resize_mode == "resize":
            transform_list.append(transforms.Resize(img_size))
        else:
            raise ValueError(
                f"resize_mode must be either 'crop' or 'resize', got {resize_mode}"
            )
    if grayscale:
        transform_list.append(transforms.Grayscale())
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    x = transform(img).unsqueeze(0).to(device=device, dtype=dtype)
    return x


def load_torch_url(url):
    r"""
    Load an array from url and read it by torch.load.

    :param str url: URL of the image file.
    :return: whatever is pickled in the file.
    """
    response = requests.get(url)
    response.raise_for_status()
    out = torch.load(BytesIO(response.content))
    return out


def load_np_url(url=None):
    r"""
    Load a numpy array from url.

    :param str url: URL of the image file.
    :return: :class:`np.array` containing the data.
    """
    response = requests.get(url)
    response.raise_for_status()
    array = np.load(BytesIO(response.content))
    return array


def demo_mri_model(device):
    """Demo MRI reconstruction model for use in relevant examples.

    As a reconstruction network, we use an unrolled network (half-quadratic splitting)
    with a trainable denoising prior based on the DnCNN architecture, as an example of a
    model-based deep learning architecture from `MoDL <https://ieeexplore.ieee.org/document/8434321>`_.

    :param str, torch.device device: device
    :return torch.nn.Module: model
    """
    from deepinv.optim.prior import PnP
    from deepinv.optim import L2
    from deepinv.models import DnCNN
    from deepinv.unfolded import unfolded_builder

    # Select the data fidelity term
    data_fidelity = L2()
    n_channels = 2  # real + imaginary parts

    # If the prior dict value is initialized with a table of length max_iter, then a distinct model is trained for each
    # iteration. For fixed trained model prior across iterations, initialize with a single model.
    prior = PnP(
        denoiser=DnCNN(
            in_channels=n_channels,
            out_channels=n_channels,
            pretrained=None,
            depth=7,
        ).to(device)
    )

    # Unrolled optimization algorithm parameters
    max_iter = 3  # number of unfolded layers
    lamb = [1.0] * max_iter  # initialization of the regularization parameter
    stepsize = [1.0] * max_iter  # initialization of the step sizes.
    sigma_denoiser = [0.01] * max_iter  # initialization of the denoiser parameters
    params_algo = {  # wrap all the restoration parameters in a 'params_algo' dictionary
        "stepsize": stepsize,
        "g_param": sigma_denoiser,
        "lambda": lamb,
    }

    trainable_params = [
        "lambda",
        "stepsize",
        "g_param",
    ]  # define which parameters from 'params_algo' are trainable

    # Define the unfolded trainable model.
    model = unfolded_builder(
        "HQS",
        params_algo=params_algo,
        trainable_params=trainable_params,
        data_fidelity=data_fidelity,
        max_iter=max_iter,
        prior=prior,
    )
    return model
