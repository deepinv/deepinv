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


def get_git_root():
    import git

    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_degradation_url(file_name: str) -> str:
    """Get URL for degradation from DeepInverse HuggingFace repository.

    :param str file_name: degradation filename in repository
    :return: degradation URL
    """
    return (
        "https://huggingface.co/datasets/deepinv/degradations/resolve/main/"
        + file_name
        + "?download=true"
    )


def get_image_url(file_name: str) -> str:
    """Get URL for image from DeepInverse HuggingFace repository.

    :param str file_name: image filename in repository
    :return str: image URL
    """
    return (
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        + file_name
        + "?download=true"
    )


def get_data_home() -> Path:
    """Return a folder to store deepinv datasets.

    This folder can be specified by setting the environment variable``DEEPINV_DATA``,
    or ``XDG_DATA_HOME``. By default, it is ``./datasets``.

    :return: pathlib Path for data home
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
    data_dir: Union[str, Path] = None,
    download: bool = True,
    url: str = None,
    file_type: str = "zip",
) -> Dataset:
    """Loads an ImageFolder dataset from DeepInverse HuggingFace repository.

    :param str, pathlib.Path dataset_name: dataset name without file extension.
    :param Callable transform: optional transform to pass to torchvision dataset.
    :param str, pathlib.Path data_dir: dataset root directory, defaults to None
    :param bool download: whether to download, defaults to True
    :param str url: download URL, if ``None``, gets URL using :func:`deepinv.utils.get_image_url`
    :param str file_type: file extension, defaults to "zip"
    :return: torchvision ImageFolder dataset.
    """
    if data_dir is None:
        data_dir = get_data_home()

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    if "fastmri" in dataset_name:
        raise ValueError(
            "Loading singlecoil fastmri with load_dataset is now deprecated. Please use deepinv.datasets.SimpleFastMRISliceDataset(download=True)."
        )

    dataset_dir = data_dir / dataset_name

    if download and not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)

        if url is None:
            url = get_image_url(f"{str(dataset_name)}.{file_type}")

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

        with zipfile.ZipFile(str(dataset_dir) + ".zip") as zip_ref:
            zip_ref.extractall(str(data_dir))

        os.remove(f"{str(dataset_dir)}.{file_type}")
        print(f"{dataset_name} dataset downloaded in {data_dir}")

    return torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)


def load_degradation(
    name: Union[str, Path],
    data_dir: Union[str, Path] = None,
    index: int = 0,
    download: bool = True,
) -> torch.Tensor:
    """Loads a degradation tensor from DeepInverse HuggingFace repository.

    :param str, pathlib.Path name: degradation name with file extension
    :param str, pathlib.Path data_dir: dataset root directory, defaults to None
    :param int index: degradation index, defaults to 0
    :param bool download: whether to download, defaults to True
    :return: (:class:`torch.Tensor`) containing degradation.
    """
    if data_dir is None:
        data_dir = get_data_home()

    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    path = data_dir / name

    if download and not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = get_degradation_url(name)
        with requests.get(url, stream=True) as r:
            with open(str(data_dir / name), "wb") as f:
                shutil.copyfileobj(r.raw, f)
        print(f"{name} degradation downloaded in {data_dir}")

    deg = np.load(path, allow_pickle=True)
    return torch.from_numpy(deg[index])


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


def load_example(name, **kwargs):
    r"""
    Load example image from the `DeepInverse HuggingFace <https://huggingface.co/datasets/deepinv/images>`_ using :func:`deepinv.utils.load_url_image`.

    :param str name: filename of the image from the HuggingFace dataset.
    :param dict kwargs: keyword args to pass to :func:`deepinv.utils.load_url_image`
    :return: :class:`torch.Tensor` containing the image.
    """
    return load_url_image(get_image_url(name), **kwargs)


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
