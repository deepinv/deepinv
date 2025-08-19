from __future__ import annotations
from typing import Union, Callable, TYPE_CHECKING
import os, shutil, zipfile, requests
from io import BytesIO

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
from torchvision import transforms

if TYPE_CHECKING:
    from deepinv.datasets.base import ImageFolder


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


def get_image_url(file_name: str, dataset: str = "images") -> str:
    """Get URL for image from DeepInverse HuggingFace repository.

    :param str file_name: image filename in repository
    :param str dataset: HuggingFace dataset name, defaults to 'images'
    :return str: image URL
    """
    return f"https://huggingface.co/datasets/deepinv/{dataset}/resolve/main/{file_name}?download=true"


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
) -> ImageFolder:
    """Loads an ImageFolder dataset from DeepInverse HuggingFace repository.

    :param str, pathlib.Path dataset_name: dataset name without file extension.
    :param Callable transform: optional transform to pass to torchvision dataset.
    :param str, pathlib.Path data_dir: dataset root directory, defaults to None
    :param bool download: whether to download, defaults to True
    :param str url: download URL, if ``None``, gets URL using :func:`deepinv.utils.get_image_url`
    :param str file_type: file extension, defaults to "zip"
    :return: :class:`deepinv.datasets.ImageFolder` dataset.
    """
    from deepinv.datasets.base import ImageFolder

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

    return ImageFolder(root=dataset_dir, transform=transform)


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


def load_url(url: str) -> BytesIO:
    """Load URL to a buffer.

    :param str url: URL of the file to load
    """
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)


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

    img = Image.open(load_url(url))
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
    Load example image from the `DeepInverse HuggingFace <https://huggingface.co/datasets/deepinv/images>`_ using
    :func:`deepinv.utils.load_url_image` if image file or :func:`deepinv.utils.load_torch_url` if torch tensor in `.pt` file
    or :func:`deepinv.utils.load_np_url` if numpy array in `npy` or `npz` file.

    Available examples for `name` include (see `the HuggingFace repo <https://huggingface.co/datasets/deepinv/images>`_ for full list):

    .. list-table:: Example Images
      :header-rows: 1

      * - Name
        - Origin
        - Image size
        - Domain
      * - `barbara.jpeg`, `butterfly.png`
        - :class:`Set14 <deepinv.datasets.Set14HR>`
        - (3, 512, 512), (3, 256, 256)
        - natural
      * - `cameraman.png`
        - Classic toy image
        - (1, 512, 512)
        - natural
      * - `CBSD_0010.png`
        - :class:`CBSD68 <deepinv.datasets.CBSD68>`
        - (2, 481, 321)
        - natural
      * - `celeba_example.jpg`
        - CelebA
        - (3, 1024, 1024)
        - natural
      * - `div2k_valid_hr_0877.png`, `div2k_valid_lr_bicubic_0877x4.png`
        - GT and measurement from :class:`Div2k <deepinv.datasets.DIV2K>`
        - (3, 1152, 2040), (3, 288, 510)
        - natural
      * - `leaves.png`
        - Set3C dataset
        - (3, 256, 256)
        - natural
      * - `mbappe.jpg`
        -
        - (3, 443, 664)
        - natural
      * - `CT100_256x256_0.pt`
        - `CT100 <https://doi.org/10.1007/s10278-013-9622-7>`_
        - (1, 256, 256)
        - medical
      * - `brainweb_t1_ICBM_1mm_subject_0.npy`
        - `BrainWeb <https://brainweb.bic.mni.mcgill.ca/brainweb/>`_ 3D MRI data
        - (181, 217, 181)
        - medical
      * - `demo_mini_subset_fastmri_brain_0.pt`
        - :class:`FastMRI <deepinv.datasets.SimpleFastMRISliceDataset>`
        - (2, 320, 320)
        - medical
      * - `SheppLogan.png`
        - Shepp Logan phantom
        - (4, 512, 512)
        - medical
      * - `FMD_TwoPhoton_MICE_R_gt_12_avg50.png`
        - :class:`FMD <deepinv.datasets.FMD>`
        - (3, 512, 512)
        - microscopy
      * - `JAX_018_011_RGB.tif`
        - Sample RGB patch from WorldView-3
        - (3, 1024, 1024)
        - satellite


    :param str name: filename of the image from the HuggingFace dataset.
    :param dict kwargs: keyword args to pass to :func:`deepinv.utils.load_url_image`
    :return: :class:`torch.Tensor` containing the image.
    """
    url = get_image_url(name)

    if name.split(".")[-1].lower() == "pt":
        return load_torch_url(url, **kwargs)
    elif name.split(".")[-1].lower() in ("npy", "npz"):
        return load_np_url(url, **kwargs)

    return load_url_image(url, **kwargs)


def download_example(name: str, save_dir: Union[str, Path]):
    r"""
    Download an image from the `DeepInverse HuggingFace <https://huggingface.co/datasets/deepinv/images>`_ to file.

    For all available examples, see :func:`deepinv.utils.load_example`.

    :param str name: filename of the image from the HuggingFace dataset.
    :param str, pathlib.Path save_dir: directory to save image to.
    """
    os.makedirs(save_dir, exist_ok=True)
    data = requests.get(get_image_url(name)).content
    with open(Path(save_dir) / name, "wb") as f:
        f.write(data)


def load_torch_url(url, device="cpu", **kwargs):
    r"""
    Load an array from url and read it by torch.load.

    :param str url: URL of the image file.
    :param str, torch.device device: Device on which to load the tensor.
    :return: whatever is pickled in the file.
    """
    return torch.load(load_url(url), weights_only=True, map_location=device)


def load_np_url(url=None, **kwargs):
    r"""
    Load a numpy array from url.

    :param str url: URL of the image file.
    :return: numpy ndarray containing the data.
    """
    return np.load(load_url(url), allow_pickle=False)
