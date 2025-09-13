from typing import Union
from pathlib import Path
from io import BytesIO
import requests
import numpy as np
import torch

# TODO user guide and API for all loaders here + in demo
# TODO move over load_image and load_url_image?
# TODO promote load_url as can use with any of the below
# TODO docs and docstrings


def load_dicom(path: Union[str, Path]) -> torch.Tensor:
    """Load image from DICOM file.

    Requires `pydicom` to be installed.

    :param str, Path path: path to DICOM file.
    :return: torch float tensor of shape `(B, 1, ...)` where `...` are the DICOM image dimensions.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "load_dicom requires pydicom, which is not installed. Please install it with `pip install pydicom`."
        )
    return torch.from_numpy(pydicom.dcmread(str(path)).pixel_array).float().unsqueeze(0)


def load_nifti(path: Union[str, Path]) -> torch.Tensor:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "load_nifti requires nibabel, which is not installed. Please install it with `pip install nibabel`."
        )
    return torch.from_numpy(nib.load(path).get_fdata())


# TODO loadmat, loadmat scipy
# TODO load h5 ISMRM
# TODO load GDAL COSAR
# TODO load GEOTIFF


def load_torch(path: Union[str, Path], device=None) -> torch.Tensor:
    return torch.load(path, weights_only=True, map_location=device)


def load_np(path: Union[str, Path]) -> torch.Tensor:
    return torch.from_numpy(np.load(path, allow_pickle=False))


def load_url(url: str) -> BytesIO:
    """Load URL to a buffer.

    :param str url: URL of the file to load
    """
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)
