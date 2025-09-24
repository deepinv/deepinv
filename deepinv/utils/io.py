from typing import Union
from pathlib import Path
from io import BytesIO
import requests
import numpy as np
import torch

# TODO user guide and API for all loaders here + in demo
# TODO docs and docstrings
# TODO tests (pydicom, nibabel, mat73, scipy to datasets op deps)


def load_dicom(path: Union[str, Path]) -> torch.Tensor:
    """Load image from DICOM file.

    Requires `pydicom` to be installed. Install it with `pip install pydicom`.

    :param str, Path path: path to DICOM file.
    :return: torch float tensor of shape `(1, ...)` where `...` are the DICOM image dimensions.
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError(
            "load_dicom requires pydicom, which is not installed. Please install it with `pip install pydicom`."
        )
    return torch.from_numpy(pydicom.dcmread(str(path)).pixel_array).float().unsqueeze(0)


def load_nifti(path: Union[str, Path]) -> torch.Tensor:
    """Load image from NIFTI `.nii.gz` file.

    Requires `nibabel` to be installed. Install it with `pip install nibabel`.

    :param str, Path path: path to NIFTI `.nii.gz` file.
    :return: torch float tensor of shape `(1, ...)` where `...` are the NIFTI image dimensions.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError(
            "load_nifti requires nibabel, which is not installed. Please install it with `pip install nibabel`."
        )
    return torch.from_numpy(nib.load(path).get_fdata()).float().unsqueeze(0)


# TODO load h5 ISMRM
# TODO load GDAL COSAR
# TODO load GEOTIFF


def load_torch(path: Union[str, Path], device=None) -> torch.Tensor:
    """Load torch tensor from file.

    :param str, pathlib.Path path: file to load.
    :param torch.device, str device: device to load onto.
    :return: :class:`torch.Tensor` containing loaded torch tensor.
    """
    return torch.load(path, weights_only=True, map_location=device)


def load_np(path: Union[str, Path]) -> torch.Tensor:
    """Load numpy array from file as torch tensor.

    :param str, pathlib.Path path: file to load.
    :return: :class:`torch.Tensor` containing loaded numpy array.
    """
    return torch.from_numpy(np.load(path, allow_pickle=False))


def load_url(url: str) -> BytesIO:
    """Load URL to a buffer.

    This can be used as the argument for other IO functions such as
    :func:`deepinv.utils.load_torch`, :func:`deepinv.utils.load_np` etc.
    to load data directly from a URL.

    :param str url: URL of the file to load
    :return: `BytesIO` buffer.
    """
    response = requests.get(url)
    response.raise_for_status()
    return BytesIO(response.content)


def load_mat(fname: str, mat73: bool = False) -> dict[str, np.ndarray]:
    """Load MATLAB array from file.

    This function depends on the ``scipy`` package. You can install it with ``pip install scipy``.

    :param str, pathlib.Path fname: filename to load
    :param bool mat73: if file is MATLAB 7.3 or above, load with ``mat73``. Requires
        ``mat73``, install with ``pip install mat73``.
    :return: dict with str keys and numpy array values.
    """
    if mat73:
        try:
            from mat73 import loadmat as loadmat73

            return loadmat73(fname)
        except ImportError:
            raise ImportError("mat73 is required, install with 'pip install mat73'.")
        except TypeError:
            pass

    try:
        from scipy.io import loadmat as scipy_loadmat

        return scipy_loadmat(fname)
    except ImportError:
        raise ImportError(
            "load_mat requires scipy, which is not installed. Install it with `pip install scipy`."
        )
