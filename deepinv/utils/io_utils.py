from typing import Callable, Optional, Union, Iterator
from pathlib import Path
from warnings import warn
from io import BytesIO
import requests
import numpy as np
import torch
from deepinv.utils.mixins import MRIMixin

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


def load_raster(
    filepath: str,
    patch: Union[bool, int, tuple[int, int]] = False,
    transform: Optional[Callable] = None,
) -> Union[torch.Tensor, Iterator[torch.Tensor]]:
    """
    Load a raster image and return patches as tensors using `rasterio`.

    This function allows you to stream patches from large rasters e.g. satellite imagery, SAR etc.

    :param str filepath: Path to the raster file, such as `.geotiff`, `.tiff`, `.cos` etc.
    :param bool, int, tuple[int, int], patch: Patch extraction mode.
        * ``False`` (default): return the entire image as a :class:`torch.Tensor` of shape `(C, H, W)` where C are bands.
        * ``True``: yield patches based on the raster's internal block windows.
            - If no block windows are available, raises ``RuntimeError``.
            - If any block has a dimension of 1 (strip layout), a warning is raised.
        * ``int`` or ``(int, int)``: yield patches of the manually specified size.
    :param Callable, None transform: Optional transform applied to each patch.

    :return: Either (where C is the band dimension)
        * a full image :class:`torch.Tensor` of shape `(C, H, W)`, if ``patch=False``, or
        * an iterator of torch tensors over patches of shape `(C, h, w)`, if ``patch=True`` or a size is specified, where `h,w` is the patch size.

    |sep|

    :Examples:

    >>> assert 1==0
    >>> from deepinv.utils.io import load_raster, load_url
    >>> file = load_url("https://download.osgeo.org/geotiff/samples/spot/chicago/SP27GTIF.TIF")
    >>> x = load_raster(file, patch=False) # Load whole image
    >>> x.shape
    torch.Size([1, 929, 699])
    >>> x = load_raster(file, patch=True) # Patch via internal block size
    >>> next(x).shape
    torch.Size([1, 11, 699])
    >>> all_patches = list(x) # Load all patches into memory
    >>> len(all_patches)
    43
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(all_patches, batch_size=2) # You can use this for training
    >>>
    >>> x = load_raster(file, patch=128) # Patch via manual size
    >>> next(x).shape
    torch.Size([1, 128, 128])
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError(
            "load_raster requires rasterio, which is not installed. Install it with `pip install rasterio`."
        )

    def _process_array(arr: np.ndarray) -> torch.Tensor:
        """Validate, normalize dimensions, handle complex data, and apply transform."""
        arr: torch.Tensor = torch.from_numpy(arr).squeeze(
            0
        )  # Remove single-band dimension if exists
        if arr.ndim == 2:
            if arr.is_complex():
                arr = MRIMixin.from_torch_complex(arr.unsqueeze(0)).squeeze(
                    0
                )  # (2, H, W)
            else:
                arr = arr.unsqueeze(0)  # (1, H, W)
        elif arr.ndim == 3:
            if arr.is_complex():
                raise RuntimeError(
                    "If array is complex, it cannot have a band dimension."
                )
        else:
            raise RuntimeError("Array should have band, x and y dimensions.")

        if transform:
            arr = transform(arr)
        return arr.float()  # (C, H, W)

    if patch is False:
        with rasterio.open(filepath) as src:
            return _process_array(src.read())

    def _patch_generator() -> Iterator[torch.Tensor]:
        with rasterio.open(filepath) as src:

            if patch is True:
                block_windows = list(
                    src.block_windows(1)
                )  # use band 1 as representative
                if not block_windows:
                    raise RuntimeError("No block windows available for this raster.")
                for _, window in block_windows:
                    w, h = window.width, window.height
                    if w == 1 or h == 1:
                        warn(
                            f"Block window {window} returns patches of width or height 1. Consider manually setting a patch size."
                        )
                    yield _process_array(src.read(window=window))
                return

            elif isinstance(patch, int):
                patch_w, patch_h = patch, patch
            elif isinstance(patch, tuple) and len(patch) == 2:
                patch_w, patch_h = patch
            else:
                raise ValueError(f"Invalid value for patch: {patch}")

            for y in range(0, src.height, patch_h):
                for x in range(0, src.width, patch_w):
                    window = rasterio.windows.Window(
                        x,
                        y,
                        min(patch_w, src.width - x),
                        min(patch_h, src.height - y),
                    )
                    yield _process_array(src.read(window=window))
            return

    return _patch_generator()
