from __future__ import annotations
from typing import Callable, Iterator, TYPE_CHECKING
from pathlib import Path
from warnings import warn
from io import BytesIO
import requests
import numpy as np
from numpy.lib.format import open_memmap
import torch
from deepinv.utils.mixins import MRIMixin

if TYPE_CHECKING:
    import nibabel as nib
    import blosc2


def load_np(
    fname: str | Path,
    as_memmap: bool = False,
    dtype: np.dtype | str = np.float32,
    **kwargs,
) -> torch.Tensor | np.ndarray:
    """Load numpy array from file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memmap, which does not load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param numpy.dtype, str dtype: data type to use when loading the numpy array. This is ignored if `as_memmap` is `True`.
    :return: :class:`torch.Tensor` containing loaded numpy array. If `as_memmap` is `True`, returns a numpy `memmap` object instead.
    """
    if as_memmap:
        return open_memmap(fname)
    else:
        return torch.from_numpy(np.load(fname, allow_pickle=False).astype(dtype))


def load_torch(
    fname: str | Path, device: torch.device | str = None, **kwargs
) -> torch.Tensor:
    """Load torch tensor from file.

    :param str, pathlib.Path fname: file to load.
    :param torch.device, str device: device to load onto.
    :return: :class:`torch.Tensor` containing loaded torch tensor.
    """
    return torch.load(fname, weights_only=True, map_location=device)


def load_url(url: str, **kwargs) -> BytesIO:
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


def load_dicom(
    fname: str | Path,
    as_tensor: bool = True,
    apply_rescale: bool = False,
    dtype: np.dtype | str = np.float32,
    **kwargs,
) -> torch.Tensor | np.ndarray:
    """Load image from DICOM file.

    Requires `pydicom` to be installed. Install it with `pip install pydicom`.

    :param str, pathlib.Path fname: path to DICOM file or buffer.
    :param bool as_tensor: if `True`, return as torch tensor (default), otherwise return as numpy array.
    :param bool apply_rescale: if `True`, map the stored values (SV) to output values according to the pydicom `apply_rescale`, default False.
        See `pydicom docs <https://pydicom.github.io/pydicom/3.0/tutorials/pixel_data/introduction.html>`_
        and `apply_rescale <https://pydicom.github.io/pydicom/2.4/reference/generated/pydicom.pixel_data_handlers.apply_rescale.html>`_ for details.
        Note this is only useful when the appropriate dicom tags are present, for example in CT for :class:`deepinv.datasets.LidcIdriSliceDataset`
        for converting to Hounsfield Units. For other applications such as SUV/PET, we recommend applying the rescaling yourself.
    :param numpy.dtype, str dtype: data type to use when loading the nifti file.
    :return: either numpy array of shape of raw data `(...)`, or torch float tensor of shape `(1, ...)` where `...` are the DICOM image dimensions.
    """
    try:
        import pydicom
    except ImportError:  # pragma: no cover
        raise ImportError(
            "load_dicom requires pydicom, which is not installed. Please install it with `pip install pydicom`."
        )
    fname = str(fname) if isinstance(fname, Path) else fname
    data = pydicom.dcmread(fname)
    x = data.pixel_array

    if apply_rescale:
        # Sources:
        # * https://pydicom.github.io/pydicom/3.0/tutorials/pixel_data/introduction.html
        # * https://pydicom.github.io/pydicom/3.0/release_notes/v3.0.0.html
        # * https://pydicom.github.io/pydicom/2.4/reference/generated/pydicom.pixel_data_handlers.apply_rescale.html
        # NOTE: This function is deprecated in pydicom 3.0.0 in favor of
        # the new function pydicom.pixels.apply_rescale. It is currently
        # kept for compatibility with Python 3.9 which is only compatible
        # with versions of pydicom older than version 3.0.0.
        if not hasattr(pydicom, "pixel_data_handlers") or not hasattr(
            pydicom.pixel_data_handlers, "apply_rescale"
        ):
            raise ImportError(
                "pydicom version is unsupported. Please install a version of pydicom ≥ 2.0.0 and < 4.0.0"
            )
        x = pydicom.pixel_data_handlers.apply_rescale(x, data)

        # NOTE: apply_rescale returns float64 arrays. Most
        # applications do not need double precision so we cast it back to
        # float32 for improved memory efficiency.

    x = x.astype(dtype)

    return torch.from_numpy(x).unsqueeze(0) if as_tensor else x


def load_ismrmd(
    fname: str | Path,
    data_name: str = "kspace",
    data_slice: tuple | None = None,
    **kwargs,
) -> torch.Tensor:
    """Load complex MRI data from ISMRMD format.

    Uses `h5py` to load data specified by `data_name` key. The data is assumed to be stored in complex type.

    .. note::
        To speed up loading, slice/index the data before converting to tensor.

    :param str, pathlib.Path fname: file to load.
    :param str data_name: key of data in file, defaults to "kspace".
    :param tuple data_slice: slice or index to use before converting to tensor, such as `int`, `slice` or `tuple` of these.
    :return: data loaded in :class:`torch.Tensor` of shape `(2, ...)` containing real and imaginary parts,
        where `...` are dimensions of the raw data.
    """
    try:
        import h5py
    except ImportError:  # pragma: no cover
        raise ImportError(
            "The h5py package is not installed. Please install it with `pip install h5py`."
        )

    with h5py.File(fname, "r") as hf:
        data = hf[data_name]
        data = data[()] if data_slice is None else data[data_slice]
        data = MRIMixin.from_torch_complex(torch.from_numpy(data).unsqueeze(0)).squeeze(
            0
        )

    return data


def load_mat(fname: str, mat73: bool = False, **kwargs) -> dict[str, np.ndarray]:
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
        except ImportError:  # pragma: no cover
            raise ImportError("mat73 is required, install with 'pip install mat73'.")
        except TypeError:  # pragma: no cover
            warn(
                "Array is incompatible with MATLAB 7.3 loader. Using scipy loader instead..."
            )

    try:
        from scipy.io import loadmat as scipy_loadmat

        return scipy_loadmat(fname)
    except ImportError:  # pragma: no cover
        raise ImportError(
            "load_mat requires scipy, which is not installed. Install it with `pip install scipy`."
        )


def load_raster(
    fname: str,
    patch: bool | int | tuple[int, int] = False,
    patch_start: tuple[int, int] = (0, 0),
    transform: Callable | None = None,
    **kwargs,
) -> torch.Tensor | Iterator[torch.Tensor]:
    """
    Load a raster image and return patches as tensors using `rasterio`.

    This function allows you to stream patches from large rasters e.g. satellite imagery, SAR etc.
    and supports all file formats supported by `rasterio`.

    This function requires `rasterio`, and should not rely on external GDAL dependencies. Install it with `pip install rasterio`.

    :param str fname: Path to the raster file, such as `.geotiff`, `.tiff`, `.cos` etc., or buffer.
    :param bool, int, tuple[int, int], patch: Patch extraction mode. If ``False`` (default), return the entire image as a
        :class:`torch.Tensor` of shape `(C, H, W)` where C are bands.
        If ``True``, yield patches based on the raster's internal block windows
        (if no block windows are available, raises error; if any block has a dimension of 1 (strip layout), raise warning).
        If ``int`` or ``(int, int)``, yield patches of the manually specified size `h, w`.
    :param tuple[int, int] patch_start: h and w indices from which to start taking patches. Defaults to `0,0`.
    :param Callable, None transform: Optional transform applied to each patch.

    :return: Either (where C is the band dimension)
        * a full image :class:`torch.Tensor` of shape `(C, H, W)`, if ``patch=False``, or
        * an iterator of torch tensors over patches of shape `(C, h, w)`, if ``patch=True`` or a size is specified, where `h,w` is the patch size.

    |sep|

    :Examples:

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
    84
    >>> from torch.utils.data import DataLoader
    >>> dataloader = DataLoader(all_patches, batch_size=2) # You can use this for training
    >>>
    >>> x = load_raster(file, patch=128, patch_start=(200, 200)) # Patch via manual size, pick away from origin
    >>> next(x).shape
    torch.Size([1, 128, 128])
    """
    try:
        import rasterio
    except ImportError:  # pragma: no cover
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
        with rasterio.open(fname) as src:
            return _process_array(src.read())

    def _patch_generator() -> Iterator[torch.Tensor]:
        src = rasterio.open(fname)

        if patch is True:
            block_windows = list(src.block_windows(1))  # use band 1 as representative
            if not block_windows:
                raise RuntimeError("No block windows available for this raster.")
            try:
                for _, window in block_windows:
                    w, h = window.width, window.height
                    if w == 1 or h == 1:
                        warn(
                            f"Block window {window} returns patches of width or height 1. Consider manually setting a patch size."
                        )
                    yield _process_array(src.read(window=window))
                return
            finally:
                src.close()

        elif isinstance(patch, int):
            patch_h, patch_w = patch, patch
        elif isinstance(patch, tuple) and len(patch) == 2:
            patch_h, patch_w = patch
        else:
            raise ValueError(f"Invalid value for patch: {patch}")
        try:
            for y in range(patch_start[0], src.height, patch_h):
                for x in range(patch_start[1], src.width, patch_w):
                    window = rasterio.windows.Window(
                        x,
                        y,
                        min(patch_w, src.width - x),
                        min(patch_h, src.height - y),
                    )
                    yield _process_array(src.read(window=window))
            return
        finally:
            src.close()

    return _patch_generator()


def load_nifti(
    fname: str | Path,
    as_memmap: bool = False,
    dtype: np.dtype | str = np.float32,
    **kwargs,
) -> torch.Tensor | nib.arrayproxy.ArrayProxy:
    """Load volume from nifti file as torch tensor.

    We assume that the data contains a channel dimension. If not, unsqueeze the output to
    add a channel dimension `x = load_nifti(...).unsqueeze(0)`.

    .. warning::

        When loading zipped nifti files (e.g., .nii.gz), it is recommended to install indexed_gzip (`pip install indexed-gzip`) to speed up loading times.

    .. warning:

        Set the `dtype` correctly to load double or complex data.
        You can also inspect the `nibabel` image object headers (result of `nib.load`) prior to calling `get_fdata` or `dataobj`,
        to get the intended `dtype` and other metadata.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a proxy array, which does not eagerly load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param numpy.dtype, str dtype: data type to use when loading the nifti file. This is ignored if `as_memmap` is `True`.
    :return: :class:`torch.Tensor` containing nifti image. If `as_memmap` is `True`, returns a proxy array instead.
    """
    try:
        import nibabel as nib
    except ImportError:  # pragma: no cover
        raise ImportError(
            "load_nifti requires nibabel, which is not installed. Please install it with `pip install nibabel`."
        )
    im = nib.load(fname)
    if as_memmap:
        return im.dataobj
    else:
        return torch.from_numpy(im.get_fdata(dtype=dtype))


def load_blosc2(
    fname: str | Path,
    as_memmap: bool = False,
    dtype: np.dtype | str = np.float32,
    **kwargs,
) -> torch.Tensor | blosc2.ndarray.NDArray:
    """Load volume from blosc2 file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memory-mapped array (which does not load the entire array into memory). This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param numpy.dtype, str dtype: data type to use when loading the blosc2 file. This is ignored if `as_memmap` is `True`.
    :return: :class:`torch.Tensor` containing loaded numpy array. If `as_memmap` is `True`, returns a blosc2 array object instead.
    """
    try:
        import blosc2
    except ImportError:  # pragma: no cover
        raise ImportError(
            "load_blosc2 requires blosc2, which is not installed. Please install it with `pip install blosc2`."
        )
    arr = blosc2.open(fname)
    if as_memmap:
        return arr
    else:
        return torch.from_numpy(arr[:].astype(dtype))
