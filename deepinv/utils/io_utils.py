from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nibabel as nib
    import blosc2


def load_np(
    fname: str | Path, as_memmap: bool = False, dtype: np.typing.DtypeLike = np.float32
) -> torch.Tensor | np.memmap:
    """Load numpy array from file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memmap, which does not load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param dtype: data type to use when loading the numpy array. This is ignored if as_memmap is True.
    :return: :class:`torch.Tensor` containing loaded numpy array. If as_memmap is True, returns a numpy memmap object instead.
    """
    if as_memmap:
        return open_memmap(fname)
    else:
        return torch.from_numpy(np.load(fname, allow_pickle=False).astype(dtype))


def load_nifti(
    fname: str | Path,
    as_memmap: bool = False,
    dtype: np.typing.DTypeLike = np.float32,
) -> torch.Tensor | nib.arrayproxy.ArrayProxy:
    """Load volume from nifti file as torch tensor.

    .. warning::

        When loading zipped nifti files (e.g., .nii.gz), it is recommended to install indexed_gzip (`pip install indexed-gzip`) to speed up loading times.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a proxy array, which does not eagerly load the entire array into memory. This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param dtype: data type to use when loading the nifti file. This is ignored if as_memmap is True.
    :return: :class:`torch.Tensor` containing nifti image. If as_memmap is True, returns a proxy array instead.
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
    dtype: np.typing.DTypeLike = np.float32,
) -> torch.Tensor | blosc2.ndarray.NDArray:
    """Load volume from blosc2 file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memory-mapped array (which does not load the entire array into memory). This is useful when extracting patches from large arrays or to quickly infer dtype and shape.
    :param dtype: data type to use when loading the blosc2 file. This is ignored if as_memmap is True.
    :return: :class:`torch.Tensor` containing loaded numpy array. If as_memmap is True, returns a blosc2 array object instead.
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
