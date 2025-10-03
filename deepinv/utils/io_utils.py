from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

import torch


def load_np(
    fname: str | Path, as_memmap=False, start_coords=[], patch_size=[]
) -> torch.Tensor:
    """Load numpy array from file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memmap and return
    :param tuple | list, start_coords: starting indices when patch is to be extracted
    :param tuple | list | int, patch_size: patch size if start_coords is given. If sequence, length must match start_coords
    :return: :class:`torch.Tensor` containing loaded numpy array.
    """
    if len(start_coords) != 0:
        arr = open_memmap(fname)
        assert len(start_coords) == len(
            arr.shape
        )  # I am assuming here that the images are single channel / have no channel on disk. To fix so start_coords can be different length for other cases
        if isinstance(patch_size, int):
            patch_size = [patch_size for i in range(len(start_coords))]
        assert len(start_coords) == len(patch_size)

        # We should check that we're not out of bound here?
        return torch.from_numpy(
            arr[tuple(slice(s, s + p) for s, p in zip(start_coords, patch_size))]
        )
    elif as_memmap:
        return open_memmap(fname)
    else:
        return torch.from_numpy(np.load(fname, allow_pickle=False))


def load_nifti(
    fname: str | Path,
    as_memmap: bool = False,
    start_coords: tuple | list = [],
    patch_size: tuple | list = [],
    dtype: np.typing.DTypeLike = np.float32,
) -> torch.Tensor:
    """Load numpy array from file as torch tensor.

    :param str, pathlib.Path fname: file to load.
    :param bool, as_memmap: open this file as a memmap and return
    :param tuple | list, start_coords: starting indices when patch is to be extracted
    :param tuple | list | int, patch_size: patch size if start_coords is given. If sequence, length must match start_coords
    :return: :class:`torch.Tensor` containing loaded numpy array.
    """
    try:
        import nibabel as nib
    except ImportError:  # pragma: no cover
        raise ImportError(
            "load_nifti requires nibabel, which is not installed. Please install it with `pip install nibabel`."
        )

    if len(start_coords) != 0:
        arr = nib.load(fname).dataobj
        assert len(start_coords) == len(
            arr.shape
        )  # I am assuming here that the images are single channel / have no channel on disk. To fix so start_coords can be different length for other cases
        if isinstance(patch_size, int):
            patch_size = [patch_size for i in range(len(start_coords))]

        assert len(start_coords) == len(patch_size)

        return torch.from_numpy(
            arr[
                tuple(slice(s, s + p) for s, p in zip(start_coords, patch_size))
            ].astype(dtype)
        )
    elif as_memmap:
        return nib.load(fname).dataobj  # dtype argument is not used in this case..
    else:
        return (
            torch.from_numpy(nib.load(fname).get_fdata(dtype=dtype))
            .float()
            .unsqueeze(0)
        )
