from __future__ import annotations
from typing import Callable, Any
import os
from pathlib import Path

import torch
import numpy as np
from deepinv.datasets.base import ImageDataset


class RandomPatchSampler(ImageDataset):
    r"""
    Dataset for nD images that samples one random patch per image.

    This dataset builds from one or two directories of nD images (must be of format ``.npy``, ``.nii(.gz)``, or ``.b2nd``, if ``loader`` is not specified).
    On each epoch, it returns a randomly sampled patch of fixed size from each volume.

    .. warning::

        This loader uses torch's random functionality. To ensure reproducibility, set the DataLoader's ``generator`` with a fixed seed.

    **Supported use cases:**

    - Single-directory: provide only the ground-truth folder ``x_dir`` or measurement folder ``y_dir`` (returns patches from that directory).

    - Paired-directory: provide both ``x_dir`` and ``y_dir`` (returns matched patches from both).

    **Channel handling:**

    - If ``ch_axis=None``: a singleton channel dimension is added at axis 0.

    - If ``ch_axis=0``: images are assumed channel-first.

    - If ``ch_axis=-1``: images are assumed channel-last and transposed to channel-first.

    - Patches are never extracted along the channel axis (patch size for that axis is ignored).

    **Patch size handling:**

    - Accepts either an integer (applied to all spatial dims) or a tuple.

    - If ``patch_size`` is tuple, and ``patch_size[i] == 1``, this is equivalent to slicing across axis i (singleton at axis i will be squeezed). This can be used to e.g. extract 2D slices from a 3D volume.

    - If tuple length is one less than the image ndim, the channel axis is auto-filled with ``None``.

    **Randomness & reproducibility:**

    - Patch coordinates are drawn with Python’s ``random`` module.

    - To ensure deterministic behavior across workers, set the DataLoader's
      ``worker_init_fn`` or ``generator`` according to the PyTorch reproducibility guidelines.

    **Notes:**

    - All images must have the same dimensionality.

    - When both directories are provided, only files present in both are used.

    - Shapes of each file are checked for consistency (spatial not smaller than ``patch_size`` + channels remain consistent across files).
    """

    def __init__(
        self,
        x_dir: str = None,
        y_dir: str = None,
        patch_size: int | tuple[int, ...] = 32,
        file_format: str = ".npy",
        ch_axis: int = None,
        dtype: torch.dtype = torch.float32,
        loader: Callable[[str | Path, bool], Any] = None,
    ):
        r"""
        :param str, optional x_dir: Path to folder of ground-truth images. Required if ``y_dir`` is not given.
        :param str, optional y_dir: Path to folder of measurement images. Required if ``x_dir`` is not given.
        :param int, tuple patch_size: Size of patches to extract. If int, applies the same size across all spatial dimensions.
        :param str file_format : File format to load. Other files are ignored.
        :param int ch_axis: Axis of the channel dimension. If None, a new singleton channel is added.
        :param torch.dtype dtype: Data type to use when loading the images.
        :param Callable loader: Custom loader function. Must accept path and the keyword ``as_memmap``, which will always be set to True. Must return an object that has shape attribute and returns a ``np.ndarray`` when sliced. If None, an internal loader is chosen based on ``file_format``.
        """
        if not (x_dir or y_dir):  # pragma: no cover
            raise RuntimeError("Provide at least one of x_dir or y_dir.")
        if ch_axis is not None:
            if not (ch_axis == 0 or ch_axis == -1):  # pragma: no cover
                raise ValueError(
                    f"Only None, 0, or -1 are supported for ch_axis. Got {ch_axis} ({type(ch_axis)})"
                )
        if isinstance(patch_size, tuple) or isinstance(patch_size, list):
            for i, p in enumerate(patch_size):
                if not isinstance(p, int):  # pragma: no cover
                    raise TypeError(
                        f"patch_size must be int or tuple of ints, got {type(p)} at index {i}"
                    )
        self.x_dir, self.y_dir = x_dir, y_dir
        self.patch_size, self.ch_ax = patch_size, ch_axis
        self.dtype = dtype
        self._load = self._get_loader(file_format) if loader is None else loader

        imgs = [None, None]  # x_imgs, y_imgs

        for i, d in enumerate([x_dir, y_dir]):
            if d is not None and not os.path.exists(d):  # pragma: no cover
                raise RuntimeError(f"Directory {d} does not exist.")
            if d is not None:
                imgs[i] = [f for f in os.listdir(d) if f.endswith(file_format)]
                if len(imgs[i]) == 0:  # pragma: no cover
                    raise RuntimeError(
                        f"Directory {d} is given but empty for file format {file_format}."
                    )

        self.imgs = sorted(
            list(
                (set(imgs[0]) & set(imgs[1]))
                if (imgs[0] and imgs[1])
                else (imgs[0] or imgs[1])
            )
        )

        if len(self.imgs) == 0:  # pragma: no cover
            raise RuntimeError("No (shared) images available.")

        self.shapes = self._get_shapes()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:

        shape = self.shapes[idx]

        start_coords = [
            torch.randint(0, s - p, (1,)).item() if p is not None else p
            for p, s in zip(self.patch_size, shape)
        ]

        fname = self.imgs[idx]

        x = (
            self._fix_ch(
                self.load(
                    os.path.join(self.x_dir, fname),
                    start_coords=start_coords,
                )
            )
            if self.x_dir
            else torch.nan
        )
        if self.y_dir is not None:
            y = self._fix_ch(
                self.load(
                    os.path.join(self.y_dir, fname),
                    start_coords=start_coords,
                )
            )
            return (x, y)
        else:
            return x

    def _fix_ch(self, v: torch.Tensor) -> torch.Tensor:
        if self.ch_ax is None:
            v = v.unsqueeze(0)
        elif self.ch_ax == -1:
            nd = len(v.shape)
            v = v.permute(nd - 1, *range(nd - 1)).contiguous()
        return v.squeeze(dim=tuple(i for i, p in enumerate(self.patch_size) if p == 1))

    def _get_shapes(self):
        ndim = None
        shapes = []
        n_ch = None
        for im in self.imgs:
            if self.y_dir and self.x_dir:
                s_x = self._load(os.path.join(self.x_dir, im), as_memmap=True).shape
                s_y = self._load(os.path.join(self.y_dir, im), as_memmap=True).shape
                if not s_x == s_y:  # pragma: no cover
                    raise RuntimeError(
                        f"Measurement and ground-truth image shapes must match, but mismatch for {im}"
                    )
                shape = s_x
            else:
                shape = self._load(
                    os.path.join(self.y_dir if self.y_dir else self.x_dir, im),
                    as_memmap=True,
                ).shape
            if not ndim:
                ndim = len(shape)
                n_ch = None if self.ch_ax is None else shape[self.ch_ax]
                if isinstance(self.patch_size, int):
                    self.patch_size = [self.patch_size for i in range(ndim)]
                self.patch_size = list(self.patch_size)  # ensure mutable
                if len(self.patch_size) == ndim:
                    if self.ch_ax is not None:
                        self.patch_size[self.ch_ax] = (
                            None  # this is silent right now, but patching along ch makes no sense?
                        )
                elif len(self.patch_size) == ndim - 1:
                    if self.ch_ax == 0:
                        self.patch_size.insert(self.ch_ax, None)
                    elif self.ch_ax == -1:
                        self.patch_size.append(None)
                self.patch_size = tuple(
                    self.patch_size
                )  # self.patch_size should not change from now.

            if not len(shape) == ndim:  # pragma: no cover
                raise RuntimeError(
                    f"Dim mismatch. Dataset has {ndim} dims, but {im} has shape {shape}"
                )
            if not all(
                s >= p if p is not None else True
                for s, p in zip(shape, self.patch_size)
            ):  # pragma: no cover
                raise RuntimeError(
                    f"Patch size {self.patch_size} is too large for image {im} with shape {shape}"
                )
            if n_ch and shape[self.ch_ax] != n_ch:  # pragma: no cover
                raise RuntimeError(
                    f"Not all images have the same number of channels. Current shape: {shape} for image {im}. Please check your data shapes + Dataset args."
                )
            shapes.append(shape)
        return tuple(shapes)

    def _get_loader(self, file_format: str):
        if file_format.endswith(".npy"):
            from deepinv.utils.io import load_np

            return load_np
        elif file_format.endswith(".nii") or file_format.endswith(".nii.gz"):
            from deepinv.utils.io import load_nifti

            return load_nifti
        elif file_format.endswith(".b2nd"):
            from deepinv.utils.io import load_blosc2

            return load_blosc2
        else:  # pragma: no cover
            raise NotImplementedError(
                f"No loader function for images with extension {file_format}"
            )

    def load(self, f: str | Path, start_coords: tuple) -> torch.Tensor:
        arr = self._load(f, as_memmap=True)
        slices = tuple(
            slice(start, start + size) if size is not None else slice(None)
            for start, size in zip(start_coords, self.patch_size)
        )
        arr = arr[slices]
        if isinstance(arr, torch.Tensor):
            return arr.to(self.dtype)
        elif isinstance(arr, np.ndarray):
            return torch.from_numpy(arr).to(self.dtype)
        else:  # pragma: no cover
            raise RuntimeError(
                f"Object returned by loader must be a np.ndarray or torch.tensor after slicing, got {type(arr)}"
            )
