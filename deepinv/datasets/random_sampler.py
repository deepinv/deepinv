from __future__ import annotations
import os
import random
import numpy as np
import torch
from deepinv.datasets.base import ImageDataset


class RandomPatchSampler(ImageDataset):
    def __init__(
        self,
        x_dir: str = None,
        y_dir: str = None,
        patch_size: int | tuple[int] = 32,
        format: str = ".npy",
        ch_axis: int = None,
    ):
        r"""
        Builds a dataset from folders of 3D images. Each epoch, a single patch is randomly sampled from each volume.

        Each image can have a different shape, but all images must have shape H, W, D. Other axis are not allowed (or will be squeezed).

        :param str x_dir: Path to folder of ground-truth images.
        :param str y_dir: Path to folder of measurements. Measurements must be images of same shape as ground-truth.
        :param int patch_size: Patch size to use, must be <= smallest shape in the dataset
        :param str format: Format to use. Other files will be ignored. Supported: .npy, .nii(.gz), .b2nd (blosc2)
        :param int ch_axis: Specifies which axis contains channels in the files. Currently, only 0 or -1 are supported. If None, will perform unsqueeze(0) to create singleton channel.
        """
        assert x_dir or y_dir, "Both x_dir and y_dir cannot be None."
        if ch_axis:
            assert (
                ch_axis == 0 or ch_axis == -1
            ), f"Only None, 0, or 1 are supported for ch_axis. Got {ch_axis} ({type(ch_axis)})"

        self.x_dir, self.y_dir = x_dir, y_dir
        self.patch_size, self.ch_ax = patch_size, ch_axis
        self._set_load(format)

        x_imgs, y_imgs = None, None

        if x_dir is not None:
            assert os.path.exists(x_dir), f"Measurement dir {x_dir} does not exist."
            x_imgs = [f for f in os.listdir(x_dir) if f.endswith(format)]
            assert (
                len(x_imgs) != 0
            ), f"Measurement dir is given but empty for file format {format}."

        if y_dir is not None:
            assert os.path.exists(y_dir), f"Ground-truth dir {y_dir} does not exist."
            y_imgs = [f for f in os.listdir(y_dir) if f.endswith(format)]
            assert (
                len(y_imgs) != 0
            ), f"Ground-truth dir is given but empty for file format {format}."

        self.imgs = (
            [f for f in x_imgs if f in y_imgs]
            if (x_imgs and y_imgs)
            else (x_imgs or y_imgs)
        )

        self._set_shapes()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        shape = self.shapes[idx]
        # We use random here: need to ensure deterministic behaviour based on seed --> seed worker function, see torch reproducibility page
        start_coords = [
            random.randint(p, s - p) if p is not None else p
            for p, s in zip(self.patch_size, shape)
        ]

        fname = self.imgs[idx]

        x = (
            self._fix_ch(
                self._load(
                    os.path.join(self.x_dir, fname),
                    start_coords=start_coords,
                    patch_size=self.patch_size,
                )
            )
            if self.x_dir
            else torch.nan
        )
        if self.y_dir is not None:
            y = self._fix_ch(
                self._load(
                    os.path.join(self.y_dir, fname),
                    start_coords=start_coords,
                    patch_size=self.patch_size,
                )
            )
            return (x, y)
        else:
            return x

    def _fix_ch(self, v: np.ndarray):
        if self.ch_ax is None:
            return v.unsqueeze(0)
        elif self.ch_ax == -1:
            nd = len(v.shape)
            return np.transpose(v, (nd - 1,) + tuple(range(nd - 1)))
        else:
            return v

    def _set_shapes(self):
        ndim = None
        self.shapes = []
        n_ch = None
        for im in self.imgs:
            shape = self._load(
                os.path.join(self.y_dir if self.y_dir else self.x_dir, im),
                as_memmap=True,
            ).shape
            if not ndim:
                ndim = len(shape)
                n_ch = None if self.ch_ax is None else shape[self.ch_ax]
                if isinstance(self.patch_size, int):
                    self.patch_size = [self.patch_size for i in range(ndim)]
                if len(self.patch_size) == ndim:
                    if self.ch_ax is not None:
                        self.patch_size[self.ch_ax] = (
                            None  # this is silent right now, but patching along ch makes no sense?
                        )
                elif len(self.patch_size) == ndim - 1:
                    self.patch_size.insert(self.ch_ax, None)
                self.patch_size = tuple(
                    self.patch_size
                )  # self.patch_size should not change from now.

            assert (
                len(shape) == ndim
            ), f"Dim mismatch. Dataset has {ndim} dims, but {im} has shape {shape}"
            assert all(
                s >= p if p is not None else True
                for s, p in zip(shape, self.patch_size)
            )
            if n_ch:
                assert (
                    shape[self.ch_ax] == n_ch
                ), f"Not all images have the same ch shape. Current shape: {shape}. Please check your data shapes + Dataset args."
            self.shapes.append(shape)
        self.shapes = tuple(self.shapes)  # avoid mutable members

    def _set_load(self, format: str):
        if format.endswith(".npy"):
            from deepinv.utils.io_utils import load_np

            self._load = load_np
        elif format.endswith(".nii") or format.endswith(".nii.gz"):
            from deepinv.utils.io_utils import load_nifti

            self._load = load_nifti
        elif format.endswith(".b2nd"):
            from deepinv.utils.io_utils import load_blosc2

            self._load = load_blosc2
        else:
            raise NotImplementedError(
                "No loader function for 3D volumes with extension {format}"
            )
