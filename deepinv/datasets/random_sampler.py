from __future__ import annotations
import os
import random
import torch
from deepinv.datasets.base import ImageDataset


class RandomPatchSampler(ImageDataset):
    r"""
    Dataset for nD images that samples one random patch per image.

    This dataset builds from one or two directories of nD images (`.npy`, `.nii(.gz)`, or `.b2nd`).
    On each epoch, it returns a randomly sampled patch of fixed size from each volume.

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
    - If ``patch_size`` is tuple, and ``patch_size[i] == 1``, this is equivalent to slicing across axis i (singleton at axis i will be squeezed). This can be used to e.g. extract 2D slices from a 3D volume
    - If tuple length is one less than the image ndim, the channel axis is auto-filled with ``None``.

    **Randomness & reproducibility:**
    - Patch coordinates are drawn with Python’s ``random`` module.
    - To ensure deterministic behavior across workers, set the DataLoader's
    ``worker_init_fn`` or ``generator`` according to the PyTorch reproducibility guidelines.

    **Notes**
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
    ):
        r"""
        :param str, optional x_dir: Path to folder of ground-truth images. Required if ``y_dir`` is not given.
        :param str, optional y_dir: Path to folder of measurement images. Required if ``x_dir`` is not given.
        :param int, tuple patch_size: Size of patches to extract. If int, applies the same size across all spatial dimensions.
        :param str file_format : File format to load. Other files are ignored.
        :param int ch_axis: Axis of the channel dimension. If None, a new singleton channel is added.
        """
        assert x_dir or y_dir, "Provide at least one of x_dir or y_dir."
        if ch_axis is not None:
            assert (
                ch_axis == 0 or ch_axis == -1
            ), f"Only None, 0, or -1 are supported for ch_axis. Got {ch_axis} ({type(ch_axis)})"
        if isinstance(patch_size, tuple) or isinstance(patch_size, list):
            for i, p in enumerate(patch_size):
                assert isinstance(
                    p, int
                ), f"patch_size arguments must be integers, got type {type(p)} at index {i}"
        self.x_dir, self.y_dir = x_dir, y_dir
        self.patch_size, self.ch_ax = patch_size, ch_axis
        self._set_load(file_format)

        x_imgs, y_imgs = None, None

        if x_dir is not None:
            assert os.path.exists(x_dir), f"Ground-truth dir {x_dir} does not exist."
            x_imgs = [f for f in os.listdir(x_dir) if f.endswith(file_format)]
            assert (
                len(x_imgs) != 0
            ), f"Ground-truth dir is given but empty for file format {file_format}."

        if y_dir is not None:
            assert os.path.exists(y_dir), f"Measurement dir {y_dir} does not exist."
            y_imgs = [f for f in os.listdir(y_dir) if f.endswith(file_format)]
            assert (
                len(y_imgs) != 0
            ), f"Measurement dir is given but empty for file format {file_format}."

        self.imgs = (
            sorted(set(x_imgs) & set(y_imgs))
            if (x_imgs and y_imgs)
            else (x_imgs or y_imgs)
        )

        assert len(self.imgs) > 0, "No (shared) images available."

        self._set_shapes()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        shape = self.shapes[idx]
        # We use random here: need to ensure deterministic behaviour based on seed --> seed worker function, see torch reproducibility page
        start_coords = [
            random.randint(0, s - p) if p is not None else p
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

    def _fix_ch(self, v: torch.Tensor):
        if self.ch_ax is None:
            v = v.unsqueeze(0)
        elif self.ch_ax == -1:
            nd = len(v.shape)
            v = v.permute(nd - 1, *range(nd - 1)).contiguous()
        return v.squeeze(dim=tuple(i for i, p in enumerate(self.patch_size) if p == 1))

    def _set_shapes(self):
        ndim = None
        self.shapes = []
        n_ch = None
        for im in self.imgs:
            if self.y_dir and self.x_dir:
                s_x = self._load(os.path.join(self.x_dir, im), as_memmap=True).shape
                s_y = self._load(os.path.join(self.y_dir, im), as_memmap=True).shape
                assert (
                    s_x == s_y
                ), f"Measurement and ground-truth image shapes must match, but mismatch for {im}"
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

    def _set_load(self, file_format: str):
        if file_format.endswith(".npy"):
            from deepinv.utils.io_utils import load_np

            self._load = load_np
        elif file_format.endswith(".nii") or file_format.endswith(".nii.gz"):
            from deepinv.utils.io_utils import load_nifti

            self._load = load_nifti
        elif file_format.endswith(".b2nd"):
            from deepinv.utils.io_utils import load_blosc2

            self._load = load_blosc2
        else:
            raise NotImplementedError(
                f"No loader function for 3D volumes with extension {file_format}"
            )
