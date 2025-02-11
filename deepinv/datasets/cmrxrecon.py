from typing import Any, Callable, Optional, Union, List, Dict, Tuple
from pathlib import Path
import os
from natsort import natsorted
from tqdm import tqdm
from warnings import warn

from numpy import ndarray
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F

from deepinv.datasets.fastmri import FastMRISliceDataset
from deepinv.datasets.utils import loadmat
from deepinv.physics.mri import MRIMixin
from deepinv.physics.generator.mri import BaseMaskGenerator


class CMRxReconSliceDataset(FastMRISliceDataset, MRIMixin):
    """CMRxRecon dynamic MRI dataset.

    Return tuples `(x, y)` of target and kspace respectively.

    Optionally, apply mask to measurements to get undersampled measurements.
    Then the dataset returns tuples `(x, y, params)` where `params` is a dict `{'mask': mask}`.
    This can be directly used with :class:`deepinv.Trainer` to train with undersampled measurements.
    If masks are present in the data folders (in file format `cine_xax_mask.mat`) then these will be loaded.
    If not, unique masks will be generated using a `mask_generator`, for example :class:`deepinv.physics.generator.RandomMaskGenerator`.

    Directly compatible with deepinv physics, i.e. TODO

    TODO test that CMRxRecon original data AccFactor04 folders contain GT in and how it's loaded
    TODO check original data compatible with dataset
    TODO we provide demo file...

    TODO example

    :param Union[str, Path] root: _description_
    :param str data_dir: _description_, defaults to 'SingleCoil/Cine/TrainingSet/FullSample'
    :param bool load_metadata_from_cache: _description_, defaults to False
    :param bool save_metadata_to_cache: _description_, defaults to False
    :param Union[str, Path] metadata_cache_file: _description_, defaults to "dataset_cache.pkl"
    :param Optional[Callable] transform: _description_, defaults to None
    :param bool apply_mask: if `False`, return data `(x,y)`, if `True`, return `(x,y,{'mask':mask})` where mask is either
        loaded from `data_folder` or generated using `mask_generator`.
    :param Optional[BaseMaskGenerator] mask_generator: _description_, defaults to None
    """

    def __init__(
        self,
        root: Union[str, Path],
        data_dir: str = "SingleCoil/Cine/TrainingSet/FullSample",
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        apply_mask: bool = True,
        mask_dir: str = "SingleCoil/Cine/TrainingSet/AccFactor04",
        mask_generator: Optional[BaseMaskGenerator] = None,
        transform: Optional[Callable] = None,
    ):

        self.root = Path(root)
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_generator = mask_generator
        self.apply_mask = apply_mask
        self.load_metadata_from_cache = load_metadata_from_cache
        self.save_metadata_to_cache = save_metadata_to_cache
        self.metadata_cache_file = metadata_cache_file

        if not self.apply_mask and (
            self.mask_generator is not None or self.mask_dir is not None
        ):
            warn(
                "mask_generator or mask_dir specified but apply_mask is False. mask_generator or mask_dir will not be used."
            )
            self.mask_dir = self.mask_generator = None

        if (
            self.apply_mask
            and self.mask_generator is not None
            and self.mask_dir is not None
        ):
            raise ValueError(
                "Only one of mask_generator or mask_dir should be specified."
            )

        if not os.path.isdir(self.root / self.data_dir) or (
            self.mask_dir is not None and not os.path.isdir(self.root / self.mask_dir)
        ):
            raise ValueError(
                f"Data or mask folder does not exist. Please set root, data_dir and mask_dir properly."
            )

        all_fnames = natsorted(
            f
            for f in (self.root / self.data_dir).rglob("**/*.mat")
            if not str(f).endswith("_mask.mat")
        )

        with self.metadata_cache_manager(self.root, []) as sample_identifiers:
            if len(sample_identifiers) == 0:
                for fname in tqdm(all_fnames):
                    metadata = self._retrieve_metadata(fname)
                    for slice_ind in range(metadata["num_slices"]):
                        sample_identifiers.append(
                            self.SliceSampleFileIdentifier(fname, slice_ind, metadata)
                        )

            self.sample_identifiers = sample_identifiers

    def _loadmat(self, fname: Union[str, Path, os.PathLike]) -> ndarray:
        """Load matrix from MATLAB 7.3 file and parse headers."""
        return next(
            v for k, v in loadmat(fname, mat73=True).items() if not k.startswith("__")
        )

    def _retrieve_metadata(
        self, fname: Union[str, Path, os.PathLike]
    ) -> Dict[str, Any]:
        """Open file and retrieve metadata

        Metadata includes width, height, slices, coils (if multicoil) and timeframes.

        :param Union[str, Path, os.PathLike] fname: filename to open
        :return: metadata dict of key-value pairs.
        """
        shape = self._loadmat(fname).shape  # WH(N)DT
        return {
            "width": shape[0],  # W
            "height": shape[1],  # H
            "num_slices": shape[-2],  # D (depth)
            "timeframes": shape[-1],  # T
        } | (
            {
                "coils": shape[2],  # N (coils)
            }
            if len(shape) == 5
            else {}
        )

    def __getitem__(self, i: int) -> Tuple[Tensor]:
        fname, slice_ind, metadata = self.sample_identifiers[i]

        kspace = self._loadmat(fname)  # shape WH(N)DT
        kspace = kspace[..., slice_ind, :]  # shape WH(N)T

        if len(kspace.shape) == 5:
            kspace = kspace[:, :, 0]  # shape WHT

        kspace = torch.from_numpy(
            np.stack((kspace.real, kspace.imag), axis=0)
        )  # shape CWHT
        kspace = kspace.moveaxis(-1, 1)  # shape CTWH
        target = self.kspace_to_im(kspace.unsqueeze(0)).squeeze(0)  # shape CTWH

        # Apply target transform
        if self.transform is not None:
            target = self.transform(target)

        # Load mask
        if self.apply_mask:
            if self.mask_generator is None:
                try:
                    mask = self._loadmat(
                        str(fname)
                        .replace(str(Path(self.data_dir)), str(Path(self.mask_dir)))
                        .replace(".mat", "_mask.mat")
                    )
                    mask = self.check_mask(mask, three_d=True)[0]  # shape CTWH
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "Mask not found in mask_dir and mask_generator not specified. Choose mask_dir containing masks, or specify mask_generator."
                    )
            else:
                mask = self.mask_generator.step(
                    seed=str(fname) + str(slice_ind),
                    img_size=kspace.shape[-2:],
                    batch_size=0,
                )["mask"]
        else:
            mask = torch.ones_like(kspace)

        # Pad
        target, mask = self.pad(target, mask, (512, 256))

        # Normalise
        target = (target - target.mean()) / (target.std() + 1e-11)

        kspace = self.im_to_kspace(target.unsqueeze(0)).squeeze(0)

        if self.apply_mask:
            kspace = kspace * mask + 0.0

        # if self.noise_level is not None and self.noise_level > 0:
        #     kspace = GaussianNoise(sigma=self.noise_level, rng=self.generator)(kspace) * mask

        return (target, kspace, {"mask": mask.float()})

    def pad(
        self, img1: Tensor, img2: Tensor, shape: Tuple[int, int] = (512, 256)
    ) -> Tuple[Tensor, Tensor]:
        """Pad images to shape. Assume images have same shape.

        :param torch.Tensor img1: input image 1 of shape [..., W, H]
        :param torch.Tensor img2: input image 2 of shape [..., W, H]
        :param tuple shape: pad to shape, defaults to (512, 256)
        :return: padded images.
        """
        assert img1.shape[-2:] == img2.shape[-2:]
        w_pad, h_pad = (shape[0] - img1.shape[-2]), (shape[1] - img1.shape[-1])
        pad = (h_pad // 2, h_pad // 2, w_pad // 2, w_pad // 2)
        return (
            F.pad(img1, pad, mode="constant", value=0),
            F.pad(img2, pad, mode="constant", value=0),
        )
