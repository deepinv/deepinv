from typing import Any, Callable, Optional, Union
from pathlib import Path
import os

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    natsorted = ImportError(
        "natsort is not available. In order to use CMRxReconSliceDataset, please install the natsort package with `pip install natsort`."
    )  # pragma: no cover

from tqdm import tqdm
from warnings import warn

from numpy import ndarray
import numpy as np
from torch import Tensor
import torch
import torch.nn.functional as F

from deepinv.datasets.fastmri import FastMRISliceDataset, MRISliceTransform
from deepinv.datasets.utils import loadmat
from deepinv.physics.mri import MRIMixin
from deepinv.physics.generator.mri import BaseMaskGenerator
from deepinv.physics.noise import NoiseModel


class CMRxReconSliceDataset(FastMRISliceDataset, MRIMixin):
    """CMRxRecon dynamic MRI dataset.

    Wrapper for dynamic 2D+t MRI dataset from the `CMRxRecon 2023 challenge <https://cmrxrecon.github.io/>`_.

    The dataset returns sequences of long axis (`lax`) views and short axis (`sax`) slices along with 2D+t acceleration masks.

    Return tuples `(x, y)` of target (ground truth) and kspace (measurements).

    Optionally apply mask to measurements to get undersampled measurements.
    Then the dataset returns tuples `(x, y, params)` where `params` is a dict `{'mask': mask}`.
    This can be directly used with :class:`deepinv.Trainer` to train with undersampled measurements.
    If masks are present in the data folders (in file format `cine_xax_mask.mat`) then these will be loaded.
    If not, unique masks will be generated using a `mask_generator`, for example :class:`deepinv.physics.generator.RandomMaskGenerator`.

    While the usual workflow in deepinv is for the dataset to return only ground truth ``x`` and the user
    generates a measurement dataset using :meth:`deepinv.datasets.generate_dataset`, here we compute the
    measurements inside the dataset (and return a triplet ``x, y, params`` where ``params`` contains the mask)
    because of the variable size of the data before padding, in line with the original CMRxRecon code.

    .. note::

        The data returned is directly compatible with :class:`deepinv.physics.DynamicMRI`.
        See :ref:`sphx_glr_auto_examples_basics_demo_tour_mri.py` for example using this dataset.

    We provide one single downloadable demo sample, see example below on how to use this.
    Otherwise, download the full dataset from the `challenge website <https://cmrxrecon.github.io/>`_.

    **Raw data file structure:** ::

        root_dir --- data_dir --- P001 --- cine_lax.mat
                  |            |        |
                  |            |        -- cine_sax.mat
                  |            -- PXXX
                  -- mask_dir --- P001 --- cine_lax_mask.mat
                               |        |
                               |        -- cine_sax_mask.mat
                               -- PXXX

    |sep|

    Example:

    >>> from deepinv.datasets import CMRxReconSliceDataset, download_archive
    >>> from deepinv.utils import get_image_url, get_data_home
    >>> from torch.utils.data import DataLoader
    >>> download_archive(
    ...     get_image_url("CMRxRecon.zip"),
    ...     get_data_home() / "CMRxRecon.zip",
    ...     extract=True,
    ... )
    >>> dataset = CMRxReconSliceDataset(get_data_home() / "CMRxRecon")
    >>> x, y, params = next(iter(DataLoader(dataset)))
    >>> x.shape # (B, C, T, H, W)
    torch.Size([1, 2, 12, 512, 256])
    >>> y.shape # (B, C, T, H, W)
    torch.Size([1, 2, 12, 512, 256])
    >>> 1 / params["mask"].mean()  # Approx 4x acceleration
    tensor(4.2402)

    :param str, pathlib.Path root: path for dataset root folder.
    :param str, pathlib.Path data_dir: directory containing target (ground truth) data, defaults to 'SingleCoil/Cine/TrainingSet/FullSample' which is default CMRxRecon folder structure
    :param bool load_metadata_from_cache: _description_, defaults to False
    :param bool save_metadata_to_cache: _description_, defaults to False
    :param str, pathlib.Path metadata_cache_file: _description_, defaults to "dataset_cache.pkl"
    :param bool apply_mask: if ``True``, mask is applied to subsample the kspace using a mask either
        loaded from `data_folder` or generated using `mask_generator`. If ``False``, the mask of ones is used.
    :param str, pathlib.Path mask_dir: dataset folder containing predefined acceleration masks. Defaults to the 4x acc. mask folder
        according to the CMRxRecon folder structure. To use masks, ``apply_mask`` must be ``True``.
    :param deepinv.physics.generator.BaseMaskGenerator mask_generator: optional mask generator to randomly generate acceleration masks
        to apply to unpadded kspace. If specified, ``mask_dir`` must be ``None`` and ``apply_mask`` must be ``True``.
    :param Callable transform: optional transform to apply to the target image sequences before padding or physics is applied.
    :param tuple pad_size: tuple of 2 ints (W, H) for all images to be padded to, if ``None``, no padding.
    :param deepinv.physics.NoiseModel noise_model: optional noise model to apply to unpadded kspace.
    """

    def __init__(
        self,
        root: Union[str, Path],
        data_dir: Union[str, Path] = "SingleCoil/Cine/TrainingSet/FullSample",
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        apply_mask: bool = True,
        mask_dir: Union[str, Path] = "SingleCoil/Cine/TrainingSet/AccFactor04",
        mask_generator: Optional[BaseMaskGenerator] = None,
        transform: Optional[Callable] = None,
        pad_size: tuple[int, int] = (512, 256),
        noise_model: NoiseModel = None,
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
        self.pad_size = pad_size
        self.noise_model = noise_model

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

        if isinstance(natsorted, ImportError):
            raise natsorted

        all_fnames = natsorted(
            f
            for f in (self.root / self.data_dir).rglob("**/*.mat")
            if not str(f).endswith("_mask.mat")
        )

        with self.metadata_cache_manager(self.root, []) as samples:
            if len(samples) == 0:
                for fname in tqdm(all_fnames):
                    metadata = self._retrieve_metadata(fname)
                    for slice_ind in range(metadata["num_slices"]):
                        samples.append(self.SliceSampleID(fname, slice_ind, metadata))

            self.samples = samples

    def _loadmat(self, fname: Union[str, Path, os.PathLike]) -> ndarray:
        """Load matrix from MATLAB 7.3 file and parse headers."""
        return next(
            v for k, v in loadmat(fname, mat73=True).items() if not k.startswith("__")
        )

    def _retrieve_metadata(
        self, fname: Union[str, Path, os.PathLike]
    ) -> dict[str, Any]:
        """Open file and retrieve metadata

        Metadata includes width, height, slices, coils (if multicoil) and timeframes.

        :param str, pathlib.Path, os.PathLike fname: filename to open
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

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Get ith data sampe.

        :param int i: dataset index to get
        :return: tuple of ground truth ``x``, measurement ``y`` and params dict containing mask ``{'mask': mask}``
        """
        fname, slice_ind, metadata = self.samples[i]

        # Load kspace data, take slice, remove coil dim,
        # create complex dim, move time dim
        kspace = self._loadmat(fname)  # shape WH(N)DT
        kspace = kspace[..., slice_ind, :]  # shape WH(N)T

        if len(kspace.shape) == 5:
            kspace = kspace[:, :, 0]  # shape WHT

        kspace = torch.from_numpy(np.stack((kspace.real, kspace.imag), axis=0))
        kspace = kspace.moveaxis(-1, 1)  # shape CTWH
        target = None

        # TODO The following is akin to :class:`deepinv.datasets.fastmri.MRISliceTransform` and will be moved
        # to a separate CMRxReconTransform in future.

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
                mask = MRISliceTransform(
                    mask_generator=self.mask_generator
                ).generate_mask(kspace, str(fname) + str(slice_ind))
        else:
            mask = torch.ones_like(kspace)

        # Construct ground truth
        target = self.kspace_to_im(kspace.unsqueeze(0)).squeeze(0)  # shape CTWH
        assert target.shape[-2:] == mask.shape[-2:]

        # Apply target transform
        if self.transform is not None:
            target = self.transform(target)

        # Pad
        if self.pad_size is not None:
            w, h = (self.pad_size[0] - target.shape[-2]), (
                self.pad_size[1] - target.shape[-1]
            )
            target = F.pad(target, (h // 2, h // 2, w // 2, w // 2))
            mask = F.pad(mask, (h // 2, h // 2, w // 2, w // 2))

        # Normalise
        target = (target - target.mean()) / (target.std() + 1e-11)

        kspace = self.im_to_kspace(target.unsqueeze(0)).squeeze(0)

        if self.noise_model is not None:
            kspace = self.noise_model(kspace) * mask

        if self.apply_mask:
            kspace = kspace * mask + 0.0
            return target, kspace.float(), {"mask": mask.float()}
        else:
            return target, kspace.float()
