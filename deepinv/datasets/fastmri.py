r"""Pytorch Dataset for fastMRI.

Code modified from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import random
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union, Tuple
import pickle
import os
import h5py
import torch
import numpy as np


class FastMRISliceDataset(torch.utils.data.Dataset):
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to MR image slices.

    | 1) The fastMRI dataset includes two types of MRI scans: knee MRIs and
    | the brain (neuro) MRIs, and containing training, validation, and masked test sets.
    | 2) MRIs are volumes (3D) made of slices (2D).
    | 3) This class in particular considers one data sample as one slice of a MRI scan,
    | thus slices of the same MRI scan are considered independently in the dataset.


    **Raw data file structure:** ::

        self.root --- file1000005.h5
                   |
                   -- xxxxxxxxxxx.h5

    | 0) To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/`
    | 1) Each MRI scan is stored in a HDF5 file and can be read with the h5py package.
    | Each file contains the k-space data, ground truth and some meta data related to the scan.
    | 2) MRI scans can either be single-coil or multi-coil with each coil in
    | a multi-coil MRI scan focusses on a different region of the image.
    | 3) In multi-coil MRIs, k-space data has the following shape:
    | (number of slices, number of coils, height, width)
    | 4) For single-coil MRIs, k-space data has the following shape:
    | (number of slices, height, width)

    :param Union[str, Path] root: Path to the dataset.
    :param bool test: Whether the split is the "test" set.
    :param str challenge: "singlecoil" or "multicoil" depending on the type of mri scan.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, Path] metadata_cache_file: A file used to cache dataset
        information for faster load times.
    :param callable, optional sample_filter: A callable object that takes a
        :meth:`SliceSampleFileIdentifier` as input and returns a boolean indicating
        whether the sample should be included in the dataset.
    :param float, optional sample_rate: A float between 0 and 1. This controls what
        fraction of all slices should be loaded. Defaults to 1.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param float, optional volume_sample_rate: A float between 0 and 1. This controls
        what fraction of the volumes should be loaded. Defaults to 1 if no value is given.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param callable, optional transform_kspace: A function/transform that takes in the
        kspace and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``
    :param callable, optional transform_target: A function/transform that takes in the
        target and returns a transformed version. E.g, ``torchvision.transforms.RandomCrop``

    |sep|

    :Examples:

        Instanciate dataset without transform ::

            from deepinv.datasets import FastMRISliceDataset
            root = "/path/to/dataset/fastMRI/knee_singlecoil/train"
            dataset = FastMRISliceDataset(root=root, test=False, challenge="singlecoil")
            target, kspace = dataset[0]
            print(target.shape)
            print(kspace.shape)

        Instanciate dataset with transform ::

            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop((256, 256), pad_if_needed=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]) # Define the transform pipeline
            root = "/path/to/dataset/fastMRI/knee_singlecoil/train"
            dataset = FastMRISliceDataset(root=root, test=False, challenge="multicoil", transform_kspace=transform, transform_target=transform)
            target, kspace = dataset[0]
            print(target.shape)
            print(kspace.shape)

    """

    class SliceSampleFileIdentifier(NamedTuple):
        """Data structure for identifying specific slices within MRI data files."""

        fname: Path
        slice_ind: int

    def __init__(
        self,
        root: Union[str, Path],
        test: bool,
        challenge: str,
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        sample_filter: Callable = lambda raw_sample: True,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        transform_kspace: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.test = test
        self.challenge = challenge
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.transforms = {
            "transform_kspace": transform_kspace,
            "transform_target": transform_target,
        }

        # check that root is a folder
        if not os.path.isdir(root):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{root}`."
            )
        # check that root folder contains only hdf5 files
        if not all([file.endswith(".h5") for file in os.listdir(root)]):
            raise ValueError(
                f"The `root` folder doesn't contain only hdf5 files. Please set `root` properly. Current value `{root}`."
            )
        # ensure that challenge is either singlecoil or multicoil
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('`challenge` should be either "singlecoil" or "multicoil"')
        # ensure that sample_rate and volume_sample_rate are not used simultaneously
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "Either set `sample_rate` (sample by slices) or `volume_sample_rate` (sample by volumes) but not both"
            )

        ### LOAD DATA SAMPLE IDENTIFIERS -----------------------------------------------

        # should contain all the information to load a slice from the storage
        self.sample_identifiers = []
        if load_metadata_from_cache:  # from a cache file
            metadata_cache_file = Path(metadata_cache_file)
            if not metadata_cache_file.exists():
                raise ValueError(
                    "`metadata_cache_file` doesn't exist. Please either deactivate"
                    + "`load_dataset_from_cache` OR set `metadata_cache_file` properly."
                )
            with open(metadata_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
                if dataset_cache.get(root) is None:
                    raise ValueError(
                        "`metadata_cache_file` doesn't contain the metadata. Please"
                        + "either deactivate `load_dataset_from_cache` OR set `metadata_cache_file` properly."
                    )
                print(f"Using dataset cache from {metadata_cache_file}.")
                self.sample_identifiers = dataset_cache[root]
        else:
            files = sorted(list(Path(root).iterdir()))
            for fname in files:
                with h5py.File(fname, "r") as hf:
                    num_slices = hf["kspace"].shape[0]

                    # add each slice to the dataset after filtering
                    for slice_ind in range(num_slices):
                        slice_id = self.SliceSampleFileIdentifier(fname, slice_ind)
                        if sample_filter(slice_id):
                            self.sample_identifiers.append(slice_id)

            # save dataset metadata
            if save_metadata_to_cache:
                dataset_cache = {}
                dataset_cache[root] = self.sample_identifiers
                print(f"Saving dataset cache to {metadata_cache_file}.")
                with open(metadata_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)

        ### RANDOM SUBSAMPLING (1 sample = 1 slice from a MRI scan) --------------------

        # set default sampling mode to get the full dataset
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        if sample_rate < 1.0:  # sample by slice / randomly keep a portion of mri slices
            random.shuffle(self.sample_identifiers)
            num_samples = round(len(self.sample_identifiers) * sample_rate)
            self.sample_identifiers = self.sample_identifiers[:num_samples]
        elif (
            volume_sample_rate < 1.0
        ):  # sample by volume / randomly keep a portion of mri scans
            vol_names = list(set([f[0].stem for f in self.sample_identifiers]))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.sample_identifiers = [
                sample_id
                for sample_id in self.sample_identifiers
                if sample_id[0].stem in sampled_vols
            ]

    def __len__(self) -> int:
        return len(self.sample_identifiers)

    def __getitem__(self, idx: int, mask: Optional[Callable] = None) -> Tuple[Any, Any]:
        r"""Returns the idx-th sample from the dataset, both kspace and target.

        The target data is compatible with the physics MRI operator and is a complex tensor of shape (2, H, W).
        """
        fname, dataslice = self.sample_identifiers[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            if not self.test:
                target = hf[self.recons_key][dataslice]

        if not self.test and self.transforms["transform_target"] is not None:
            # by default, shape is (1, H, W), we want to get rid of the first dimension when moving to complex type
            target = self.transforms["transform_target"](target)[0]
            target = target + 0 * 1j
            target = torch.view_as_real(target)
            target = torch.moveaxis(target, -1, 0)  # shape (2, H, W)

        if self.transforms["transform_kspace"] is not None:
            if self.challenge == "multicoil":
                # (number of coils, height, width) -> (height, width, number of coils)
                kspace = np.moveaxis(kspace, 0, -1)
            kspace = self.transforms["transform_kspace"](kspace)

        if self.test:
            return kspace
        return target, kspace
