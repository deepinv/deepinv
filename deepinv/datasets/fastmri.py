r"""Pytorch Dataset for fastMRI.

Code modified from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import torchvision
import random  # beware
from pathlib import Path
from typing import (
    Callable,
    NamedTuple,
    Optional,
    Union,
)
import os
import h5py
import torch


class SliceSampleFileIdentifier(NamedTuple):
    fname: Path
    slice_ind: int


class FastMRISliceDataset(torch.utils.data.Dataset):
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to MR image slices.

    Expected files structure:
    self.root --- file1000005.h5
               |
               -- xxxxxxxxxxx.h5
               |
               -- ???

    The fastMRI dataset is distributed as a set of HDF5 files and can be
    read with the h5py package. Each file corresponds to one MRI scan and
    contains the k-space data, ground truth and some meta data related to
    the scan.

    MRI scans can either be single-coil or multi-coil with each coil in
    a multi-coil MRI scan focusses on a different region of the image.

    In multi-coil MRIs, k-space has the following shape:
    (number of slices, number of coils, height, width)

    For single-coil MRIs, k-space has the following shape:
    (number of slices, height, width)

    MRIs are acquired as 3D volumes, the first dimension is the number of 2D slices.

    This dataset considers one sample as one slice of a MRI scan,
    thus slices of the same MRI scan are considered independently in the dataset.

    :param Union[str, Path] root: Path to the dataset.
    :param str challenge: "singlecoil" or "multicoil" depending on the type of mri scan.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, Path] metadata_cache_file: A file in which to cache dataset
        information for faster load times.
    :param callable, optional sample_filter: A callable object that takes an raw_sample
        metadata as input and returns a boolean indicating whether the
        raw_sample should be included in the dataset.
    :param float, optional sample_rate: A float between 0 and 1. This controls what
        fraction of the slices should be loaded. Defaults to 1.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param float, optional volume_sample_rate: A float between 0 and 1. This controls
        what fraction of the volumes should be loaded. Defaults to 1 if no value is given.
        When creating a sampled dataset either set sample_rate (sample by slices)
        or volume_sample_rate (sample by volumes) but not both.
    :param callable, optional transform: A function that pre-processes the raw
        data into appropriate form. The transform function should take
        'kspace', 'target', 'attributes', 'filename', and 'slice' as
        inputs. 'target' may be null for test data.
    """

    def __init__(
        self,
        root: Union[str, Path],
        challenge: str,
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        sample_filter: Callable = lambda raw_sample: True,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        transform_kspace: Optional[Callable] = torchvision.transforms.ToTensor(),
        transform_target: Optional[Callable] = torchvision.transforms.ToTensor(),
    ):
        # check that root is a folder
        if not os.path.isdir(root):
            raise ValueError(
                f"""
            The `root` folder doesn't exist.
            Please set `root` properly.
            Current value `{root}`."""
            )
        # check that root contains hdf5 files
        if not any([file.endswith(".h5") for file in os.listdir(root)]):
            raise ValueError(
                f"""
            The `root` folder doesn't contain any hdf5 file.
            Please set `root` properly.
            Current value "{root}"."""
            )
        # ensure that challenge is either singlecoil or multicoil
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(
                f'''
            `challenge` should be either "singlecoil" or "multicoil"'''
            )
        # ensure that sample_rate and volume_sample_rate are
        # not used simultaneously
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                f"""
            Either set `sample_rate` (sample by slices) or 
            `volume_sample_rate` (sample by volumes) but not both"""
            )
        self.transforms = {'transform_kspace': transform_kspace,
                            'transform_target': transform_target
                          }
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        ### load dataset metadata

        # should contain all the information to load a slice from the storage
        self.sample_identifiers = []
        if load_metadata_from_cache:  # from a cache file
            metadata_cache_file = Path(metadata_cache_file)
            if not metadata_cache_file.exists():
                raise ValueError(
                    f"""
                `metadata_cache_file` doesn't exist.
                Please either deactivate `load_dataset_from_cache`
                or set `metadata_cache_file` properly."""
                )
            with open(metadata_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
                if dataset_cache.get(root) is None:
                    raise ValueError(
                        f"""
                    `metadata_cache_file` doesn't contain the metadata.
                    Please either deactivate `load_dataset_from_cache`
                    or set `metadata_cache_file` properly."""
                    )
                logging.info(f"Using dataset cache from {metadata_cache_file}.")
                self.raw_samples = dataset_cache[root]
        else:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                with h5py.File(fname, "r") as hf:
                    num_slices = hf["kspace"].shape[0]

                    # add each slice to the dataset after filtering
                    for slice_ind in range(num_slices):
                        slice_id = SliceSampleFileIdentifier(fname, slice_ind)
                        if sample_filter(slice_id):
                            self.sample_identifiers.append(slice_id)

            # save dataset metadata
            if save_metadata_to_cache:
                dataset_cache = {}
                dataset_cache[root] = self.sample_identifiers
                logging.info(f"Saving dataset cache to {metadata_cache_file}.")
                with open(metadata_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)

        ### random subsampling (1 sample = 1 slice from a MRI scan)

        # set default sampling mode to get the full dataset
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.sample_identifiers)
            num_samples = round(len(self.sample_identifiers) * sample_rate)
            self.sample_identifiers = self.sample_identifiers[:num_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.sample_identifiers])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.sample_identifiers = [
                sample_id
                for sample_id in self.sample_identifiers
                if sample_id[0].stem in sampled_vols
            ]

    def __len__(self):
        return len(self.sample_identifiers)

    def __getitem__(self, idx: int, mask: Optional[Callable] = None):
        fname, dataslice = self.sample_identifiers[idx]

        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]

            target = hf[self.recons_key][dataslice]

        if self.transforms is not None:
            target = self.transforms['transform_target'](target)
            kspace = self.transforms['transform_kspace'](kspace)

        return target, kspace
