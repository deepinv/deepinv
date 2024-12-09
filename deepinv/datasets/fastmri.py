r"""Pytorch Dataset for fastMRI.

Code modified from: https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/mri_data.py

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Key modifications:
- Use torch.randperm instead of random shuffle
- Convert to SimpleFastMRISliceDataset
- Return torch tensors
- Remove redundant sample_rate (this can be easily achieved using torch.utils.data.Subset)
- Add slice_index parameter
- Remove redundant challenge parameter
- Clean up messy fastmri code
"""

from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union, Tuple
from collections import defaultdict
import pickle
import warnings
import os
import h5py
from tqdm import tqdm
import torch
from torchvision.transforms import Compose, CenterCrop
from deepinv.datasets.utils import ToComplex, Rescale, download_archive
from deepinv.utils.demo import get_image_dataset_url


class SimpleFastMRISliceDataset(torch.utils.data.Dataset):
    """Simple FastMRI image dataset.

    Loads a saved subset of 2D slices from the full FastMRI slice dataset.

    We provide a pregenerated saved subset for singlecoil FastMRI knees (total 973 images)
    and RSS reconstructions of multicoil brains (total 455 images).
    These originate from their respective original fully-sampled train volumes.
    Each slice is the middle slice from one independent volume.
    The images are of shape (2x320x320) and are normalised (0-1) and padded.
    Download the dataset using ``download=True``, and load them using the ``anatomy`` argument.

    These datasets were generated using :meth:`deepinv.datasets.fastmri.FastMRISliceDataset.save_simple_dataset`.
    You can use this to generate a custom dataset and load using the ``file_name`` argument.

    :param str, Path root_dir: dataset root directory
    :param str, Path file_name: optional, name of local dataset to load, overrides ``anatomy``. If ``None``, load dataset based on ``anatomy`` parameter.
    :param str anatomy: load either fastmri "knee" or "brain" slice datasets.
    :param bool train: whether to use training set or test set, defaults to True
    :param int sample_index: if specified only load this sample, defaults to None
    :param float train_percent: percentage train for train/test split, defaults to 0.925
    :param callable transform: optional transform for images, defaults to None
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        file_name: Union[str, Path] = None,
        anatomy: str = "knee",
        train: bool = True,
        sample_index: int = None,
        train_percent: float = 0.925,
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        if anatomy not in ("knee", "brain", None):
            raise ValueError("anatomy must be either 'knee' or 'brain' or None.")
        elif anatomy is None and file_name is None:
            raise ValueError("Either anatomy or file_name must be passed.")

        os.makedirs(root_dir, exist_ok=True)
        root_dir = Path(root_dir)
        file_name = (
            file_name
            if file_name is not None
            else Path(f"fastmri_{anatomy}_singlecoil.pt")
        )

        try:
            x = torch.load(root_dir / file_name)
        except FileNotFoundError:
            if download:
                url = get_image_dataset_url(str(file_name), None)
                download_archive(url, root_dir / file_name)
                x = torch.load(root_dir / file_name)
            else:
                raise FileNotFoundError(
                    "Local dataset not downloaded. Download by setting download=True."
                )

        x = x.squeeze()
        self.transform = Compose(
            [ToComplex()] + ([transform] if transform is not None else [])
        )

        if train:
            self.x = x[: int(train_percent * len(x))]
        else:
            self.x = x[int(train_percent * len(x)) :]

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Load images from memory and optionally apply transform.

        :param int index: dataset index
        :return: tensor of shape (2, H, W)
        """
        x = self.x[index]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.x)


class FastMRISliceDataset(torch.utils.data.Dataset):
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to MR image slices.

    This dataset randomly selects 2D slices from a dataset of 3D MRI volumes.
    This class considers one data sample as one slice of a MRI scan, thus slices of the same MRI scan are considered independently in the dataset.

    To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/`
    The fastMRI dataset includes two types of MRI scans: knee MRIs and the brain (neuro) MRIs, and containing training, validation, and masked test sets.

    The dataset is loaded as pairs ``(kspace, target)`` where ``kspace`` are the measurements of shape ``(2, (N,) H, W)``
    where N is the optional coil dimension depending on whether the data is singlecoil or multicoil,
    and ``target`` are the magnitude root-sum-square reconstructions of shape ``(H, W)``.

    See the `fastMRI README <https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md>`_ for more details.

    **Raw data file structure:** ::

        self.root --- file1000005.h5
                   |
                   -- xxxxxxxxxxx.h5

    Each file contains the k-space data, ground truth and some metadata related to the scan.
    When using this class, consider using the ``metadata_cache`` options to speed up class initialisation after the first initialisation.

    .. note::

        We also provide a simple FastMRI dataset class in :class:`deepinv.datasets.fastmri.SimpleFastMRISliceDataset`.
        This allows you to save and load the dataset as 2D singlecoil slices much faster and all in-memory.
        You can generate this using the method ``save_simple_dataset``.

    :param Union[str, Path] root: Path to the dataset.
    :param bool test: Whether the split is the "test" set, if yes, only return kspace measurements.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, Path] metadata_cache_file: A file used to cache dataset information for faster load times.
    :param float, optional subsample_volumes: proportion of volumes to be randomly subsampled (float between 0 and 1).
    :param str, int, tuple slice_index: if "all", keep all slices per volume, if ``int``, keep only that indexed slice per volume,
        if "middle", keep the middle slice. If "random", select random slice. Defaults to "all".
    :param callable, optional transform_kspace: transform function for (multicoil) kspace operating on images of shape (..., 2, H, W).
    :param callable, optional transform_target: transform function for ground truth recon targets operating on single-channel images of shape (H, W).
    :param torch.Generator, None rng: optional torch random generator for shuffle slice indices

    |sep|

    :Examples:

        Instantiate train dataset:

            from deepinv.datasets import FastMRISliceDataset
            root = "/path/to/dataset/fastMRI/knee_singlecoil/train"
            dataset = FastMRISliceDataset(root=root)
            target, kspace = dataset[0]
            print(target.shape)
            print(kspace.shape)

        Instantiate dataset with metadata cache (speeds up subsequent instantiation)

            dataset = FastMRISliceDataset(root=root, load_metadata_from_cache=True, save_metadata_to_cache=True)

        Normalise and pad images:

            from torchvision.transforms import Compose, CenterCrop
            from deepinv.utils import Rescale
            dataset = FastMRISliceDataset(root=root, transform_target=Compose([Rescale(), CenterCrop(320)]))

    """

    @staticmethod
    def torch_shuffle(x: list, generator=None):
        return [x[i] for i in torch.randperm(len(x), generator=generator).tolist()]

    class SliceSampleFileIdentifier(NamedTuple):
        """Data structure for identifying specific slices within MRI data files."""

        fname: Path
        slice_ind: int

    def __init__(
        self,
        root: Union[str, Path],
        test: bool = False,
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        slice_index: Union[str, int] = "all",
        subsample_volumes: Optional[float] = 1.0,
        transform_kspace: Optional[Callable] = None,
        transform_target: Optional[Callable] = None,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.root = root
        self.test = test
        self.transform_kspace = transform_kspace
        self.transform_target = transform_target

        if not os.path.isdir(root):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{root}`."
            )

        if not all([file.endswith(".h5") for file in os.listdir(root)]):
            raise ValueError(
                f"The `root` folder doesn't contain only hdf5 files. Please set `root` properly. Current value `{root}`."
            )

        if slice_index not in ("all", "random", "middle") and not isinstance(
            slice_index, int
        ):
            raise ValueError('slice_index must be "all", "random", "middle", or int.')

        # Load all slices
        self.sample_identifiers = defaultdict(list)
        all_fnames = sorted(list(Path(root).iterdir()))

        if load_metadata_from_cache and os.path.exists(metadata_cache_file):
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
            if load_metadata_from_cache and not os.path.exists(metadata_cache_file):
                warnings.warn(
                    f"Couldn't find dataset cache at {metadata_cache_file}. Loading dataset from scratch."
                )

            for fname in tqdm(all_fnames):
                with h5py.File(fname, "r") as hf:
                    for i in range(hf["kspace"].shape[0]):
                        self.sample_identifiers[str(fname)].append(
                            self.SliceSampleFileIdentifier(fname, i)
                        )

            if save_metadata_to_cache:
                dataset_cache = {}
                dataset_cache[root] = self.sample_identifiers
                print(f"Saving dataset cache to {metadata_cache_file}.")
                with open(metadata_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)

        # Random slice subsampling
        if slice_index != "all":
            for fname, samples in self.sample_identifiers.items():
                if isinstance(slice_index, int):
                    chosen_sample = samples[slice_index]
                elif slice_index == "middle":
                    chosen_sample = samples[len(samples) // 2]
                elif slice_index == "random":
                    chosen_sample = self.torch_shuffle(samples, generator=rng)[0]
                self.sample_identifiers[fname] = [chosen_sample]

        # Randomly keep a portion of MRI volumes
        if subsample_volumes < 1.0:
            subsampled_fnames = self.torch_shuffle(
                list(self.sample_identifiers.keys()), generator=rng
            )[: round(len(all_fnames) * subsample_volumes)]
            self.sample_identifiers = {
                k: self.sample_identifiers[k] for k in subsampled_fnames
            }

        # Flatten to list of samples
        self.sample_identifiers = [
            samp for samps in self.sample_identifiers.values() for samp in samps
        ]

    def __len__(self) -> int:
        return len(self.sample_identifiers)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        r"""Returns the idx-th sample from the dataset, i.e. kspace of shape (2, (N,) H, W) and target of shape (2, H, W)"""
        fname, dataslice = self.sample_identifiers[idx]

        with h5py.File(fname, "r") as hf:
            kspace = torch.from_numpy(
                hf["kspace"][dataslice]
            )  # ((N,) H, W) dtype complex
            kspace = torch.view_as_real(kspace)  # ((N,) H, W, 2)
            kspace = kspace.moveaxis(-1, -3)  # ((N,) 2, H, W)
            if self.transform_kspace is not None:
                kspace = self.transform_kspace(
                    kspace
                )  # torchvision transforms require (..., C, H, W)
            kspace = kspace.moveaxis(-3, 0)  # (2, N, H, W)

            if not self.test:
                recons_key = (
                    "reconstruction_esc"
                    if "reconstruction_esc" in hf.keys()
                    else "reconstruction_rss"
                )
                target = torch.from_numpy(hf[recons_key][dataslice])  # shape (H, W)

                if self.transform_target is not None:
                    target = self.transform_target(target)

        return kspace if self.test else (target, kspace)

    def save_simple_dataset(
        self,
        dataset_path: str,
        pad_to_size: Tuple[int] = (320, 320),
        to_complex: bool = False,
    ) -> SimpleFastMRISliceDataset:
        """Convert dataset to a 2D singlecoil dataset and save as pickle file.
        This allows the dataset to be loaded in memory with :class:`deepinv.datasets.fastmri.SimpleFastMRISliceDataset`.

        :Example:

            from deepinv.datasets import FastMRISliceDataset
            root = "/path/to/dataset/fastMRI/brain/multicoil_train"
            dataset = FastMRISliceDataset(root=root, slice_index="middle")
            subset = dataset.save_simple_dataset(
                root + "/fastmri_brain_singlecoil.pt"
            )

        :param str dataset_path: desired path of dataset to be saved with file extension e.g. ``fastmri_knee_singlecoil.pt``.
        :param bool pad_to_size: if not None, normalise images to 0-1 then pad to provided shape. Must be set if images are of varying size,
            in order to successfully stack images to tensor.
        :return: loaded SimpleFastMRISliceDataset
        :rtype: SimpleFastMRISliceDataset
        """
        if self.test:
            raise ValueError(
                "save_simple_dataset can only be used when targets are provided, i.e. test = False."
            )

        transform_target = self.transform_target

        transform_list = [transform_target] if transform_target is not None else []
        transform_list += [Rescale()]
        if pad_to_size is not None:
            transform_list += [CenterCrop(pad_to_size)]
        if to_complex:
            transform_target += [ToComplex()]
        self.transform_target = Compose(transform_list)

        xs = [self.__getitem__(i)[0] for i in tqdm(range(self.__len__()))]

        torch.save(torch.stack(xs), str(dataset_path))

        self.transform_target = transform_target

        dataset_path = Path(dataset_path)

        return SimpleFastMRISliceDataset(
            root_dir=dataset_path.parent,
            file_name=dataset_path.name,
            anatomy=None,
            train=True,
            train_percent=1.0,
            transform=None,
            download=False,
        )
