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
from deepinv.utils.demo import get_image_url


class SimpleFastMRISliceDataset(torch.utils.data.Dataset):
    """Simple FastMRI image dataset.

    Loads in-memory a saved and processed subset of 2D slices from the full FastMRI slice dataset for quick loading.

    .. important::

        By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.

    These datasets are generated using :func:`deepinv.datasets.FastMRISliceDataset.save_simple_dataset`.
    You can use this to generate your own custom dataset and load using the ``file_name`` argument.

    We provide a pregenerated mini saved subset for singlecoil FastMRI knees (total 2 images)
    and RSS reconstructions of multicoil brains (total 2 images).
    These originate from their respective fully-sampled volumes converted to images via root-sum-of-square (RSS).
    Each slice is the middle slice from one independent volume.
    The images are of shape (2x320x320) and are normalised per-sample (0-1) and padded.
    Download the dataset using ``download=True``, and load them using the ``anatomy`` argument.

    .. note ::

        Since images are obtained from RSS, the imaginary part of each sample is 0.

    |sep|

    :Examples:

        Load mini demo knee dataset:

        >>> from deepinv.datasets import SimpleFastMRISliceDataset
        >>> from deepinv.utils import get_data_home
        >>> dataset = SimpleFastMRISliceDataset(get_data_home(), anatomy="knee", download=True)
        >>> len(dataset)
        2

    :param str, pathlib.Path root_dir: dataset root directory
    :param str anatomy: load either fastmri "knee" or "brain" slice datasets.
    :param str, pathlib.Path file_name: optional, name of local dataset to load, overrides ``anatomy``. If ``None``, load dataset based on ``anatomy`` parameter.
    :param bool train: whether to use training set or test set, defaults to True
    :param int sample_index: if specified only load this sample, defaults to None
    :param float train_percent: percentage train for train/test split, defaults to 1.
    :param Callable transform: optional transform for images, defaults to None
    :param bool download: If ``True``, downloads the dataset from the internet and puts it in root directory.
        If dataset is already downloaded, it is not downloaded again. Default at False.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        anatomy: str = "knee",
        file_name: Union[str, Path] = None,
        train: bool = True,
        sample_index: int = None,
        train_percent: float = 1.0,
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
            else Path(f"demo_mini_subset_fastmri_{anatomy}.pt")
        )

        try:
            x = torch.load(root_dir / file_name, weights_only=True)
        except FileNotFoundError:
            if download:
                url = get_image_url(str(file_name))
                download_archive(url, root_dir / file_name)
                x = torch.load(root_dir / file_name, weights_only=True)
            else:
                raise FileNotFoundError(
                    "Local dataset not downloaded. Download by setting download=True."
                )

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
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to raw MR image slices.

    This dataset randomly selects 2D slices from a dataset of 3D MRI volumes.
    This class considers one data sample as one slice of a MRI scan, thus slices of the same MRI scan are considered independently in the dataset.

    To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/` to download the volumes as ``h5`` files.
    The fastMRI dataset includes two types of MRI scans: knee MRIs and the brain (neuro) MRIs, and containing training, validation, and masked test sets.

    The dataset is loaded as pairs ``(kspace, target)`` where ``kspace`` are the measurements of shape ``(2, (N,) H, W)``
    where N is the optional coil dimension depending on whether the data is singlecoil or multicoil,
    and ``target`` are the magnitude root-sum-square reconstructions of shape ``(1, H, W)``.

    .. tip::

        ``x`` and ``y`` are related by :meth:`deepinv.physics.MRI.A_adjoint` or :meth:`deepinv.physics.MultiCoilMRI.A_adjoint`
        depending on if ``y`` are multicoil or not, with ``crop=True, rss=True``.

    See the `fastMRI README <https://github.com/facebookresearch/fastMRI/blob/main/fastmri/data/README.md>`_ for more details.

    **Raw data file structure:** ::

        self.root --- file1000005.h5
                   |
                   -- xxxxxxxxxxx.h5

    Each file contains the k-space data, reconstructed images and some metadata related to the scan.
    When using this class, consider using the ``metadata_cache`` options to speed up class initialisation after the first initialisation.

    .. note::

        We also provide a simple FastMRI dataset class in :class:`deepinv.datasets.fastmri.SimpleFastMRISliceDataset`.
        This allows you to save and load the dataset as 2D singlecoil slices much faster and all in-memory.
        You can generate this using the method ``save_simple_dataset``.

    .. important::

        By using this dataset, you confirm that you have agreed to and signed the `FastMRI data use agreement <https://fastmri.med.nyu.edu/>`_.


    :param Union[str, pathlib.Path] root: Path to the dataset.
    :param bool test: Whether the split is the `"test"` set, if yes, only return kspace measurements.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, pathlib.Path] metadata_cache_file: A file used to cache dataset information for faster load times.
    :param float subsample_volumes: (optional) proportion of volumes to be randomly subsampled (float between 0 and 1).
    :param str, int, tuple slice_index: if `"all"`, keep all slices per volume, if ``int``, keep only that indexed slice per volume,
        if `"middle"`, keep the middle slice. If `"random"`, select random slice. Defaults to `"all"`.
    :param Callable transform_kspace: optional transform function for (multicoil) kspace operating on images of shape (..., 2, H, W).
    :param Callable transform_target: optional transform function for ground truth recon targets operating on single-channel images of shape (1, H, W).
    :param torch.Generator, None rng: optional torch random generator for shuffle slice indices

    |sep|

    :Examples:

        Instantiate dataset with sample data (from a demo multicoil brain volume):

        >>> from deepinv.datasets import FastMRISliceDataset, download_archive
        >>> from deepinv.utils import get_image_url, get_data_home
        >>> url = get_image_url("demo_fastmri_brain_multicoil.h5")
        >>> root = get_data_home() / "fastmri" / "brain"
        >>> download_archive(url, root / "demo.h5")
        >>> dataset = FastMRISliceDataset(root=root, slice_index="all")
        >>> len(dataset)
        16
        >>> target, kspace = dataset[0]
        >>> target.shape # (1, W, W), varies per sample
        torch.Size([1, 213, 213])
        >>> kspace.shape # (2, N, H, W), varies per sample
        torch.Size([2, 4, 512, 213])

        Normalise and pad images, and load one slice per volume:

        >>> from torchvision.transforms import Compose, CenterCrop
        >>> from deepinv.datasets.utils import Rescale
        >>> dataset = FastMRISliceDataset(root=root, slice_index=0, transform_target=Compose([Rescale(), CenterCrop(320)]))

        Convert to a simple normalised padded in-memory slice dataset from the middle slices only:

        >>> simple_set = FastMRISliceDataset(root=root, slice_index="middle").save_simple_dataset(root.parent / "simple_set.pt")
        >>> len(simple_set)
        1

        Instantiate dataset with metadata cache (speeds up subsequent instantiation):

        >>> dataset = FastMRISliceDataset(root=root, load_metadata_from_cache=True, save_metadata_to_cache=True, metadata_cache_file=root.parent / "cache.pkl") # doctest: +ELLIPSIS
        Saving dataset cache to ...
        >>> import shutil; shutil.rmtree(root.parent)

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
                # torchvision transforms require (..., C, H, W)
                kspace = self.transform_kspace(kspace)
            kspace = kspace.moveaxis(-3, 0)  # (2, N, H, W)

            if not self.test:
                recons_key = (
                    "reconstruction_esc"
                    if "reconstruction_esc" in hf.keys()
                    else "reconstruction_rss"
                )
                # to shape (1, H, W)
                target = torch.from_numpy(hf[recons_key][dataslice]).unsqueeze(0)

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

            Load local brain dataset and convert to simple dataset ::

                from deepinv.datasets import FastMRISliceDataset
                root = "/path/to/dataset/fastMRI/brain/multicoil_train"
                dataset = FastMRISliceDataset(root=root, slice_index="middle")
                subset = dataset.save_simple_dataset(root + "/fastmri_brain_singlecoil.pt")

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

        xs = [self.__getitem__(i)[0].squeeze(0) for i in tqdm(range(self.__len__()))]

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
