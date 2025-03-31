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
from contextlib import contextmanager
from typing import Any, Callable, NamedTuple, Optional, Union, Tuple, Dict, Any
from collections import defaultdict
import pickle
import math
import warnings
import os
from natsort import natsorted
import h5py
from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, CenterCrop

from deepinv.datasets.utils import ToComplex, Rescale, download_archive, loadmat
from deepinv.utils.demo import get_image_url
from deepinv.physics.generator.mri import BaseMaskGenerator, ceildiv
from deepinv.physics.mri import MultiCoilMRI, MRIMixin
from deepinv.physics.noise import NoiseModel

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


class FastMRISliceDataset(torch.utils.data.Dataset, MRIMixin):
    """Dataset for `fastMRI <https://fastmri.med.nyu.edu/>`_ that provides access to raw MR image slices.

    This dataset randomly selects 2D slices from a dataset of 3D MRI volumes.
    This class considers one data sample as one slice of a MRI scan, thus slices of the same MRI scan are considered independently in the dataset.

    To download raw data, please go to the bottom of the page `https://fastmri.med.nyu.edu/` to download the brain/knee and train/validation/test volumes as ``h5`` files.

    The dataset is loaded as pairs ``(x, y)`` where `y` are the kspace measurements of shape ``(2, (N,) H, W)``
    where N is the optional coil dimension depending on whether the data is singlecoil or multicoil,
    and `x` ("target") are the magnitude root-sum-square reconstructions of shape ``(1, H, W)``.

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
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param Union[str, pathlib.Path] metadata_cache_file: A file used to cache dataset information for faster load times.
    :param float subsample_volumes: (optional) proportion of volumes to be randomly subsampled (float between 0 and 1).
    :param str, int, tuple slice_index: if `"all"`, keep all slices per volume, if ``int``, keep only that indexed slice per volume,
        if ``tuple`` or `int`s, index those slices, if `"middle"`, keep the middle slice, if `"middle+i"`, keep :math:`2i+1` about
        middle slice, if `"random"`, select random slice. Defaults to `"all"`.
    :param Callable transform: optional transform function taking in (multicoil) kspace of shape (2, (N,) H, W) and targets of shape (1, H, W).
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

        Load one slice per volume:

        >>> dataset = FastMRISliceDataset(root=root, slice_index=0)

        Use MRI transform to mask, estimate sensitivity maps, normalise and/or crop:

        >>> #TODO

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
    def torch_shuffle(x: list, generator: torch.Generator = None) -> list:
        """Shuffle list reproducibly using torch generator.

        :param list x: list to be shuffled
        :param torch.Generator generator: torch Generator.
        :return list: shuffled list
        """
        return [x[i] for i in torch.randperm(len(x), generator=generator).tolist()]

    class SliceSampleID(NamedTuple):
        """Data structure containing ID and metadata of specific slices within MRI data files."""

        fname: Path
        slice_ind: int
        metadata: Dict[str, Any]

    @contextmanager
    def metadata_cache_manager(self, root: Union[str, Path], samples: Any):
        """Read/write metadata cache file for populating list of sample ids.

        :param Union[str, pathlib.Path] root: root dir to save to metadata cache
        :param Any samples: iterable (list, dict etc.) for populating with samples to read/write to metadata cache
        :yield: samples, either populated from metadata cache, or blank, to be yielded to be written to.
        """
        if self.load_metadata_from_cache and os.path.exists(self.metadata_cache_file):
            with open(self.metadata_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
                if dataset_cache.get(root) is None:
                    raise ValueError(
                        "`metadata_cache_file` doesn't contain the metadata. Please "
                        + "either deactivate `load_dataset_from_cache` OR set `metadata_cache_file` properly."
                    )
                print(f"Using dataset cache from {self.metadata_cache_file}.")
                samples = dataset_cache[root]

            yield samples

        else:
            if self.load_metadata_from_cache and not os.path.exists(
                self.metadata_cache_file
            ):
                warnings.warn(
                    f"Couldn't find dataset cache at {self.metadata_cache_file}. Loading dataset from scratch."
                )

            yield samples

            if self.save_metadata_to_cache:
                dataset_cache = {}
                dataset_cache[root] = samples
                print(f"Saving dataset cache to {self.metadata_cache_file}.")
                with open(self.metadata_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)

    def __init__(
        self,
        root: Union[str, Path],
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "dataset_cache.pkl",
        slice_index: Union[str, int] = "all",
        subsample_volumes: Optional[float] = 1.0,
        transform: Optional[Callable] = None,
        filter_id: Optional[Callable] = None,
        rng: Optional[torch.Generator] = None,

    ) -> None:
        self.root = root
        self.transform = transform
        self.load_metadata_from_cache = load_metadata_from_cache
        self.save_metadata_to_cache = save_metadata_to_cache
        self.metadata_cache_file = metadata_cache_file

        if not os.path.isdir(root):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{root}`."
            )    

        # Load all slices
        all_fnames = sorted(list(Path(root).glob("*.h5")))

        with self.metadata_cache_manager(root, defaultdict(list)) as samples:
            if len(samples) == 0:
                for fname in tqdm(all_fnames):
                    metadata = self._retrieve_metadata(fname)
                    for slice_ind in range(metadata["num_slices"]):
                        samples[str(fname)].append(
                            self.SliceSampleID(fname, slice_ind, metadata)
                        )

            self.samples = samples

        # Random slice subsampling
        if slice_index != "all":
            for fname, samples in self.samples.items():
                if isinstance(slice_index, int):
                    chosen = samples[slice_index]
                elif isinstance(slice_index, (tuple, list)):
                    chosen = [samples[i] for i in slice_index]
                elif "middle" in slice_index:
                    i = slice_index.split("+")[-1]
                    i = int(i) if "+" in slice_index and i.isdigit() else 0
                    chosen = samples[len(samples) // 2 - i: len(samples) // 2 + i + 1]
                elif slice_index == "random":
                    chosen = self.torch_shuffle(samples, generator=rng)[0]
                else:
                    raise ValueError('slice_index must be "all", "random", "middle", "middle+i", int or tuple.')
                
                self.samples[fname] = chosen if isinstance(chosen, list) else [chosen]

        # Randomly keep a portion of MRI volumes
        if subsample_volumes < 1.0:
            subsampled_fnames = self.torch_shuffle(
                list(self.samples.keys()), generator=rng
            )[: round(len(all_fnames) * subsample_volumes)]
            self.samples = {k: self.samples[k] for k in subsampled_fnames}

        # Flatten to list of samples
        self.samples = [samp for samps in self.samples.values() for samp in samps]

        if filter_id is not None:
            self.samples = list(filter(filter_id, self.samples))

    @staticmethod
    def _retrieve_metadata(fname: Union[str, Path, os.PathLike]) -> Dict[str, Any]:
        """Open file and retrieve metadata.
        Metadata includes number of slices in volume.

        :param Union[str, pathlib.Path, os.PathLike] fname: filename to open
        :return: metadata dict of key-value pairs.
        """
        with h5py.File(fname, "r") as hf:
            shape = hf["kspace"].shape
            metadata = {
                "width": shape[-1],  # W
                "height": shape[-2],  # H
                "num_slices": shape[0],  # D (depth)
            } | (
                {
                    "coils": shape[1],  # N (coils)
                }
                if len(shape) == 4
                else {}
            )

        return metadata

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        r"""Returns idx-th target of shape (1, H, W) and kspace of shape (2, (N,) H, W)
        if target exists else just space (i.e. in challenge test set).
        Outputs may be modifed by transform if specified."""
        fname, slice_ind, metadata = self.samples[idx]

        with h5py.File(fname, "r") as hf:
            # ((N,) H, W) dtype complex -> (2, (N,) H, W) real
            kspace = self.from_torch_complex(torch.from_numpy(hf["kspace"][slice_ind]).unsqueeze(0)).squeeze(0)

            if any("reconstruction" in key for key in hf.keys()):
                # shape (1, H, W)
                target = torch.from_numpy(hf[
                    "reconstruction_esc"
                    if "reconstruction_esc" in hf.keys()
                    else "reconstruction_rss"
                ][slice_ind]).unsqueeze(0)
            else:
                target = None

        #TODO load mask if exists (i.e. challenge test set)

        if self.transform is not None:
            return self.transform(target, kspace, seed=str(fname) + str(slice_ind))
        else:
            return (target, kspace) if target is not None else kspace

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
        transform = [Rescale()]
        if pad_to_size is not None:
            transform += [CenterCrop(pad_to_size)]
        if to_complex:
            transform += [ToComplex()]
        transform = Compose(transform)

        xs = [transform(self.__getitem__(i)[0]).squeeze(0) for i in tqdm(range(self.__len__()))]

        torch.save(torch.stack(xs), str(dataset_path))

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

    def save_local_dataset(self, data_dir: str) -> torch.utils.data.Dataset:
        #TODO generalise local dataset for all datasets
        os.makedirs(data_dir, exist_ok=True)
        for i in tqdm(range(self.__len__())):
            x, y, params = self.__getitem__(i)
            np.savez_compressed(
                f"{str(data_dir)}/{i}.npz",
                x=x.numpy(),
                y=y.numpy(),
                **{k: v.numpy() for (k, v) in params.items()},
            )
        return LocalDataset(data_dir)


class LocalDataset(torch.utils.data.Dataset):
    def __init__(self, root: Union[str, Path], pattern: str = "*.npz", cache: float = False):
        self.files = natsorted(Path(root).glob(pattern))
        self.use_cache = cache
        self.cache = {}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        
        if f in self.cache:
            data = self.cache[f]
        else:
            with np.load(f) as d:
                data = dict(d)
            if self.use_cache:
                self.cache[f] = data

        x = torch.tensor(data["x"]) if "x" in data else None
        y = torch.tensor(data["y"])
        
        params = {k: torch.tensor(data[k]) for k in ["mask", "coil_maps"] if k in data}

        if x is None and not params:
            return y
        
        return (() if x is None else (x,)) + (y,) + ((params,) if params else ())


class FastMRITransform:
    def __init__(
        self,
        mask_generator: Optional[BaseMaskGenerator] = None,
        estimate_coil_maps: Union[bool, int] = False,
    ):
        if 1:
            # TODO mask_generator is None and estimate_coil_maps is True but not int. Either pass in mask_generator, or specify acs size with estimate_coil_maps
            pass

        self.mask_generator = mask_generator
        self.estimate_coil_maps = estimate_coil_maps

    def generate_mask(self, kspace: torch.Tensor, seed: Union[str, int]) -> torch.Tensor:
        return self.mask_generator.step(
            seed=seed,
            img_size=kspace.shape[-2:],
            batch_size=0,
        )["mask"]

    def generate_maps(self, kspace: torch.Tensor) -> torch.Tensor:
        calib_size = (
            self.mask_generator.n_center
            if self.estimate_coil_maps == True
            else self.estimate_coil_maps
        )
        coil_maps = MultiCoilMRI.estimate_coil_maps(
            kspace.unsqueeze(0), calib_size=calib_size
        ).squeeze(0)
        #TODO this should be optional
        coil_maps[coil_maps == 0] = 1 / math.sqrt(coil_maps.shape[0])
        return coil_maps

    def __call__(self, target: torch.Tensor, kspace: torch.Tensor, seed: Union[str, int] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # target of shape (1, H, W) and kspace of shape (2, (N,) H, W)
        params = {}
        if self.mask_generator is not None:
            params["mask"] = self.generate_mask(kspace, seed)
            kspace *= params["mask"]
        if self.estimate_coil_maps:
            params["coil_maps"] = self.generate_maps(kspace)

        #TODO if just (y,) then return just y
        return (
            (() if target is None else (target,)) + (kspace,) + ((params,) if params else ())
        )

class FullMultiCoilFastMRITransform(FastMRITransform, MRIMixin):
    def __init__(
        self,
        mask_generator: Optional[BaseMaskGenerator] = None,
        estimate_coil_maps: Union[bool, int] = True,
        crop_size: Tuple[int, int] = (384, 320),
        rescale: bool = False,
        prewhiten: Tuple[slice, slice] = (slice(0, 30), slice(0, 30)),
        noise_model: NoiseModel = None,
        norm_percentile: float = .99,
        target_method: str = "A_dagger",
    ):
        super().__init__(mask_generator, estimate_coil_maps)
        self.crop_size = crop_size
        self.prewhiten = prewhiten
        self.norm_percentile = norm_percentile
        self.noise_model = noise_model
        self.rescale = rescale
        self.target_method = target_method
    
    def prewhiten_kspace(self, kspace: torch.Tensor) -> torch.Tensor:
        # kspace of shape (2, (N,) H, W)
        ksp = self.to_torch_complex(kspace.unsqueeze(0)).squeeze(0) # N, H, W complex

        n = ksp[:, self.prewhiten[0], self.prewhiten[1]].flatten(-2)
        n -= n.mean(axis=-1, keepdim=True)
        cov = n @ n.H # (N, N) complex

        try:
            ksp = ksp.flatten(-2) # N, H*W
            ksp = torch.linalg.solve(torch.linalg.cholesky(cov), ksp)
            return self.from_torch_complex(ksp.unflatten(-1, kspace.shape[-2:]).unsqueeze(0)).squeeze(0) # 2,N,H,W
        except torch.linalg.LinAlgError:
            # Non-PSD due to ksp all zeros
            return kspace

    def normalise(self, kspace: torch.Tensor) -> torch.Tensor:
        # Extract ACS
        acs = (
            self.mask_generator.n_center
            if self.estimate_coil_maps == True
            else self.estimate_coil_maps
        )
        H, W = kspace.shape[-2:]
        mask = torch.zeros_like(kspace)
        mask[..., H // 2 - acs // 2 : H // 2 + ceildiv(acs, 2), W // 2 - acs // 2 : W // 2 + ceildiv(acs, 2)] = 1

        # Normalise by percentile of RSS of ACS
        return kspace / self.rss(self.kspace_to_im((kspace * mask).unsqueeze(0))).quantile(0.99)


    def __call__(self, _: Any, kspace: torch.Tensor, seed: Union[str, int] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # kspace of shape (2, (N,) H, W)

        # Pre-whiten
        if self.prewhiten:
            kspace = self.prewhiten_kspace(kspace)

        # Add noise
        if self.noise_model is not None:
            kspace = self.noise_model(kspace) #TODO string seed here

        # Crop in image space
        kspace = self.im_to_kspace(self.crop(
            self.kspace_to_im(kspace.unsqueeze(0)),
            shape=self.crop_size,
            rescale=self.rescale,
        )).squeeze(0)

        # Normalise kspace
        kspace = self.normalise(kspace)

        # Estimate sens maps
        params = {"coil_maps": self.generate_maps(kspace)}

        # PI-CS (least squares with CG and no regularisation)
        physics = MultiCoilMRI(img_size=self.crop_size, **params)
        match self.target_method:
            case "A_dagger":
                target = physics.A_dagger(kspace.unsqueeze(0)).squeeze(0)
            case "A_adjoint":
                target = physics.A_adjoint(kspace.unsqueeze(0)).squeeze(0)
            case "rss":
                target = physics.A_adjoint(kspace.unsqueeze(0), rss=True).squeeze(0) # 1,H,W
                target = torch.cat([target, torch.zeros_like(target)], dim=0)
            case _:
                raise ValueError("target_method invalid.")


        # Generate masks
        if self.mask_generator is not None:
            params["mask"] = self.generate_mask(kspace, seed)
            kspace *= params["mask"]

        # Shapes:
        # target 2, H, W float32
        # kspace 2, N, H, W float32
        # mask 1, H, W float32
        # maps N, H, W complex64
        return target, kspace, params