from typing import Sequence, Union, Callable
from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from deepinv.datasets.fastmri import FastMRISliceDataset
from deepinv.physics.mri import MRIMixin

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover
    natsorted = ImportError(
        "natsort is not available. In order to use SKMTEASliceDataset, please install the natsort package with `pip install natsort`."
    )  # pragma: no cover


class SKMTEASliceDataset(FastMRISliceDataset, MRIMixin):
    """SKM-TEA dataset for raw multicoil MRI kspace data.

    Wraps the SKM-TEA dataset proposed in :footcite:t:`desai2021skm`.
    The dataset returns 2D slices from a dataset of 3D MRI volumes.

    To download raw data as `h5` files, see the `SKM-TEA website <https://github.com/StanfordMIMI/skm-tea>`_.

    The dataset is loaded as tuples `(x, y, params)` where:

    * `y` are the undersampled kspace measurements of shape ``(2, N, H, W)`` where N is the coil dimension.
    * `x` are the complex SENSE reconstructions from fully-sampled kspace of shape ``(2, H, W)``.
    * `params` is a dict containing parameters `mask` and `coil_maps` provided by the dataset, where `mask` are
      elliptical Poisson disc undersampling masks and `coil_maps` are sensitivity maps estimated using JSENSE.

    .. tip::

        The data can be directly related with :class:`deepinv.physics.MultiCoilMRI` ::

            x, y, params = next(iter(DataLoader(SKMTEADataset())))
            from deepinv.physics import MultiCoilMRI
            physics = MultiCoilMRI(**params)
            y1 = physics(x)

        Then `y` and `y1` are almost identical.

    **Raw data file structure:** (each file contains the k-space data and some metadata related to the scan) ::

        self.root --- xxx0.h5
                   |
                   -- xxx1.h5.

    When using this class, consider using the ``metadata_cache`` options to speed up class initialisation after the first initialisation.

    :param str, pathlib.Path root: Path to the dataset.
    :param int echo: which qDESS echo to use, defaults to 0.
    :param int acc: acceleration of mask to load, choose from 4, 6, 8, 10, 12 or 16.
    :param bool load_metadata_from_cache: Whether to load dataset metadata from cache.
    :param bool save_metadata_to_cache: Whether to cache dataset metadata.
    :param str, pathlib.Path metadata_cache_file: A file used to cache dataset information for faster load times.
    :param Callable filter_id: optional function that takes `SliceSampleID` named tuple and returns whether this id should be included.

    |sep|

    :Examples:

        Load data:

        >>> from deepinv.datasets import SKMTEADataset
        >>> from torch.utils.data import DataLoader
        >>> dataset = SKMTEADataset(".")
        >>> len(dataset) # Number of slices * number of volumes
        512
        >>> x, y, params = next(iter(DataLoader(dataset)))
        >>> x.shape # (B, 2, H, W)
        torch.Size([1, 2, 512, 160])
        >>> y.shape # (B, 2, N, H, W) # N coils
        torch.Size([1, 2, 8, 512, 160])

    """

    def __init__(
        self,
        root: str,
        echo: int = 0,
        acc: int = 6,
        load_metadata_from_cache: bool = False,
        save_metadata_to_cache: bool = False,
        metadata_cache_file: Union[str, Path] = "skmtea_dataset_cache.pkl",
        filter_id: Callable = None,
    ):

        self.root = Path(root)
        self.echo = echo
        self.acc = acc

        self.load_metadata_from_cache = load_metadata_from_cache
        self.save_metadata_to_cache = save_metadata_to_cache
        self.metadata_cache_file = metadata_cache_file

        all_fnames = natsorted(self.root.rglob("*.h5"))

        with self.metadata_cache_manager(self.root, []) as samples:
            if len(samples) == 0:
                for fname in tqdm(all_fnames):
                    metadata = self._retrieve_metadata(fname)
                    for slice_ind in range(metadata["num_slices"]):
                        samples.append(self.SliceSampleID(fname, slice_ind, metadata))

            self.samples = samples

        if filter_id is not None:
            self.samples = list(filter(filter_id, self.samples))

    @staticmethod
    def _retrieve_metadata(fname):
        with h5py.File(fname, "r") as hf:
            shape = hf["kspace"].shape
            metadata = {
                "num_slices": shape[0],
                "height": shape[1],
                "width": shape[2],
                "echos": shape[3],
                "coils": shape[4],
            }
        return metadata

    def zero_pad(
        self, x: torch.Tensor, shape: Sequence[int], mode="constant", value=0
    ) -> torch.Tensor:
        """Perform zero padding.

        Code taken from https://github.com/ad12/meddlr/blob/main/meddlr/ops/utils.py#L38
        """
        x_shape = x.shape[1 : 1 + len(shape)]

        total_padding = tuple(
            desired - current if desired is not None else 0
            for current, desired in zip(x_shape, shape, strict=True)
        )
        # Adding no padding for terminal dimensions.
        # torch.nn.functional.pad pads dimensions in reverse order.
        total_padding += (0,) * (len(x.shape) - 1 - len(x_shape))
        total_padding = total_padding[::-1]

        pad = []
        for padding in total_padding:
            pad1 = padding // 2
            pad2 = padding - pad1
            pad.extend([pad1, pad2])

        return F.pad(x, pad, mode=mode, value=value)

    def __getitem__(self, idx):
        """Load SKM-TEA data.

        Notation: `slice` is the slice dim, `H, W` are `y` and `z` kspace dims,
        `N` is the coil dim, `E` is the echo dimension.

        Raw data shapes and types:

        * `x`: `(slice, H, W, N, 1)` complex
        * `y`: `(slice, H, W, E, N)` complex
        * `mask`: `(h, w)` bool where `h, w` are mask shape
        * `maps`: `(slice, H, W, E, 1)` complex

        Return data shapes and types, following convention in :class:`deepinv.physics.MultiCoilMRI`:

        * `x`: `(2, H, W)` real
        * `y`: `(2, N, H, W)` real
        * `mask`: `(1, H, W)` real
        * `maps`: `(N, H, W)` complex

        """
        fname, slice_ind, metadata = self.samples[idx]

        with h5py.File(fname, "r") as f:
            x = torch.as_tensor(f["target"][[slice_ind], :, :, self.echo, 0])
            y = torch.as_tensor(f["kspace"][[slice_ind], :, :, self.echo, :])
            mask = torch.as_tensor(np.array(f[f"masks/poisson_{self.acc}.0x"]))
            maps = torch.as_tensor(f["maps"][[slice_ind], :, :, :, 0])

        mask = (
            self.zero_pad(mask.unsqueeze(0), y.shape[1:3]) * 1.0
        )  # (h, w) -> (1, H, W)

        y = y.moveaxis(-1, 1)  # (1, H, W, N) -> (1, N, H, W) complex
        y = self.from_torch_complex(y)  # (1, N, H, W) complex -> (1, 2, N, H, W) real
        y = y.squeeze(0) * mask.unsqueeze(0)

        return (
            self.from_torch_complex(x).squeeze(0),
            y,
            {"mask": mask, "coil_maps": maps.moveaxis(-1, 1).squeeze(0)},
        )
