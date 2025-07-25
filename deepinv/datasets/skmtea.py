import torch
import h5py
import torch.nn.functional as F
from typing import Sequence, Union, Callable
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm

from deepinv.datasets.fastmri import FastMRISliceDataset
from deepinv.physics.mri import MRIMixin

class SKMTEADataset(FastMRISliceDataset, MRIMixin):
    def __init__(
            self, 
            root: str, 
            echo: int = 0,
            load_metadata_from_cache: bool = False,
            save_metadata_to_cache: bool = False,
            metadata_cache_file: Union[str, Path] = "skmtea_dataset_cache.pkl",
            filter_id: Callable = None
        ):
        self.root = Path(root)
        self.echo = echo

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
            metadata = (
                {
                    "num_slices": shape[0],
                    "height": shape[1],
                    "width": shape[2],
                    "echos": shape[3],
                    "coils": shape[4],
                }
            )
        return metadata

    def meddlr_pad(self, x: torch.Tensor, shape: Sequence[int], mode="constant", value=0) -> torch.Tensor:
        x_shape = x.shape[1 : 1 + len(shape)]
        assert all(
            x_shape[i] <= shape[i] or shape[i] is None for i in range(len(shape))
        ), f"Tensor spatial dimensions {x_shape} smaller than zero pad dimensions"

        total_padding = tuple(
            desired - current if desired is not None else 0 for current, desired in zip(x_shape, shape)
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
        fname, slice_ind, metadata = self.samples[idx]

        with h5py.File(fname, "r") as f:
            # Simulate 6x undersampled data
            # kspace (x, ky, kz, #echos, #coils)
            # image (x, ky, kz, #echos, #maps) - #maps = 1 for SKM-TEA
            # maps (x, ky, kz, #coils, #maps) - maps are the same for both echos

            kspace = torch.as_tensor(f["kspace"][[slice_ind], :, :, :, :])
            maps = torch.as_tensor(f["maps"][[slice_ind], :, :, :, :])
            mask = torch.as_tensor(f["masks/poisson_6.0x"][()]).unsqueeze(0)
            img_gt = torch.as_tensor(f["target"][[slice_ind], :, :, :, :])
        
        mask = self.meddlr_pad(mask, kspace.shape[1:3]) * 1.0

        kspace = kspace * mask.unsqueeze(-1).unsqueeze(-1).type(kspace.dtype)
        
        kspace = kspace[:, :, :, self.echo, :] # 1, y, z, n complex
        kspace = kspace.moveaxis(-1, 1) # 1, n, y, z complex
        kspace = self.from_torch_complex(kspace).squeeze(0) # 2, n, y, z real
        
        assert img_gt.shape[4] == 1
        img_gt = img_gt[:, :, :, self.echo, 0] # 1, y, z complex
        img_gt = self.from_torch_complex(img_gt).squeeze(0) # 2, y, z real

        assert maps.shape[4] == 1
        maps = maps[:, :, :, :, 0] # 1, y, z, n complex
        maps = maps.moveaxis(-1, 1).squeeze(0) # n, y, z complex

        return img_gt, kspace, {"mask": mask, "coil_maps": maps}