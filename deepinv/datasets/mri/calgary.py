from deepinv.datasets.base import ImageDataset
try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = ImportError(
        "The h5py package is not installed. Please install it with `pip install h5py`."
    )  # pragma: no cover
import torch
import numpy as np
from typing import Union, Optional, Callable
from pathlib import Path
import os
from deepinv.utils.demo import get_image_url
from deepinv.datasets.utils import download_archive, torch_shuffle


def _load_calgary_volume(
    filename, img_size=(256, 218, 170), partial_fourier_factor=0.85
):
    """Load a Calgary brain MRI volume from a .h5 file."""
    with h5py.File(filename, "r") as h5obj:
        kspace_hybrid = h5obj["kspace"][:]
    # Explicit zero-filling after 85% in the slice-encoded direction
    Nz = kspace_hybrid.shape[2]
    Nz_sampled = int(np.ceil(Nz * partial_fourier_factor))
    kspace_hybrid[:, :, Nz_sampled:, :] = 0
    # Move coils dimension to front
    kspace_hybrid = torch.tensor(
        kspace_hybrid[:, :, :, ::2] + 1j * kspace_hybrid[:, :, :, 1::2],
        dtype=torch.complex64,
    )
    # from x,ky,kz to x,y,z
    images = torch.fft.ifft2(
        torch.fft.ifftshift(kspace_hybrid, axis=[1, 2]), axis=[1, 2]
    )
    # Crop around center
    if images.shape[-2] != img_size[-1]:
        D = (images.shape[-2] - img_size[-1]) // 2
        images = images[:, :, D:-D, :]
    images = images.permute(3, 0, 1, 2) 
    return images


class Calgary3DBrainMRIDataset(ImageDataset):
    """Dataset for Calgary 3D Brain MRI"""

    def __init__(
        self,
        root: Union[str, Path],
        target_root: Optional[Union[str, Path]] = None,
        img_size: tuple[int, int, int] = (256, 218, 170),
        subsample_volumes: Optional[float] = 1.0,
        transform: Optional[Callable] = None,
        filter_id: Optional[Callable] = None,
        download_example: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = (
            transform if transform is not None else CalgaryDataTransformer()
        )
        self.target_root = Path(target_root) if target_root is not None else None
        self.img_size = img_size
        if download_example:
            os.makedirs(root, exist_ok=True)
            file_name = "calgary_brain_12channel.h5"
            if not os.path.isfile(self.root / file_name):
                url = get_image_url(str(file_name))
                download_archive(url, self.root / file_name)
            
        if isinstance(h5py, ImportError):
            raise h5py

        if not os.path.isdir(root):
            raise ValueError(
                f"The `root` folder doesn't exist. Please set `root` properly. Current value `{root}`."
            )

        # Load all slices
        all_fnames = sorted(list(Path(root).glob("*.h5")))

        # Randomly keep a portion of MRI volumes
        if subsample_volumes < 1.0:
            subsampled_fnames = torch_shuffle(all_fnames, generator=rng)[
                : round(len(all_fnames) * subsample_volumes)
            ]
        self.fnames = subsampled_fnames if subsample_volumes < 1.0 else all_fnames

        if filter_id is not None:
            self.fnames = list(filter(filter_id, self.fnames))

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fname = self.fnames[idx]
        volume = _load_calgary_volume(
            fname, img_size=self.img_size, partial_fourier_factor=0.85
        )
        if self.target_root is not None:
            with h5py.File(self.target_root / fname, "r") as hf:
                target = torch.from_numpy(
                    hf[
                        (
                            "reconstruction"
                            if "reconstruction" in f.keys()
                            else "reconstruction_rss"
                        )
                    ]
                )
        else:
            target = None
        if self.transform is not None:
            return self.transform(volume, target)
        return volume, target


class CalgaryDataTransformer:
    def __init__(self, foward_model=None, noise_level=0.0, scale_factor=1e-5, seed=0):
        self.forward_model = foward_model
        self.noise_level = noise_level
        self.seed = seed
        self.scale_factor = scale_factor

    def __call__(self, volumes, target=None):
        data = volumes
        if target is None:
            target = torch.linalg.norm(data, dim=0, keepdim=True)
        if self.scale_factor != 1:
            data = data * self.scale_factor
            target = target * self.scale_factor
        if self.forward_model is not None:
            # Sample k-space data
            data = self.forward_model.A(data)
        if self.noise_level > 0:
            torch.manual_seed(self.seed)
            noise = (
                (torch.randn_like(data) + 1j * torch.randn_like(data))
                / np.sqrt(2)
                * self.noise_level
            )
            data = data + noise
        return data, target