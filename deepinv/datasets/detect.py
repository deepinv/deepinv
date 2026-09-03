from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from deepinv.utils.io import load_tiff

class DeteCTDataset(Dataset):
    """2DeteCT dataset.

    GT x = iterative recon of shape (1,1024,1024) from mode2 data.
    y = real projection data, preprocessed (flat/dark-corrected, log-transformed) of shape (1,n_angles,956).
    The dataset follows the same preprocessing steps as in LION https://github.com/CambridgeCIA/LION/blob/main/LION/data_loaders/2deteCT,
    but adapted to `torch`.

    :param root: root dir, should contain subfolders named `2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`)
    :param str problem: benchmarking problem from 2DeteCT + https://www.aimsciences.org/article/doi/10.3934/ammc.2025001
      - full: mode2 acquired data (3600 projections)
      - sparse_view: mode2 acquired data then evenly subsampled
      - limited_angle: mode2 acquired data then limited angles taken
      - low_dose: mode1 acquired data
      - beam_hardening: mode3 acquired data
    :param int n_angles: kept projections for sparse_view/limited_angle.
    :param str slice_ids: `all` (default, every slice found from 1-5000) or `train`/`val`/`test` (LION 3930/550/470 sample split).
        NOTE: sequential, such that test set only requires downloading archive for slices 4001-5000.

    TODO add download parameter/download instructions.
    """
    def __init__(self, root: str | Path, problem: str = "full", n_angles: int = 3600, slice_ids: int = "all"):
        self.root = Path(root)
        self.problem, self.n_angles = problem, n_angles
        self.mode = {"low_dose": "mode1", "beam_hardening": "mode3"}.get(problem, "mode2")

        lo, hi = {"all": (1, 5000), "train": (1, 3930), "val": (3931, 4480), "test": (4531, 5000)}[slice_ids]
        self.slices = sorted(int(p.name[5:]) for p in self.root.glob("2DeteCT_slices*[0-9]/slice[0-9]*") if p.is_dir() and lo <= int(p.name[5:]) <= hi)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        slice_num = self.slices[i]
        block_start = (slice_num - 1) // 1000 * 1000 + 1          # 1, 1001, ..., 4001
        block = f"2DeteCT_slices{block_start}-{block_start + 999}"
        stem = f"slice{slice_num:05d}"

        data_dir = self.root / block / stem / self.mode

        sino = load_tiff(data_dir / "sinogram.tif")[:, :, :-1] # (1, 1, 3600, 1912)
        dark = load_tiff(data_dir / "dark.tif") # (1, 1, 1, 1912)
        flat = 0.5 * (load_tiff(data_dir / "flat1.tif") + load_tiff(data_dir / "flat2.tif"))

        if slice_num < 2830 or 5520 < slice_num < 5871:
            def detector_shift(a):
                out = torch.empty_like(a)
                out[..., :-1] = a[..., 1:]
                out[..., -1] = 2 * a[..., -1] - a[..., -2]
                return out

            sino, flat, dark = detector_shift(sino), detector_shift(flat), detector_shift(dark)

        # Bin detector pixels
        sino = sino[..., 0::2] + sino[..., 1::2] # (1, 1, 3600, 956)
        dark = dark[..., 0::2] + dark[..., 1::2] # (1, 1, 1, 956)
        flat = flat[..., 0::2] + flat[..., 1::2]

        # Detector corrections:
        sino = (sino - dark) / (flat - dark) # flat/dark-field correction
        sino = -sino.clip(min=1e-6).log() # Beer-Lambert
        sino = sino.flip(dims=(-1,))# flip detector

        if self.problem == "sparse_view":
            sino = sino[:, :, ::3600 // self.n_angles] # (1, 1, n_angles, 956)
        elif self.problem == "limited_angle":
            sino = sino[:, :, :self.n_angles] # (1, 1, n_angles, 956)

        x = load_tiff(self.root / (block + "_RecSeg") / stem / 'mode2' / "reconstruction.tif")

        return x.squeeze(0).contiguous().float(), sino.squeeze(0).contiguous().float()