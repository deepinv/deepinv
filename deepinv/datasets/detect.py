from __future__ import annotations
from pathlib import Path
import torch
from deepinv.datasets.base import ImageDataset
from deepinv.utils.io import load_tiff


class DeteCTDataset(ImageDataset):
    """2DeteCT dataset of 2D Computed Tomography acquisitions.

    The dataset was acquired by :footcite:t:`kiss20232detect` and used for benchmarking CT reconstruction algorithms in :footcite:t:`kiss2025benchmarking`.
    The data is industrial CT projection data (i.e. sinograms) of various materials acquired using a proprietary scanner from CWI.

    The projections (shape `(1,n_angles,956)`) are preprocessed (flat/dark-corrected, log-transformed, all in PyTorch) following `LION <https://github.com/CambridgeCIA/LION>`_
    such that the setup matches exactly :footcite:t:`kiss2025benchmarking`, such that the dataset can be used to compare DeepInverse image reconstruction methods
    with the values reported in :footcite:t:`kiss2025benchmarking`.

    Each sample is scanned 3 times: `mode1`, `mode2` and `mode3`. See below for their usage.

    "Ground truth" `x` are also provided as iterative recons using all angles, of shape `(1,1024,1024)`.

    To download: TODO download instructions
    Note for test set, you only need to download slices 4001-5000 from `Zenodo <https://zenodo.org/records/8014874>`_.


    :param root: root dir, should contain subfolders named `2DeteCT_slicesXXXX-YYYY` (+ `_RecSeg`)
    :param str problem: benchmarking problem from 2DeteCT.
      - `full`: `mode2` acquired data (3600 projections)
      - `sparse_view`: `mode2` acquired data then evenly subsampled
      - `limited_angle`: `mode2` acquired data then limited angles taken
      - `low_dose`: `mode1` acquired data (3W instead of 90W)
      - `beam_hardening`: `mode3` acquired data (acquired without a filter, leading to beam-hardening)

    :param int n_angles: kept projections for sparse_view/limited_angle, defaults to 3600 (i.e. all angles).
    :param str slice_ids: `all` (default, every slice found from 1-5000) or `train`/`val`/`test` (LION 3930/550/470 sample split).
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    Example:

    TODO using sample slide + recon on HF
    """

    def __init__(
        self,
        root: str | Path,
        problem: str = "full",
        n_angles: int = 3600,
        slice_ids: int = "all",
        use_dict_output: bool = False,
    ):
        super().__init__(use_dict_output=use_dict_output)
        self.root = Path(root)
        self.problem, self.n_angles = problem, n_angles
        self.mode = {"low_dose": "mode1", "beam_hardening": "mode3"}.get(
            problem, "mode2"
        )

        lo, hi = {
            "all": (1, 5000),
            "train": (1, 3930),
            "val": (3931, 4480),
            "test": (4531, 5000),
        }[slice_ids]

        self.slices = sorted(
            int(p.name[5:])
            for p in self.root.glob("2DeteCT_slices*[0-9]/slice[0-9]*")
            if p.is_dir() and lo <= int(p.name[5:]) <= hi
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, i):
        slice_num = self.slices[i]
        block_start = (slice_num - 1) // 1000 * 1000 + 1  # e.g. 1, 1001, ..., 4001
        block = f"2DeteCT_slices{block_start}-{block_start + 999}"
        stem = f"slice{slice_num:05d}"

        data_dir = self.root / block / stem / self.mode

        sino = load_tiff(data_dir / "sinogram.tif")[:, :, :-1]  # (1, 1, 3600, 1912)
        dark = load_tiff(data_dir / "dark.tif")  # (1, 1, 1, 1912)
        flat = 0.5 * (
            load_tiff(data_dir / "flat1.tif") + load_tiff(data_dir / "flat2.tif")
        )

        if slice_num < 2830 or 5520 < slice_num < 5871:

            def detector_shift(a):
                out = torch.empty_like(a)
                out[..., :-1] = a[..., 1:]
                out[..., -1] = 2 * a[..., -1] - a[..., -2]
                return out

            sino, flat, dark = (
                detector_shift(sino),
                detector_shift(flat),
                detector_shift(dark),
            )

        # Bin detector pixels
        sino = sino[..., 0::2] + sino[..., 1::2]  # (1, 1, 3600, 956)
        dark = dark[..., 0::2] + dark[..., 1::2]  # (1, 1, 1, 956)
        flat = flat[..., 0::2] + flat[..., 1::2]

        # Detector corrections:
        sino = (sino - dark) / (flat - dark)  # flat/dark-field correction
        sino = -sino.clip(min=1e-6).log()  # Beer-Lambert
        sino = sino.flip(dims=(-1,))  # flip detector

        if self.problem == "sparse_view":
            sino = sino[:, :, :: 3600 // self.n_angles]  # (1, 1, n_angles, 956)
        elif self.problem == "limited_angle":
            sino = sino[:, :, : self.n_angles]  # (1, 1, n_angles, 956)

        y = sino.squeeze(0).contiguous().float()

        x = (
            load_tiff(
                self.root / (block + "_RecSeg") / stem / "mode2" / "reconstruction.tif"
            )
            .squeeze(0)
            .contiguous()
            .float()
        )

        return {"x": x, "y": y} if self.use_dict_output else (x, y)
