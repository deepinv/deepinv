from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch

from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import resolve_root


class BrainWebMRI(ImageDataset):
    r"""Dataset for `BrainWeb <https://brainweb.bic.mni.mcgill.ca/>`_.

    BrainWeb brain phantom for Magnetic Resonance Imaging (MRI) research :footcite:p:`collinsDesignConstructionRealistic1998`.
    The dataset consists of 22 MRI brain phantom scans: 21 normal brains and 1 multiple sclerosis brain (patient 1).
    Several contrasts are available for each patient: T1, T2, T2* and PD.
    Each volume has shape (1, 181, 217, 181).

    :param int, collections.abc.Sequence[int], None subject_ids: Subjects to include in the dataset. Possible values: [0, 1, 4, 5, 6, 18, 20, 38, 41-54]. Defaults to `None` which includes all subjects.
    :param str contrast: MRI contrast to return: `"T1"`, `"T2"`, `"T2*"` or `"PD"`. Defaults to ``"T1"``.
    :param bool download: Download missing subjects. Defaults to `True`.
    :param collections.abc.Callable, None transform: Optional volume transform.
    :param str, pathlib.Path, None root: Root directory of dataset. Directory path from where we load and save the dataset.
    """

    def __init__(
        self,
        subject_ids: int | list[int] = 0,
        contrast: Literal["T1", "T2", "T2*", "PD"] = "T1",
        download: bool = True,
        root: str | Path | None = None,
        transform: Callable | None = None,
    ) -> None:
        try:
            from brainweb_dl import get_mri
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "BrainWebMRI requires brainweb-dl. Install it with `pip install deepinv[dataset]`."
            ) from error

        self.root = resolve_root(root, "BrainWebMRI")
        self.subject_ids = (
            [subject_ids] if isinstance(subject_ids, int) else subject_ids
        )
        self.contrast = contrast
        self.download = download
        self.transform = transform
        self.get_mri = get_mri

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        subject_id = self.subject_ids[index]
        if subject_id == 0 and self.contrast in ("T1", "T2", "PD"):
            filename = f"{self.contrast}_ICBM_normal_1mm_pn0_rf0.nii.gz"
        elif subject_id != 0 and self.contrast == "T1":
            filename = f"subject{subject_id:02d}_t1w.nii.gz"
        elif subject_id == 0:
            filename = "phantom_1.0mm_normal_fuzzy.nii.gz"
        else:
            filename = f"brainweb_s{subject_id:02d}_fuzzy.nii.gz"
        if not self.download and not (self.root / filename).exists():
            raise FileNotFoundError(
                f"BrainWeb MRI data for subject {subject_id} with "
                f"contrast {self.contrast!r} was not found at `{self.root / filename}`. "
                "Set `download=True` to download it."
            )
        volume = self.get_mri(
            sub_id=subject_id,
            contrast=self.contrast,
            brainweb_dir=self.root,
        )
        volume = torch.from_numpy(volume).to(torch.float32).unsqueeze(0) / 4095
        return self.transform(volume) if self.transform is not None else volume
