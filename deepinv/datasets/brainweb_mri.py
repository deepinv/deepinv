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
    Each T1 volume has shape (1, 181, 256, 256) and is scaled by 1 / 4095 to that it is normalized with values in [0, 1].

    This dataset relies on the original implementation of Pierre-Antoine Comby:
    <https://github.com/paquiteau/brainweb-dl>`_. Install it with `pip install brainweb-dl`.

    .. note::
        For a version of this dataset with dedicated features for emission tomography, such
        as emission / attenuation maps and hot lesions, see :class:`deepinv.datasets.BrainWebPET`.

    :param int, collections.abc.Sequence[int], None subject_ids: Subjects to include in the dataset. Possible values: [4, 5, 6, 18, 20, 38, 41-54]. Defaults to all subjects.
    :param str contrast: MRI contrast to return: `"T1"`, `"T2"`, `"T2*"` or `"PD"`. Defaults to ``"T1"``.
    :param bool download: Download missing subjects. Defaults to `True`.
    :param collections.abc.Callable, None transform: Optional volume transform.
    :param str, pathlib.Path, None root: Root directory of dataset. Directory path from where we load and save the dataset.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).
    """

    def __init__(
        self,
        subject_ids: int | list[int] | None = None,
        contrast: Literal["T1", "T2", "T2*", "PD"] = "T1",
        download: bool = True,
        root: str | Path | None = None,
        transform: Callable | None = None,
        use_dict_output: bool = False,
    ) -> None:
        super().__init__(use_dict_output=use_dict_output)
        try:
            from brainweb_dl import get_mri
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "BrainWebMRI requires brainweb-dl. Install it with `pip install deepinv[dataset]`."
            ) from error

        self.root = resolve_root(root, "BrainWebMRI")
        if subject_ids is None:
            subject_ids = [4, 5, 6, 18, 20, 38, *range(41, 55)]
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
        if self.contrast == "T1":
            filename = f"subject{subject_id:02d}_t1w.nii.gz"
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
        volume = self.transform(volume) if self.transform is not None else volume

        return {"x": volume} if self.use_dict_output else volume
