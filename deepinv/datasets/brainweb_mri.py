from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import resolve_root


class BrainwebMRI(ImageDataset):
    r"""BrainWeb MRI volumes.

    Thin PyTorch wrapper around `brainweb-dl
    <https://github.com/paquiteau/brainweb-dl>`_. Each sample is a native-resolution,
    channel-first 3D volume downloaded and cached on first access.

    :param str, pathlib.Path, None root: Dataset directory. Defaults to the
        DeepInverse cache.
    :param int, list[int] subject_ids: Subjects to expose.
        Defaults to subject 0.
    :param str contrast: MRI contrast. Defaults to ``"T1"``.
    :param collections.abc.Callable, None transform: Optional volume transform.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        subject_ids: int | list[int] = 0,
        contrast: Literal["T1", "T2", "T2*", "PD"] = "T1",
        transform: Callable | None = None,
    ) -> None:
        try:
            from brainweb_dl import get_mri
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "BrainwebMRI requires brainweb-dl. Install it with "
                "`pip install deepinv[dataset]`."
            ) from error

        self.root = resolve_root(root, "BrainwebMRI")
        self.subject_ids = (
            [subject_ids] if isinstance(subject_ids, int) else subject_ids
        )
        self.contrast = contrast
        self.transform = transform
        self.get_mri = get_mri

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int) -> torch.Tensor:
        volume = self.get_mri(
            sub_id=self.subject_ids[index],
            contrast=self.contrast,
            brainweb_dir=self.root,
        )
        volume = torch.from_numpy(np.asarray(volume, dtype=np.float32)).unsqueeze(0)
        return self.transform(volume) if self.transform is not None else volume
