from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
from numbers import Real
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import download_archive, resolve_root


@dataclass(frozen=True)
class BrainWebLesion:
    r"""Configuration for a circular or spherical lesion.

    :param float diameter_mm: Lesion diameter in millimetres.
    :param float activity: Activity inside the lesion.
    :param tuple[float, ...], None center_voxel: Optional centre in voxel coordinates.
    """

    diameter_mm: float
    activity: float
    center_voxel: tuple[float, ...] | None = None


class BrainWebDataset(ImageDataset):
    r"""BrainWeb anatomical phantoms for PET.

    Requested subjects are downloaded from the `BrainWeb database
    <https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html>`_. Samples
    contain an activity image and an attenuation map that can be passed to
    :class:`deepinv.physics.PET`. Set ``slice_index`` for a 2D axial slice, or leave
    it as ``None`` for the full 3D volume.

    Custom activity levels can be given for each tissue. The ``"fdg"`` and
    ``"amyloid"`` presets use values from `casperdcl/brainweb
    <https://github.com/casperdcl/brainweb>`_. Custom attenuation levels can use
    tissue names or the groups ``"bone"`` and ``"soft_tissue"``; the defaults
    follow the same reference.

    Lesions with no centre are placed randomly in ``random_lesion_tissues``.
    Controlled lesions use ``(z, y, x)`` centres in 3D and ``(y, x)`` centres in
    2D. When requested, ``params`` also contains ``lesion_mask`` and
    ``lesion_centers``.

    :param str, pathlib.Path, None root: Dataset directory. Defaults to the DeepInv
        cache.
    :param int, sequence[int] subject_ids: Subjects to expose. Default is subject 4.
    :param bool download: Download missing subjects. Default is ``False``.
    :param str, mapping activity_levels: Tissue activities or a preset name.
    :param mapping, None attenuation_levels: Tissue attenuation coefficients in
        ``cm^-1``.
    :param int, None slice_index: Axial slice returned as a 2D image. By default,
        the full 3D volume is returned.
    :param sequence[BrainWebLesion], None lesions: Lesions to insert.
    :param sequence[str] random_lesion_tissues: Tissues used for random centres.
    :param int, None seed: Seed for random lesion placement.
    :param bool return_lesion_mask: Return lesion masks and centres. Default is
        ``True``.

    |sep|

    :Example:

    >>> from deepinv.datasets import BrainWebDataset, BrainWebLesion
    >>> dataset = BrainWebDataset(
    ...     root="data/brainweb",
    ...     download=True,
    ...     slice_index=181,
    ...     lesions=[BrainWebLesion(diameter_mm=10, activity=192)],
    ... )
    >>> activity, params = dataset[0]
    >>> activity.shape == params["attenuation"].shape
    True
    """

    subjects: ClassVar[tuple[int, ...]] = (4, 5, 6, 18, 20, 38, *range(41, 55))
    tissues: ClassVar[tuple[str, ...]] = (
        "background",
        "csf",
        "gray_matter",
        "white_matter",
        "fat",
        "muscle",
        "skin",
        "skull",
        "vessels",
        "connective_tissue",
        "dura",
        "marrow",
    )
    image_size: ClassVar[tuple[int, int, int]] = (362, 434, 362)
    voxel_size: ClassVar[tuple[float, float, float]] = (0.5, 0.5, 0.5)
    activity_presets: ClassVar[dict[str, dict[str, float]]] = {
        "fdg": {"white_matter": 32.0, "gray_matter": 128.0, "skin": 16.0},
        "amyloid": {"white_matter": 29.0, "gray_matter": 66.0, "skin": 35.0},
    }
    default_attenuation: ClassVar[dict[str, float]] = {
        "bone": 0.13,
        "soft_tissue": 0.0975,
    }
    _tissue_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "bone": ("skull", "dura", "marrow"),
        "soft_tissue": (
            "csf",
            "gray_matter",
            "white_matter",
            "fat",
            "muscle",
            "skin",
            "vessels",
            "connective_tissue",
        ),
    }
    _url_template = (
        "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1"
        "?do_download_alias=subject{subject_id:02d}_crisp"
        "&format_value=raw_byte&zip_value=gnuzip"
        "&download_for_real=%5BStart+download%21%5D"
    )

    def __init__(
        self,
        root: str | Path | None = None,
        subject_ids: int | Sequence[int] = 4,
        download: bool = False,
        activity_levels: str | Mapping[str, float] = "fdg",
        attenuation_levels: Mapping[str, float] | None = None,
        slice_index: int | None = None,
        lesions: Sequence[BrainWebLesion] | None = None,
        random_lesion_tissues: Sequence[str] = ("gray_matter", "white_matter"),
        seed: int | None = 0,
        return_lesion_mask: bool = True,
    ) -> None:
        self.root = resolve_root(root, "BrainWeb")
        if isinstance(subject_ids, int):
            subject_ids = (subject_ids,)
        elif isinstance(subject_ids, Sequence) and not isinstance(subject_ids, str):
            subject_ids = tuple(subject_ids)
        else:
            raise ValueError(
                f"Incorrect subject_ids. Available values are {self.subjects}."
            )
        if (
            not subject_ids
            or any(subject not in self.subjects for subject in subject_ids)
            or len(set(subject_ids)) != len(subject_ids)
        ):
            raise ValueError(
                f"Incorrect subject_ids. Available values are {self.subjects}."
            )
        self.subject_ids = subject_ids

        self._volume_size = self.image_size
        if slice_index is not None and (
            not isinstance(slice_index, int)
            or not 0 <= slice_index < self._volume_size[0]
        ):
            raise ValueError(
                f"Incorrect slice_index. Available values are 0 to "
                f"{self._volume_size[0] - 1}."
            )
        self.slice_index = slice_index
        if slice_index is not None:
            self.image_size = self.image_size[1:]
            self.voxel_size = self.voxel_size[1:]

        if isinstance(activity_levels, str):
            if activity_levels not in self.activity_presets:
                raise ValueError(
                    f"Incorrect activity_levels. Available presets are "
                    f"{tuple(self.activity_presets)}."
                )
            activity_levels = self.activity_presets[activity_levels]
        elif not isinstance(activity_levels, Mapping):
            raise ValueError("Incorrect activity_levels. Expected a preset or mapping.")
        self.activity_levels = dict(activity_levels)
        self._activity_lut = np.zeros(len(self.tissues), dtype=np.float32)
        for name, value in self.activity_levels.items():
            if name not in self.tissues or not isinstance(value, Real) or value < 0:
                raise ValueError(
                    f"Incorrect activity_levels. Available tissues are {self.tissues}."
                )
            self._activity_lut[self.tissues.index(name)] = value

        self.attenuation_levels = dict(
            self.default_attenuation
            if attenuation_levels is None
            else attenuation_levels
        )
        self._attenuation_lut = np.zeros(len(self.tissues), dtype=np.float32)
        valid_attenuation_names = (*self.tissues, *self._tissue_groups)
        for name, value in self.attenuation_levels.items():
            if (
                name not in valid_attenuation_names
                or not isinstance(value, Real)
                or value < 0
            ):
                raise ValueError(
                    f"Incorrect attenuation_levels. Available names are "
                    f"{valid_attenuation_names}."
                )
            for tissue in self._tissue_groups.get(name, (name,)):
                self._attenuation_lut[self.tissues.index(tissue)] = value

        self.lesions = tuple(lesions or ())
        if (
            len(self.lesions) > np.iinfo(np.uint8).max
            or any(not isinstance(lesion, BrainWebLesion) for lesion in self.lesions)
            or any(
                lesion.diameter_mm <= 0
                or lesion.activity < 0
                or (
                    lesion.center_voxel is not None
                    and len(lesion.center_voxel) != len(self.image_size)
                )
                for lesion in self.lesions
            )
        ):
            raise ValueError("Incorrect lesions.")
        self.random_lesion_tissues = tuple(random_lesion_tissues)
        if not self.random_lesion_tissues or any(
            name not in self.tissues for name in self.random_lesion_tissues
        ):
            raise ValueError(
                f"Incorrect random_lesion_tissues. Available values are {self.tissues}."
            )
        self._random_lesion_labels = tuple(
            self.tissues.index(name) for name in self.random_lesion_tissues
        )
        self.seed = seed
        self.return_lesion_mask = return_lesion_mask

        self.root.mkdir(parents=True, exist_ok=True)
        missing = [
            subject
            for subject in self.subject_ids
            if not self._subject_path(subject).is_file()
        ]
        if missing and not download:
            raise RuntimeError(
                f"BrainWeb subjects {missing} not found in {self.root}. Set download=True."
            )
        for subject in missing:
            download_archive(
                self._url_template.format(subject_id=subject),
                self._subject_path(subject),
            )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int):
        subject = self.subject_ids[index]
        labels = self._load_labels(subject)
        activity = self._activity_lut[labels]
        params = {
            "attenuation": torch.from_numpy(self._attenuation_lut[labels]).unsqueeze(0)
        }

        if self.lesions:
            activity, lesion_mask, lesion_centers = self._insert_lesions(
                activity, labels, subject
            )
            if self.return_lesion_mask:
                params["lesion_mask"] = torch.from_numpy(lesion_mask).unsqueeze(0)
                params["lesion_centers"] = torch.tensor(
                    lesion_centers, dtype=torch.float32
                )
        return torch.from_numpy(activity).unsqueeze(0), params

    def _subject_path(self, subject: int) -> Path:
        return self.root / f"subject_{subject:02d}.raw_byte.bin.gz"

    def _load_labels(self, subject: int) -> np.ndarray:
        path = self._subject_path(subject)
        try:
            with gzip.open(path, "rb") as file:
                labels = np.frombuffer(file.read(), dtype=np.uint8)
        except (OSError, EOFError) as error:
            raise RuntimeError(f"Could not read BrainWeb data from {path}.") from error

        expected_size = int(np.prod(self._volume_size))
        if labels.size != expected_size:
            raise RuntimeError(
                f"Invalid BrainWeb volume in {path}: got {labels.size} voxels, "
                f"expected {expected_size}."
            )
        labels = labels.reshape(self._volume_size)
        if labels.max(initial=0) >= len(self.tissues):
            raise RuntimeError(f"Invalid BrainWeb tissue label in {path}.")
        if self.slice_index is not None:
            labels = labels[self.slice_index]
        return labels.copy()

    def _insert_lesions(
        self, activity: np.ndarray, labels: np.ndarray, subject: int
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, ...]]]:
        lesion_mask = np.zeros(self.image_size, dtype=np.uint8)
        centers = []
        rng = torch.Generator()
        rng.seed() if self.seed is None else rng.manual_seed(self.seed + subject)

        for index, lesion in enumerate(self.lesions, start=1):
            if lesion.center_voxel is None:
                center, slices, shape = self._sample_lesion_location(
                    labels, lesion_mask, lesion, rng
                )
            else:
                center = tuple(float(value) for value in lesion.center_voxel)
                slices, shape = self._lesion_shape(center, lesion.diameter_mm)
                if np.any(lesion_mask[slices][shape]):
                    raise ValueError(f"Lesion {index - 1} overlaps an earlier lesion.")

            activity[slices][shape] = lesion.activity
            lesion_mask[slices][shape] = index
            centers.append(center)
        return activity, lesion_mask, centers

    def _sample_lesion_location(
        self,
        labels: np.ndarray,
        lesion_mask: np.ndarray,
        lesion: BrainWebLesion,
        rng: torch.Generator,
    ) -> tuple[tuple[float, ...], tuple[slice, ...], np.ndarray]:
        radius = np.asarray(
            [lesion.diameter_mm / (2 * spacing) for spacing in self.voxel_size]
        )
        margin = np.ceil(radius).astype(int)
        upper = np.asarray(self.image_size) - margin
        if np.any(margin >= upper):
            raise ValueError(f"Lesion diameter {lesion.diameter_mm} mm does not fit.")

        for _ in range(10_000):
            center_int = tuple(
                int(torch.randint(int(low), int(high), (), generator=rng))
                for low, high in zip(margin, upper, strict=True)
            )
            if labels[center_int] not in self._random_lesion_labels:
                continue
            center = tuple(float(value) for value in center_int)
            slices, shape = self._lesion_shape(center, lesion.diameter_mm)
            if not np.any(lesion_mask[slices][shape]):
                return center, slices, shape
        raise RuntimeError(f"Could not place lesion in {self.random_lesion_tissues}.")

    def _lesion_shape(
        self, center: tuple[float, ...], diameter_mm: float
    ) -> tuple[tuple[slice, ...], np.ndarray]:
        center = np.asarray(center, dtype=float)
        radius_mm = diameter_mm / 2
        radius = radius_mm / np.asarray(self.voxel_size)
        if np.any(center - radius < 0) or np.any(
            center + radius > np.asarray(self.image_size) - 1
        ):
            raise ValueError(f"Lesion at {tuple(center)} does not fit in the image.")

        starts = np.floor(center - radius).astype(int)
        stops = np.ceil(center + radius).astype(int) + 1
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )
        distance = 0.0
        for axis, (start, stop, coordinate, spacing) in enumerate(
            zip(starts, stops, center, self.voxel_size, strict=True)
        ):
            values = (np.arange(start, stop) - coordinate) * spacing / radius_mm
            shape = [1] * len(self.image_size)
            shape[axis] = values.size
            distance = distance + values.reshape(shape) ** 2
        return slices, distance <= 1
