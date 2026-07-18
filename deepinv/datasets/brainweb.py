from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch

from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import download_archive, resolve_root


@dataclass(frozen=True)
class BrainWebLesion:
    r"""Configuration for a spherical lesion in a BrainWeb activity volume.

    A lesion with ``center_voxel=None`` is placed randomly in one of the tissues
    selected by :class:`BrainWebDataset`. Random placement is deterministic for a
    given dataset seed and subject. Supplying ``center_voxel`` places the lesion at
    that location instead.

    Lesions set an absolute activity value inside the sphere. Thus, both hot and
    cold lesions use the same representation: their activity is simply higher or
    lower than that of the surrounding tissue.

    :param float diameter_mm: Lesion diameter in millimetres.
    :param float activity: Absolute activity value inside the lesion. Must be
        non-negative.
    :param tuple[float, float, float], None center_voxel: Optional lesion centre in
        native BrainWeb voxel coordinates ``(z, y, x)``. If ``None``, a centre is
        sampled randomly.
    """

    diameter_mm: float
    activity: float
    center_voxel: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.diameter_mm <= 0:
            raise ValueError("diameter_mm must be strictly positive.")
        if self.activity < 0:
            raise ValueError("activity must be non-negative.")
        if self.center_voxel is not None and len(self.center_voxel) != 3:
            raise ValueError("center_voxel must contain exactly three coordinates.")


class BrainWebDataset(ImageDataset):
    r"""BrainWeb anatomical phantoms converted to PET activity volumes.

    The dataset downloads only the requested subjects from the `BrainWeb database
    <https://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html>`_ and
    caches only their compressed discrete anatomical models. Tissue labels are
    converted on demand to an activity volume and a PET attenuation map.

    Samples are returned as ``(activity, params)``, where ``activity`` has shape
    ``(1, 362, 434, 362)`` and ``params["attenuation"]`` has the same shape. This
    matches DeepInv's dataset convention for subject-dependent physics parameters
    and can be passed to :class:`deepinv.physics.PET`.

    The dataset preserves the native BrainWeb geometry: ``image_size`` is
    ``(362, 434, 362)`` and ``voxel_size`` is ``(0.5, 0.5, 0.5)`` mm. Scanner
    geometry and any deliberate spatial transform should be configured separately.

    Built-in activity presets are ``"fdg"`` and ``"amyloid"``. A custom mapping
    may instead assign absolute values to any of :attr:`tissues`; unspecified
    tissues receive zero activity. ``"grey_matter"`` is accepted as an alias for
    ``"gray_matter"``.

    The default attenuation map uses ``0.13 cm^-1`` for bone and ``0.0975 cm^-1``
    for soft tissue, matching the values used by ``casperdcl/brainweb``. Custom
    attenuation mappings may use tissue names or the groups ``"bone"`` and
    ``"soft_tissue"``.

    If lesions are configured, ``params`` also contains:

    - ``lesion_mask``: an integer label volume with zero outside lesions and
      ``i + 1`` inside lesion ``i``;
    - ``lesion_centers``: the resolved centres in ``(z, y, x)`` voxel coordinates.

    :param str, pathlib.Path, None root: Dataset directory. If ``None``, defaults
        to ``<deepinv cache>/datasets/BrainWeb``.
    :param int, sequence[int] subject_ids: Subject or subjects to expose. Only these
        subjects are downloaded. Default is subject 4.
    :param bool download: Download missing requested subjects. Default is ``False``.
    :param str, mapping activity_levels: ``"fdg"``, ``"amyloid"``, or a mapping
        from tissue names to absolute activity values.
    :param mapping, None attenuation_levels: Optional mapping from tissue names or
        tissue groups to linear attenuation coefficients in ``cm^-1``.
    :param sequence[BrainWebLesion], None lesions: Explicit and/or randomly placed
        lesions. A lesion is random when its ``center_voxel`` is ``None``.
    :param sequence[str] random_lesion_tissues: Tissue classes in which random
        lesion centres may be sampled.
    :param int, None seed: Base seed for random lesion placement. With an integer,
        each subject is reproducible independently of access order. With ``None``,
        placement is stochastic.
    :param bool return_lesion_mask: Include the lesion label mask and resolved
        centres in ``params`` when lesions are configured. Default is ``True``.

    |sep|

    :Example:

    >>> from deepinv.datasets import BrainWebDataset, BrainWebLesion
    >>> dataset = BrainWebDataset(
    ...     root="data/brainweb",
    ...     subject_ids=4,
    ...     download=True,
    ...     activity_levels="fdg",
    ...     lesions=[
    ...         BrainWebLesion(diameter_mm=10, activity=192),
    ...         BrainWebLesion(
    ...             diameter_mm=8, activity=0, center_voxel=(181, 217, 181)
    ...         ),
    ...     ],
    ...     seed=0,
    ... )
    >>> activity, params = dataset[0]
    >>> activity.shape == params["attenuation"].shape
    True
    >>> params["lesion_mask"].shape == activity.shape
    True
    """

    subjects = (4, 5, 6, 18, 20, 38, *range(41, 55))
    tissues = (
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
    image_size = (362, 434, 362)
    voxel_size = (0.5, 0.5, 0.5)

    activity_presets: ClassVar[dict[str, dict[str, float]]] = {
        "fdg": {"white_matter": 32.0, "gray_matter": 128.0, "skin": 16.0},
        "amyloid": {
            "white_matter": 29.0,
            "gray_matter": 66.0,
            "skin": 35.0,
        },
    }
    default_attenuation: ClassVar[dict[str, float]] = {
        "bone": 0.13,
        "soft_tissue": 0.0975,
    }

    _tissue_aliases: ClassVar[dict[str, str]] = {
        "grey_matter": "gray_matter",
        "around_fat": "connective_tissue",
        "connective": "connective_tissue",
        "tissue": "soft_tissue",
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
        lesions: Sequence[BrainWebLesion] | None = None,
        random_lesion_tissues: Sequence[str] = ("gray_matter", "white_matter"),
        seed: int | None = 0,
        return_lesion_mask: bool = True,
    ) -> None:
        self.root = resolve_root(root, "BrainWeb")
        self.subject_ids = self._validate_subject_ids(subject_ids)
        self.activity_levels = self._resolve_activity_levels(activity_levels)
        self.attenuation_levels = dict(
            self.default_attenuation
            if attenuation_levels is None
            else attenuation_levels
        )
        self._activity_lut = self._levels_to_lut(self.activity_levels)
        self._attenuation_lut = self._levels_to_lut(self.attenuation_levels)

        self.lesions = tuple(lesions or ())
        if len(self.lesions) > np.iinfo(np.uint8).max:
            raise ValueError("At most 255 lesions are supported.")
        if not all(isinstance(lesion, BrainWebLesion) for lesion in self.lesions):
            raise TypeError("lesions must contain only BrainWebLesion instances.")
        self.random_lesion_tissues = tuple(
            self._canonical_tissue_name(name) for name in random_lesion_tissues
        )
        if not self.random_lesion_tissues:
            raise ValueError("random_lesion_tissues must not be empty.")
        if any(name not in self.tissues for name in self.random_lesion_tissues):
            raise ValueError(
                "random_lesion_tissues must contain individual tissue names, not "
                "tissue groups."
            )
        self._random_lesion_labels = tuple(
            self.tissues.index(name) for name in self.random_lesion_tissues
        )
        self.seed = seed
        self.return_lesion_mask = return_lesion_mask

        self.root.mkdir(parents=True, exist_ok=True)
        missing = [
            subject_id
            for subject_id in self.subject_ids
            if not self._subject_path(subject_id).is_file()
        ]
        if missing and not download:
            missing_text = ", ".join(f"{subject_id:02d}" for subject_id in missing)
            raise RuntimeError(
                f"BrainWeb subject(s) {missing_text} not found in {self.root}. "
                "Set download=True to download the requested data."
            )
        for subject_id in missing:
            self._download_subject(subject_id)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int):
        subject_id = self.subject_ids[index]
        labels = self._load_labels(subject_id)
        activity = self._activity_lut[labels]
        attenuation = self._attenuation_lut[labels]

        params = {
            "attenuation": torch.from_numpy(attenuation).unsqueeze(0),
        }
        if self.lesions:
            activity, lesion_mask, lesion_centers = self._insert_lesions(
                activity, labels, subject_id
            )
            if self.return_lesion_mask:
                params["lesion_mask"] = torch.from_numpy(lesion_mask).unsqueeze(0)
                params["lesion_centers"] = torch.as_tensor(
                    lesion_centers, dtype=torch.float32
                )

        return torch.from_numpy(activity).unsqueeze(0), params

    @classmethod
    def _validate_subject_ids(cls, subject_ids: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(subject_ids, int):
            subject_ids = (subject_ids,)
        elif isinstance(subject_ids, (str, bytes)):
            raise TypeError("subject_ids must be an integer or a sequence of integers.")
        else:
            subject_ids = tuple(subject_ids)
        if not subject_ids:
            raise ValueError("subject_ids must not be empty.")
        if not all(isinstance(subject_id, int) for subject_id in subject_ids):
            raise TypeError("subject_ids must contain only integers.")
        unsupported = sorted(set(subject_ids).difference(cls.subjects))
        if unsupported:
            raise ValueError(
                f"Unsupported BrainWeb subject(s): {unsupported}. "
                f"Available subjects are {list(cls.subjects)}."
            )
        if len(set(subject_ids)) != len(subject_ids):
            raise ValueError("subject_ids must not contain duplicates.")
        return tuple(subject_ids)

    @classmethod
    def _canonical_tissue_name(cls, name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Tissue names must be strings.")
        name = name.strip().lower().replace("-", "_").replace(" ", "_")
        name = cls._tissue_aliases.get(name, name)
        if name not in cls.tissues and name not in cls._tissue_groups:
            valid = sorted((*cls.tissues, *cls._tissue_groups))
            raise ValueError(
                f"Unknown BrainWeb tissue {name!r}. Expected one of {valid}."
            )
        return name

    @classmethod
    def _resolve_activity_levels(
        cls, activity_levels: str | Mapping[str, float]
    ) -> dict[str, float]:
        if isinstance(activity_levels, str):
            preset = activity_levels.lower()
            if preset not in cls.activity_presets:
                raise ValueError(
                    f"Unknown activity preset {activity_levels!r}. "
                    f"Expected one of {sorted(cls.activity_presets)} or a mapping."
                )
            return dict(cls.activity_presets[preset])
        if not isinstance(activity_levels, Mapping):
            raise TypeError("activity_levels must be a preset name or a mapping.")
        return dict(activity_levels)

    @classmethod
    def _levels_to_lut(cls, levels: Mapping[str, float]) -> np.ndarray:
        lut = np.zeros(len(cls.tissues), dtype=np.float32)
        for name, value in levels.items():
            name = cls._canonical_tissue_name(name)
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"Value for tissue {name!r} must be numeric."
                ) from error
            if value < 0:
                raise ValueError(f"Value for tissue {name!r} must be non-negative.")
            names = cls._tissue_groups.get(name, (name,))
            for tissue_name in names:
                lut[cls.tissues.index(tissue_name)] = value
        return lut

    def _subject_path(self, subject_id: int) -> Path:
        return self.root / f"subject_{subject_id:02d}.raw_byte.bin.gz"

    def _download_subject(self, subject_id: int) -> None:
        download_archive(
            self._url_template.format(subject_id=subject_id),
            self._subject_path(subject_id),
        )

    def _load_labels(self, subject_id: int) -> np.ndarray:
        path = self._subject_path(subject_id)
        try:
            with gzip.open(path, "rb") as file:
                labels = np.frombuffer(file.read(), dtype=np.uint8)
        except (OSError, EOFError) as error:
            raise RuntimeError(f"Could not read BrainWeb data from {path}.") from error

        expected_size = int(np.prod(self.image_size))
        if labels.size != expected_size:
            raise RuntimeError(
                f"Invalid BrainWeb volume in {path}: got {labels.size} voxels, "
                f"expected {expected_size}."
            )
        labels = labels.reshape(self.image_size).copy()
        if labels.max(initial=0) >= len(self.tissues):
            raise RuntimeError(
                f"Invalid BrainWeb tissue label in {path}: found {labels.max()}, "
                f"expected labels between 0 and {len(self.tissues) - 1}."
            )
        return labels

    def _insert_lesions(
        self, activity: np.ndarray, labels: np.ndarray, subject_id: int
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]]]:
        lesion_mask = np.zeros(self.image_size, dtype=np.uint8)
        lesion_centers = []
        rng = self._subject_rng(subject_id)

        for lesion_index, lesion in enumerate(self.lesions, start=1):
            if lesion.center_voxel is None:
                center, slices, sphere = self._sample_lesion_location(
                    labels, lesion_mask, lesion, rng
                )
            else:
                center = tuple(float(value) for value in lesion.center_voxel)
                slices, sphere = self._sphere(center, lesion.diameter_mm)
                if np.any(lesion_mask[slices][sphere]):
                    raise ValueError(
                        f"Lesion {lesion_index - 1} overlaps an earlier lesion."
                    )

            activity_patch = activity[slices]
            activity_patch[sphere] = lesion.activity
            mask_patch = lesion_mask[slices]
            mask_patch[sphere] = lesion_index
            lesion_centers.append(center)

        return activity, lesion_mask, lesion_centers

    def _subject_rng(self, subject_id: int) -> np.random.Generator:
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(np.random.SeedSequence([self.seed, subject_id]))

    def _sample_lesion_location(
        self,
        labels: np.ndarray,
        lesion_mask: np.ndarray,
        lesion: BrainWebLesion,
        rng: np.random.Generator,
        max_attempts: int = 10_000,
    ) -> tuple[tuple[float, float, float], tuple[slice, slice, slice], np.ndarray]:
        radius_voxel = np.asarray(
            [lesion.diameter_mm / (2 * spacing) for spacing in self.voxel_size]
        )
        margin = np.ceil(radius_voxel).astype(int)
        lower = margin
        upper = np.asarray(self.image_size) - margin
        if np.any(lower >= upper):
            raise ValueError(
                f"Lesion diameter {lesion.diameter_mm} mm does not fit in the volume."
            )

        for _ in range(max_attempts):
            center_int = tuple(
                int(rng.integers(low, high))
                for low, high in zip(lower, upper, strict=True)
            )
            if labels[center_int] not in self._random_lesion_labels:
                continue
            center = tuple(float(value) for value in center_int)
            slices, sphere = self._sphere(center, lesion.diameter_mm)
            if not np.any(lesion_mask[slices][sphere]):
                return center, slices, sphere

        raise RuntimeError(
            f"Could not place a {lesion.diameter_mm} mm lesion after "
            f"{max_attempts} attempts in tissues {self.random_lesion_tissues}."
        )

    def _sphere(
        self, center: tuple[float, float, float], diameter_mm: float
    ) -> tuple[tuple[slice, slice, slice], np.ndarray]:
        center_array = np.asarray(center, dtype=float)
        radius_mm = diameter_mm / 2
        radius_voxel = np.asarray(
            [radius_mm / spacing for spacing in self.voxel_size], dtype=float
        )
        if np.any(center_array - radius_voxel < 0) or np.any(
            center_array + radius_voxel > np.asarray(self.image_size) - 1
        ):
            raise ValueError(
                f"Lesion centred at {center} with diameter {diameter_mm} mm "
                f"does not fit inside volume shape {self.image_size}."
            )

        starts = np.floor(center_array - radius_voxel).astype(int)
        stops = np.ceil(center_array + radius_voxel).astype(int) + 1
        slices = tuple(
            slice(int(start), int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )

        distance = 0.0
        ndim = len(self.image_size)
        for axis, (start, stop, centre, spacing) in enumerate(
            zip(starts, stops, center_array, self.voxel_size, strict=True)
        ):
            coordinate = (np.arange(start, stop) - centre) * spacing / radius_mm
            shape = [1] * ndim
            shape[axis] = coordinate.size
            distance = distance + coordinate.reshape(shape) ** 2
        return slices, distance <= 1
