from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from deepinv.datasets.base import ImageDataset
from deepinv.datasets.utils import resolve_root

if TYPE_CHECKING:
    import brainweb


class BrainWebPET(ImageDataset):
    r"""BrainWeb PET phantoms.

    Loads synthetic 3D volumes from BrainWeb dataset :footcite:p:`collinsDesignConstructionRealistic1998`,
    of shape `(1, 127, 344, 344)`.
    The dataset has been adapted to emission tomography, and returns an emission and attenuation map, at the Siemens Biograph mMR isotropic resolution of 2.0863 mm per voxel.

    Passing `lesion_diameters` adds high activity lesions with `brainweb.add_lesions` and includes a `lesion_mask` in the returned params, where the background is labelled ``0`` and lesions are labelled from ``1`` onwards.

    This dataset relies on the original implementation of Casper da Costa-Luis:
    <https://github.com/casperdcl/brainweb>`_. Install it with `pip install brainweb`.
    See the original implementation for a detailed description of the keyword arguments.

    .. note::
        For a version of this dataset dedicated to magnetic resonance imaging, which contains
        more contrast options, see :class:`deepinv.datasets.BrainWebMRI`.

    :param str, pathlib.Path, None root: Dataset directory. Defaults to the DeepInv cache.
    :param int, collections.abc.Sequence[int], None subject_ids: Subjects to include in the dataset. Defaults to `None` which includes all subjects.
    :param bool download: Download missing subjects. Defaults to `True`.
    :param type[brainweb.Act], None pet_class: BrainWeb PET activity preset. Defaults to `brainweb.FDG`.
    :param list[float], None lesion_diameters: Lesion diameters in mm. Defaults to `None`, which adds no lesions.
    :param str, collections.abc.Sequence[str] contrast: Contrasts to include in the returned parameters. Valid values are `"T1"` and `"T2"`. Defaults to an empty tuple.
    :param dict, None lesion_kwargs: Keyword arguments for `brainweb.add_lesions`.
    :param dict, None random_degradations_kwargs: Keyword arguments for `brainweb.get_mmr_fromfile` controlling random structural degradations.
    :param collections.abc.Callable, None transform: Optional transform to apply to the returned volumes.
    :param int, None seed: Seed used when adding random lesions.
    :param bool use_dict_output: whether to return output as dict with keys "x", "y", "params" instead of tuple (default `False`).

    |sep|

    :Example:

    >>> import brainweb
    >>> import numpy as np
    >>> from deepinv.datasets import BrainWebPET
    >>> class RandomFDG(brainweb.FDG):
    ...     greyMatter = lambda: np.random.normal(128, 8)
    >>> dataset = BrainWebPET(
    ...     root="data/brainweb_pet",
    ...     random_degradations_kwargs={"petNoise": 0.5, "petSigma": 2},
    ...     contrast=["T1", "T2"],
    ...     pet_class=RandomFDG,
    ...     lesion_diameters=[15, 7],
    ...     lesion_kwargs={"intensity": [200, 150], "blur": [0, 0], "thresh": 30},
    ... )
    >>> emission, params = dataset[0]
    >>> emission.shape == params["attenuation"].shape
    True
    """

    def __init__(
        self,
        root: str | Path | None = None,
        subject_ids: int | Sequence[int] | None = None,
        download: bool = True,
        transform: Callable | None = None,
        pet_class: type[brainweb.Act] | None = None,
        contrast: str | Sequence[str] = (),
        random_degradations_kwargs: dict[str, object] | None = None,
        lesion_diameters: list[float] | None = None,
        lesion_kwargs: dict[str, object] | None = None,
        seed: int | None = 0,
        use_dict_output: bool = False,
    ) -> None:
        super().__init__(use_dict_output=use_dict_output)
        try:
            import brainweb
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "BrainWebPET requires brainweb. Install it with `pip install brainweb`."
            ) from error

        pet_class = brainweb.FDG if pet_class is None else pet_class
        activities = {}
        for name in (*pet_class.attrs, "hot", "cold"):
            value = getattr(pet_class, name, None)
            if callable(value):
                activities[name] = value()
        if activities:
            pet_class = type(pet_class.__name__, (pet_class,), activities)

        self.root = resolve_root(root, "BrainWebPET")
        self.transform = transform
        contrast = (
            (contrast,)
            if isinstance(contrast, str)
            else tuple(contrast) if contrast is not None else ()
        )
        self.contrast = tuple(name.upper() for name in contrast)
        available = {
            int(name[8:10]): (name, url) for name, url in brainweb.LINKS.items()
        }
        self.subject_ids = (
            tuple(available)
            if subject_ids is None
            else (subject_ids,) if isinstance(subject_ids, int) else tuple(subject_ids)
        )
        if any(x not in available for x in self.subject_ids):
            raise ValueError(
                f"Incorrect subject_ids. Available values are {tuple(available)}."
            )

        self.files = []
        for subject in self.subject_ids:
            name, url = available[subject]
            path = self.root / name
            if download:
                path = Path(brainweb.get_file(name, url, cache_dir=self.root))
            elif not path.is_file():
                raise RuntimeError(
                    f"BrainWeb PET subject {subject} not found in {self.root}. "
                    "Set download=True."
                )
            self.files.append(path)

        degradation_defaults = {
            "petNoise": 0.0,
            "t1Noise": 0.0,
            "t2Noise": 0.0,
            "petSigma": 0.0,
            "t1Sigma": 0.0,
            "t2Sigma": 0.0,
        }

        if random_degradations_kwargs is not None:
            degradation_defaults.update(random_degradations_kwargs)

        random_degradations_kwargs = degradation_defaults

        self.brainweb_kwargs = {
            **random_degradations_kwargs,
            "PetClass": pet_class,
        }
        self.lesion_kwargs = (
            {
                **(lesion_kwargs or {}),
                "diam": lesion_diameters,
                "PetClass": pet_class,
            }
            if lesion_diameters
            else None
        )
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        import brainweb

        subject = self.subject_ids[index]
        volumes = brainweb.get_mmr_fromfile(
            str(self.files[index]), **self.brainweb_kwargs
        )
        emission = volumes["PET"]
        params = {
            "attenuation": torch.as_tensor(
                volumes["uMap"], dtype=torch.float32
            ).unsqueeze(0),
        }
        for contrast in self.contrast:
            params[contrast.lower()] = torch.as_tensor(
                volumes[contrast], dtype=torch.float32
            ).unsqueeze(0)

        if self.lesion_kwargs is not None:
            if self.seed is not None:
                brainweb.seed(self.seed + subject)

            lesion_mask = torch.zeros(emission.shape, dtype=torch.uint8)
            for lesion_index, diameter in enumerate(self.lesion_kwargs["diam"]):
                lesion_label = lesion_index + 1
                original = emission.copy()
                lesion_kwargs = self.lesion_kwargs.copy()
                lesion_kwargs["diam"] = [diameter]
                for name in ("intensity", "blur"):
                    if name in lesion_kwargs and lesion_kwargs[name] is not None:
                        lesion_kwargs[name] = [lesion_kwargs[name][lesion_index]]

                emission = brainweb.add_lesions(emission, **lesion_kwargs)
                lesion_mask[torch.as_tensor(emission != original)] = lesion_label

            params["lesion_mask"] = lesion_mask.unsqueeze(0)

        emission = torch.as_tensor(emission, dtype=torch.float32).unsqueeze(0)

        if self.transform is not None:
            for key in params:
                params[key] = self.transform(params[key])
            emission = self.transform(emission)

        return (
            {"x": emission, "params": params}
            if self.use_dict_output
            else (emission, params)
        )
