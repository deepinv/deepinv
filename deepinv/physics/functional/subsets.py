from __future__ import annotations

from math import sqrt

import torch

from deepinv.physics.forward import LinearPhysics, StackedLinearPhysics
from deepinv.physics.pet import PET
from deepinv.physics.tomography import Tomography, TomographyWithAstra
from deepinv.utils.tensorlist import TensorList

SUPPORTED_TOMOGRAPHY_PHYSICS = (Tomography, TomographyWithAstra, PET)
UNSUPPORTED_SPLIT_MEASUREMENTS_ERROR = (
    "split_measurements is currently supported for deepinv.physics.Tomography, "
    "deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
)
UNSUPPORTED_SPLIT_PHYSICS_ERROR = (
    "split_physics is currently supported for deepinv.physics.Tomography, "
    "deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
)


def get_subset_indices(
    num_angles: int,
    num_subsets: int,
    strategy: str = "default",
    device: torch.device | str | None = None,
) -> list[torch.Tensor]:
    r"""
    Returns ordered angular subset indices.

    The ``"default"`` strategy splits the acquisition angles into interleaved,
    non-overlapping subsets of equal size.

    :param int num_angles: number of acquisition angles.
    :param int num_subsets: number of subsets.
    :param str strategy: subsetting strategy. Currently only ``"default"`` is supported.
    :param torch.device, str, None device: device of the returned index tensors.
    :return: list of index tensors.
    """
    if not isinstance(num_subsets, int) or num_subsets < 1:
        raise ValueError("num_subsets must be a positive integer.")
    if num_subsets > num_angles:
        raise ValueError("num_subsets cannot exceed the number of angles.")
    if strategy != "default":
        raise ValueError(f'Unknown subsetting strategy "{strategy}".')
    if num_angles % num_subsets != 0:
        raise ValueError(
            "The default subsetting strategy requires num_angles to be divisible "
            "by num_subsets."
        )

    subset_size = num_angles // num_subsets
    indices = torch.arange(num_angles, device=device)
    return list(indices.reshape(subset_size, num_subsets).T)


def split_measurements(
    y: torch.Tensor,
    physics: LinearPhysics,
    num_subsets: int,
    strategy: str = "default",
) -> TensorList:
    r"""
    Splits tomography measurements into angular subsets.

    .. warning::

        The expected measurement layout depends on the tomography physics used:

        * :class:`deepinv.physics.Tomography`: ``[B, C, N, A]``, where ``A``
          is the angle axis and ``N`` is the detector axis.
        * :class:`deepinv.physics.TomographyWithAstra`: ``[B, C, A, N]`` in
          2D and ``[B, C, V, A, N]`` in 3D, where ``V`` and ``N`` are the
          detector axes.
        * :class:`deepinv.physics.PET`: ``[B, C, N, A]`` in 2D and
          ``[B, C, N, A, P]`` in 3D for the default RVP sinogram order, where
          ``C = 1``, ``N`` is the radial detector axis, ``A`` is the view
          axis, and ``P`` is the plane axis.

    :param torch.Tensor y: full measurement tensor.
    :param deepinv.physics.LinearPhysics physics: tomography physics.
    :param int num_subsets: number of subsets.
    :param str strategy: subsetting strategy. Currently only ``"default"`` is supported.
    :return: measurements as a :class:`deepinv.utils.TensorList`.
    """
    if isinstance(physics, Tomography):
        num_angles = len(physics.angles)
        dim = -1
    elif isinstance(physics, TomographyWithAstra):
        num_angles = physics.num_angles
        dim = -2
    elif isinstance(physics, PET):
        num_angles = physics.num_views
        dim = 2 + physics.proj.lor_descriptor.view_axis_num
    else:
        raise TypeError(UNSUPPORTED_SPLIT_MEASUREMENTS_ERROR)

    if dim < 0:
        dim = y.dim() + dim
    indices = get_subset_indices(
        num_angles, num_subsets, strategy=strategy, device=y.device
    )
    return TensorList([y.index_select(dim, idx) for idx in indices])


def _get_tomography_subset_kwargs(physics: Tomography) -> dict:
    """Extract the geometry needed to rebuild a native tomography subset."""
    return {
        "img_width": physics.img_width,
        "circle": physics.radon.circle,
        "parallel_computation": physics.radon.parallel_computation,
        "adjoint_via_backprop": physics.adjoint_via_backprop,
        "fbp_interpolate_boundary": physics.fbp_interpolate_boundary,
        "fan_beam": physics.fan_beam,
        "fan_parameters": physics.radon.fan_parameters,
        "normalize": False,
        "device": physics.device,
        "dtype": physics.dtype,
    }


def _get_astra_subset_kwargs(physics: TomographyWithAstra) -> dict:
    """Extract the geometry needed to rebuild an ASTRA tomography subset."""
    projection_geometry = physics.projection_geometry
    object_geometry = physics.object_geometry
    object_options = object_geometry["option"]
    is_vector_geometry = "vec" in projection_geometry["type"]

    if physics.is_2d:
        detector_spacing = float(projection_geometry["DetectorSpacingX"])
        pixel_spacing = (
            (object_options["WindowMaxX"] - object_options["WindowMinX"])
            / object_geometry["GridColCount"],
            (object_options["WindowMaxY"] - object_options["WindowMinY"])
            / object_geometry["GridRowCount"],
        )
        bounding_box = (
            object_options["WindowMinX"],
            object_options["WindowMaxX"],
            object_options["WindowMinY"],
            object_options["WindowMaxY"],
        )
    else:
        detector_spacing = (
            (1.0, 1.0)
            if is_vector_geometry
            else (
                float(projection_geometry["DetectorSpacingY"]),
                float(projection_geometry["DetectorSpacingX"]),
            )
        )
        pixel_spacing = (
            (object_options["WindowMaxX"] - object_options["WindowMinX"])
            / object_geometry["GridColCount"],
            (object_options["WindowMaxY"] - object_options["WindowMinY"])
            / object_geometry["GridRowCount"],
            (object_options["WindowMaxZ"] - object_options["WindowMinZ"])
            / object_geometry["GridSliceCount"],
        )
        bounding_box = (
            object_options["WindowMinX"],
            object_options["WindowMaxX"],
            object_options["WindowMinY"],
            object_options["WindowMaxY"],
            object_options["WindowMinZ"],
            object_options["WindowMaxZ"],
        )

    geometry_parameters = (
        {
            "source_radius": projection_geometry["DistanceOriginSource"],
            "detector_radius": projection_geometry["DistanceOriginDetector"],
        }
        if physics.geometry_type in ("fanbeam", "conebeam") and not is_vector_geometry
        else None
    )
    return {
        "img_size": physics.img_size,
        "n_detector_pixels": physics.n_detector_pixels,
        "detector_spacing": detector_spacing,
        "pixel_spacing": pixel_spacing,
        "bounding_box": bounding_box,
        "geometry_type": physics.geometry_type,
        "geometry_parameters": geometry_parameters,
        "normalize": False,
        "device": physics.device,
    }


def _get_pet_subset_kwargs(physics: PET) -> dict:
    """Extract the geometry and acquisition settings for a PET subset."""
    gain = getattr(physics.noise_model, "gain", torch.ones(1))
    normalize_counts = getattr(physics.noise_model, "normalize", torch.tensor(False))
    return {
        "img_size": physics.img_size,
        "voxel_size": physics.voxel_size,
        "fwhm_data_mm": physics.fwhm_data_mm,
        "scanner": physics.scanner,
        "radial_trim": physics.radial_trim,
        "gain": gain.detach().clone(),
        "normalize": False,
        "normalize_counts": bool(normalize_counts.item()),
        "device": physics.background.device,
    }


def _get_tomography_subset_physics(
    physics: Tomography, indices: list[torch.Tensor], subset_kwargs: dict
) -> list[Tomography]:
    """Construct native tomography physics for each angular subset."""
    return [
        Tomography(
            angles=physics.angles.index_select(0, idx.to(physics.angles.device)),
            **subset_kwargs,
        )
        for idx in indices
    ]


def _get_astra_subset_physics(
    physics: TomographyWithAstra, indices: list[torch.Tensor], subset_kwargs: dict
) -> list[TomographyWithAstra]:
    """Construct ASTRA tomography physics for each angular subset."""
    angles = physics.angles
    geometry_vectors = (
        torch.as_tensor(physics.projection_geometry["Vectors"], device=physics.device)
        if angles is None
        else None
    )
    subsets = []
    for idx in indices:
        if angles is not None:
            subset = TomographyWithAstra(
                angles=angles.index_select(0, idx.to(angles.device)),
                **subset_kwargs,
            )
        else:
            subset_vectors = geometry_vectors.index_select(
                0, idx.to(geometry_vectors.device)
            )
            subset = TomographyWithAstra(
                angles=torch.arange(len(subset_vectors), device=physics.device),
                geometry_vectors=subset_vectors,
                **subset_kwargs,
            )
        subsets.append(subset)
    return subsets


def _get_pet_subset_physics(
    physics: PET, indices: list[torch.Tensor], subset_kwargs: dict
) -> list[PET]:
    """Construct PET physics for each angular subset."""
    view_dim = 2 + physics.proj.lor_descriptor.view_axis_num
    return [
        PET(
            views=physics.views.index_select(0, idx.to(physics.views.device)),
            background=physics.background.index_select(
                view_dim, idx.to(physics.background.device)
            ),
            attenuation=physics.attenuation.index_select(
                view_dim, idx.to(physics.attenuation.device)
            ),
            **subset_kwargs,
        )
        for idx in indices
    ]


def split_physics(
    physics: LinearPhysics,
    num_subsets: int,
    strategy: str = "default",
) -> StackedLinearPhysics:
    r"""
    Builds a stacked tomography physics with one operator per angular subset.

    :param deepinv.physics.LinearPhysics physics: tomography physics.
    :param int num_subsets: number of subsets.
    :param str strategy: subsetting strategy. Currently only ``"default"`` is supported.
    :return: :class:`deepinv.physics.StackedLinearPhysics` over angular subsets.
    """
    if not isinstance(physics, SUPPORTED_TOMOGRAPHY_PHYSICS):
        raise TypeError(UNSUPPORTED_SPLIT_PHYSICS_ERROR)

    # Get the total number of angles
    if isinstance(physics, Tomography):
        num_angles = len(physics.angles)
    elif isinstance(physics, TomographyWithAstra):
        num_angles = physics.num_angles
    else:
        num_angles = physics.num_views

    # Get the angles indices corresponding to each subset
    indices = get_subset_indices(
        num_angles,
        num_subsets,
        strategy=strategy,
        device=physics.device,
    )

    # Branch depending on the tomography physics
    if isinstance(physics, Tomography):
        subset_kwargs = _get_tomography_subset_kwargs(physics)
        subset_physics = _get_tomography_subset_physics(physics, indices, subset_kwargs)
    elif isinstance(physics, TomographyWithAstra):
        subset_kwargs = _get_astra_subset_kwargs(physics)
        subset_physics = _get_astra_subset_physics(physics, indices, subset_kwargs)
    else:
        subset_kwargs = _get_pet_subset_kwargs(physics)
        subset_physics = _get_pet_subset_physics(physics, indices, subset_kwargs)

    # Approximate operator norm of each subset physics
    if physics.normalize:
        subset_operator_norm = physics.operator_norm.detach().clone() / sqrt(
            num_subsets
        )
        for subset in subset_physics:
            subset.normalize = True
            subset.register_buffer(
                "operator_norm", subset_operator_norm.detach().clone()
            )

    return StackedLinearPhysics(subset_physics)
