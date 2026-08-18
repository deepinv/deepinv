from __future__ import annotations

import torch

from deepinv.physics.forward import LinearPhysics, StackedLinearPhysics
from deepinv.physics.pet import PET
from deepinv.physics.tomography import Tomography, TomographyWithAstra
from deepinv.utils.tensorlist import TensorList


def get_subset_tensor(
    tensor: torch.Tensor,
    num_subsets: int,
) -> list[torch.Tensor]:
    r"""Return indices that interleave a tensor into equal subsets.

    :param torch.Tensor tensor: tensor containing angles or geometry vectors to split along its first dimension.
    :param int num_subsets: number of subsets.
    :return: list of index tensors.
    """
    if not isinstance(num_subsets, int) or num_subsets < 1:
        raise ValueError("num_subsets must be a positive integer.")
    if num_subsets > len(tensor):
        raise ValueError("num_subsets cannot exceed the number of views.")
    if len(tensor) % num_subsets != 0:
        raise ValueError("Tensor length must be divisible by num_subsets.")

    indices = torch.arange(len(tensor), device=tensor.device)
    return list(indices.reshape(-1, num_subsets).T)


def split_measurements(
    y: torch.Tensor,
    physics: LinearPhysics,
    num_subsets: int,
) -> TensorList:
    r"""
    Splits tomography measurements into angular subsets.

    .. note::

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
    :return: measurements as a :class:`deepinv.utils.TensorList`.
    """
    if isinstance(physics, Tomography):
        indices = get_subset_tensor(physics.angles, num_subsets)
        dim = -1
    elif isinstance(physics, TomographyWithAstra):
        angles = physics.angles
        if angles is not None:
            indices = get_subset_tensor(angles, num_subsets)
        else:
            geometry_vectors = physics.geometry_vectors
            indices = get_subset_tensor(geometry_vectors, num_subsets)
        dim = -2
    elif isinstance(physics, PET):
        indices = get_subset_tensor(physics.views, num_subsets)
        dim = 2 + physics.proj.lor_descriptor.view_axis_num
    else:
        raise TypeError(
            "split_measurements is currently supported for "
            "deepinv.physics.Tomography, deepinv.physics.TomographyWithAstra, "
            "and deepinv.physics.PET physics."
        )

    if dim < 0:
        dim = y.dim() + dim
    return TensorList([y.index_select(dim, idx.to(y.device)) for idx in indices])


def split_physics(
    physics: LinearPhysics,
    num_subsets: int,
    device: torch.device | str,
) -> StackedLinearPhysics:
    r"""
    Builds a stacked tomography physics with one operator per angular subset.

    .. warning::

        If ``physics`` is normalized, each subset reuses the operator norm of the
        complete physics instead of computing its own.
        Computing the real subset physics operator norm would result in a mismatch
        between the projections of the full physics and the subset physics.

    :param deepinv.physics.LinearPhysics physics: tomography physics.
    :param int num_subsets: number of subsets.
    :param torch.device, str device: device on which to create the subset physics.
    :return: :class:`deepinv.physics.StackedLinearPhysics` over angular subsets.
    """
    if not isinstance(physics, (Tomography, TomographyWithAstra, PET)):
        raise TypeError(
            "split_physics is currently supported for deepinv.physics.Tomography, deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
        )

    if isinstance(physics, Tomography):
        subset_physics = [
            Tomography(
                angles=physics.angles.index_select(0, idx).to(device),
                img_width=physics.img_width,
                circle=physics.radon.circle,
                parallel_computation=physics.radon.parallel_computation,
                adjoint_via_backprop=physics.adjoint_via_backprop,
                fbp_interpolate_boundary=physics.fbp_interpolate_boundary,
                normalize=False,
                fan_beam=physics.fan_beam,
                fan_parameters=physics.radon.fan_parameters,
                device=device,
                dtype=physics.radon.all_grids.dtype,
            )
            for idx in get_subset_tensor(physics.angles, num_subsets)
        ]

    elif isinstance(physics, TomographyWithAstra):
        angles = physics.angles
        if angles is not None:
            subset_physics = [
                TomographyWithAstra(
                    img_size=physics.img_size,
                    angles=angles.index_select(0, idx).to(device),
                    angular_range=physics.angular_range,
                    n_detector_pixels=physics.n_detector_pixels,
                    detector_spacing=physics.detector_spacing,
                    pixel_spacing=physics.pixel_spacing,
                    bounding_box=physics.bounding_box,
                    geometry_type=physics.geometry_type,
                    geometry_parameters=physics.geometry_parameters,
                    normalize=False,
                    device=device,
                )
                for idx in get_subset_tensor(angles, num_subsets)
            ]
        else:
            geometry_vectors = physics.geometry_vectors
            subset_physics = [
                TomographyWithAstra(
                    img_size=physics.img_size,
                    angles=torch.arange(len(idx), device=device),
                    angular_range=physics.angular_range,
                    n_detector_pixels=physics.n_detector_pixels,
                    detector_spacing=physics.detector_spacing,
                    pixel_spacing=physics.pixel_spacing,
                    bounding_box=physics.bounding_box,
                    geometry_type=physics.geometry_type,
                    geometry_parameters=physics.geometry_parameters,
                    geometry_vectors=geometry_vectors.index_select(0, idx).to(device),
                    normalize=False,
                    device=device,
                )
                for idx in get_subset_tensor(geometry_vectors, num_subsets)
            ]

    else:
        indices = get_subset_tensor(physics.views, num_subsets)
        view_dim = 2 + physics.proj.lor_descriptor.view_axis_num
        subset_physics = [
            PET(
                img_size=physics.img_size,
                voxel_size=physics.voxel_size,
                fwhm_data_mm=physics.fwhm_data_mm,
                scanner=physics.scanner,
                radial_trim=physics.radial_trim,
                gain=physics.noise_model.gain.detach().clone().to(device),
                normalize=False,
                normalize_counts=bool(physics.noise_model.normalize),
                device=device,
                views=physics.views.index_select(0, idx).to(device),
                background=physics.background.index_select(view_dim, idx).to(device),
                attenuation=physics.attenuation.index_select(view_dim, idx).to(device),
            )
            for idx in indices
        ]

    if physics.normalize:
        for subset in subset_physics:
            subset.register_buffer(
                "operator_norm",
                physics.operator_norm.detach().clone().to(device),
            )
            subset.normalize = True

    return StackedLinearPhysics(subset_physics)
