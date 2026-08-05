from __future__ import annotations

import warnings

import torch

from deepinv.physics.forward import LinearPhysics, StackedLinearPhysics
from deepinv.physics.pet import PET
from deepinv.physics.tomography import Tomography, TomographyWithAstra
from deepinv.utils.tensorlist import TensorList


def get_subset_angles(
    angles: torch.Tensor,
    num_subsets: int,
) -> list[torch.Tensor]:
    r"""Return indices that interleave angles into equal subsets.

    :param torch.Tensor angles: acquisition angles of shape ``(num_angles,)``.
    :param int num_subsets: number of subsets.
    :return: list of index tensors.
    """
    indices = torch.arange(len(angles), device=angles.device)
    return list(indices.reshape(-1, num_subsets).T)


def get_subset_vectors(
    geometry_vectors: torch.Tensor,
    num_subsets: int,
) -> list[torch.Tensor]:
    r"""Return indices that interleave ASTRA vectors into equal subsets.

    :param torch.Tensor geometry_vectors: ASTRA geometry vectors of shape
        ``(num_views, 12)``.
    :param int num_subsets: number of subsets.
    :return: list of index tensors.
    """
    indices = torch.arange(len(geometry_vectors), device=geometry_vectors.device)
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
        indices = get_subset_angles(physics.angles, num_subsets)
        dim = -1
    elif isinstance(physics, TomographyWithAstra):
        angles = physics.angles
        if angles is not None:
            indices = get_subset_angles(angles, num_subsets)
        else:
            geometry_vectors = torch.as_tensor(
                physics.projection_geometry["Vectors"], device=y.device
            )
            indices = get_subset_vectors(geometry_vectors, num_subsets)
        dim = -2
    elif isinstance(physics, PET):
        indices = get_subset_angles(physics.views, num_subsets)
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
) -> StackedLinearPhysics:
    r"""
    Builds a stacked tomography physics with one operator per angular subset.

    :param deepinv.physics.LinearPhysics physics: tomography physics.
    :param int num_subsets: number of subsets.
    :return: :class:`deepinv.physics.StackedLinearPhysics` over angular subsets.
    """
    if not isinstance(physics, (Tomography, TomographyWithAstra, PET)):
        raise TypeError(
            "split_physics is currently supported for deepinv.physics.Tomography, deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
        )

    if physics.normalize:
        warnings.warn(
            "Subsetted physics cannot be normalized. Divide the output of the subsetted physics by the operator norm of the complete physics.",
            stacklevel=2,
        )

    if isinstance(physics, Tomography):
        subset_physics = [
            physics.clone(angles=physics.angles.index_select(0, idx))
            for idx in get_subset_angles(physics.angles, num_subsets)
        ]

    elif isinstance(physics, TomographyWithAstra):
        angles = physics.angles
        if angles is not None:
            subset_physics = [
                physics.clone(angles=angles.index_select(0, idx))
                for idx in get_subset_angles(angles, num_subsets)
            ]
        else:
            geometry_vectors = torch.as_tensor(
                physics.projection_geometry["Vectors"], device=physics.device
            )
            subset_physics = [
                physics.clone(geometry_vectors=geometry_vectors.index_select(0, idx))
                for idx in get_subset_vectors(geometry_vectors, num_subsets)
            ]

    else:
        indices = get_subset_angles(physics.views, num_subsets)
        view_dim = 2 + physics.proj.lor_descriptor.view_axis_num
        background_scale = physics.operator_norm if physics.normalize else 1.0
        subset_physics = [
            physics.clone(
                views=physics.views.index_select(0, idx.to(physics.views.device)),
                background=physics.background.index_select(
                    view_dim, idx.to(physics.background.device)
                )
                * background_scale,
                attenuation=physics.attenuation.index_select(
                    view_dim, idx.to(physics.attenuation.device)
                ),
            )
            for idx in indices
        ]

    return StackedLinearPhysics(subset_physics)
