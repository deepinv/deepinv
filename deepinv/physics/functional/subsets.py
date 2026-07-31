from __future__ import annotations

import torch

from deepinv.physics.forward import LinearPhysics, StackedLinearPhysics
from deepinv.physics.pet import PET
from deepinv.physics.tomography import Tomography, TomographyWithAstra
from deepinv.utils.tensorlist import TensorList
from math import sqrt


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
        num_angles = len(physics.theta)
        dim = -1
    elif isinstance(physics, TomographyWithAstra):
        num_angles = physics.num_angles
        dim = -2
    elif isinstance(physics, PET):
        num_angles = physics.num_views
        dim = 2 + physics.proj.lor_descriptor.view_axis_num
    else:
        raise TypeError(
            "split_measurements is currently supported for deepinv.physics.Tomography, "
            "deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
        )

    if dim < 0:
        dim = y.dim() + dim
    indices = get_subset_indices(
        num_angles, num_subsets, strategy=strategy, device=y.device
    )
    return TensorList([y.index_select(dim, idx) for idx in indices])


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
    if isinstance(physics, Tomography):
        indices = get_subset_indices(
            len(physics.theta), num_subsets, strategy=strategy, device=physics.device
        )
        subset_physics = []
        subset_operator_norm = (
            physics.operator_norm.detach().clone() / sqrt(num_subsets)
            if physics.normalize
            else None
        )
        for idx in indices:
            # Tomography uses angles only to define the geometry
            theta_subset = physics.theta.index_select(0, idx.to(physics.theta.device))
            subset = Tomography(
                angles=theta_subset,
                normalize=False,
                device=physics.device,
                **physics._subset_kwargs,
            )
            subset.normalize = physics.normalize
            if subset_operator_norm is not None:
                subset.register_buffer(
                    "operator_norm", subset_operator_norm.detach().clone()
                )
            subset_physics.append(subset)
        return StackedLinearPhysics(subset_physics)

    if isinstance(physics, TomographyWithAstra):
        indices = get_subset_indices(
            physics.num_angles, num_subsets, strategy=strategy, device=physics.device
        )
        astra_geometry_vectors = (
            torch.as_tensor(
                physics.projection_geometry["Vectors"], device=physics.device
            )
            if "vec" in physics.projection_geometry["type"]
            else None
        )
        astra_angles = physics.angles
        subset_physics = []
        for idx in indices:
            # Astra can use both angles and vectors to define the geometry
            if astra_geometry_vectors is not None:
                geometry_vectors = astra_geometry_vectors.index_select(
                    0, idx.to(astra_geometry_vectors.device)
                )
                angles = torch.arange(len(geometry_vectors), device=physics.device)
            else:
                geometry_vectors = None
                angles = astra_angles.index_select(0, idx.to(astra_angles.device))

            subset = TomographyWithAstra(
                angles=angles,
                geometry_vectors=geometry_vectors,
                normalize=False,
                device=physics.device,
                **physics._subset_kwargs,
            )
            subset.normalize = physics.normalize
            if physics.normalize:
                subset.register_buffer(
                    "operator_norm", subset_operator_norm.detach().clone()
                )
            subset_physics.append(subset)
        return StackedLinearPhysics(subset_physics)

    if isinstance(physics, PET):
        indices = get_subset_indices(
            physics.num_views,
            num_subsets,
            strategy=strategy,
            device=physics.views.device,
        )
        view_dim = 2 + physics.proj.lor_descriptor.view_axis_num
        gain = getattr(physics.noise_model, "gain", torch.ones(1))
        normalize_counts = getattr(
            physics.noise_model, "normalize", torch.tensor(False)
        )
        subset_physics = []
        for idx in indices:
            # parallelproj uses angles only to define the geometry
            views = physics.views.index_select(0, idx.to(physics.views.device))
            background = physics.background.index_select(
                view_dim, idx.to(physics.background.device)
            )
            attenuation = physics.attenuation.index_select(
                view_dim, idx.to(physics.attenuation.device)
            )
            subset = PET(
                views=views,
                background=background,
                attenuation=attenuation,
                gain=gain.detach().clone(),
                normalize=False,
                normalize_counts=bool(normalize_counts.item()),
                device=physics.background.device,
                **physics._subset_kwargs,
            )
            subset.normalize = physics.normalize
            if physics.normalize:
                subset.register_buffer(
                    "operator_norm", subset_operator_norm.detach().clone()
                )
            subset_physics.append(subset)
        return StackedLinearPhysics(subset_physics)

    raise TypeError(
        "split_physics is currently supported for deepinv.physics.Tomography, "
        "deepinv.physics.TomographyWithAstra, and deepinv.physics.PET physics."
    )
