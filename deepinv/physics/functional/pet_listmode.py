from __future__ import annotations

import math
import torch

from deepinv.physics.pet import PET


def sinogram_to_listmode(
    physics, sinogram: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    r"""
    Expand a single integer count sinogram into events.

    :param deepinv.physics.PET physics: Sinogram PET acquisition.
    :param torch.Tensor sinogram: A single sinogram, without batch or channel dimensions,
        with the shape returned by ``A(x)[0, 0]``. Counts must be nonnegative integers.
    :return: Event start and end coordinates in mm, signed ToF bin numbers (or
        ``None``), and flattened sinogram indices. The indices can be passed to
        :func:`listmode_to_sinogram` and refer to this projector's selected views.

    Event ordering follows the flattened sinogram, not acquisition time. This
    conversion expands counts into individual events and can use substantial memory.
    """
    indices = torch.repeat_interleave(
        torch.arange(sinogram.numel(), device=sinogram.device),
        sinogram.reshape(-1).to(torch.long),
    )
    start, end = physics.proj.lor_descriptor.get_lor_coordinates(views=physics.views)
    if physics.tof_info is None:
        lor_indices = indices
        tof_bins = None
    else:
        n = physics.tof_info.num_tofbins
        lor_indices = torch.div(indices, n, rounding_mode="floor")
        tof_bins = (indices.remainder(n) - n // 2).to(torch.int16)
    return (
        start.reshape(-1, 3)[lor_indices],
        end.reshape(-1, 3)[lor_indices],
        tof_bins,
        indices,
    )


def listmode_physics_from_sinogram(
    physics, sinogram: torch.Tensor
) -> tuple[PET, torch.Tensor]:
    r"""Construct a listmode projector and event weights from one count sinogram.

    The acquisition geometry, normalization, attenuation and background are
    taken from this PET operator. The returned event weights are all ones,
    with shape ``(1, 1, num_events)``. The sinogram has no batch or channel
    dimensions. It must contain integer counts, not gain-normalized data.
    The PET forward model and background must be expressed in count units:
    for counts simulated with this operator, use ``gain=1``.
    Event attenuation and background are captured at construction; update
    them as well if the full acquisition calibration subsequently changes.

    :param deepinv.physics.PET physics: Sinogram PET acquisition.
    :param torch.Tensor sinogram: Integer counts without batch or channel dimensions.
    :return: Listmode PET physics and event weights.
    """
    start, end, bins, indices = sinogram_to_listmode(physics, sinogram)
    shape = sinogram.shape
    attenuation = torch.broadcast_to(physics.attenuation, (1, 1) + shape).reshape(-1)[
        indices
    ]
    background = torch.broadcast_to(physics.background, (1, 1) + shape).reshape(-1)[
        indices
    ]
    listmode = PET(
        physics.img_size,
        voxel_size=physics.voxel_size,
        fwhm_data_mm=physics.fwhm_data_mm,
        scanner=physics.scanner,
        radial_trim=physics.radial_trim,
        device=sinogram.device,
        views=physics.views,
        attenuation=physics.attenuation,
        tof_info=physics.tof_info,
        listmode=True,
        event_start_coordinates=start,
        event_end_coordinates=end,
        event_tofbins=bins,
        event_attenuation=attenuation,
        event_background=background,
    )
    listmode.operator_norm = physics.operator_norm

    return listmode, torch.ones((1, 1, indices.numel()), device=sinogram.device)


def listmode_to_sinogram(physics, indices: torch.Tensor) -> torch.Tensor:
    r"""
    Bin flattened event indices returned by :func:`sinogram_to_listmode`.

    :param deepinv.physics.PET physics: Sinogram PET acquisition.
    :param torch.Tensor indices: One flattened sinogram index per event.
    :return: Integer count sinogram without batch or channel dimensions.

    The indices must refer to this projector's geometry and selected views.
    Coordinates from arbitrary scanners cannot be binned with this method.
    """
    shape = physics.proj.out_shape
    if physics.is_2d:
        shape = shape[:-1] if physics.tof_info is None else shape[:-2] + shape[-1:]
    size = math.prod(shape)
    return torch.bincount(indices.to(torch.long), minlength=size).reshape(shape)
