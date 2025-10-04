"""Signal processing utilities"""

from __future__ import annotations

import torch


def normalize_signal(inp, *, mode):
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape (B, *).
    :param str mode: the normalization, either 'min_max' for min-max normalization or 'clip' for clipping. Note that min-max normalization of constant signals is ill-defined and here it amounts to mapping the constant value to the closest value between zero and one (which is equivalent to clipping).
    :return: the normalized batch of signals.
    """
    if mode == "min_max":
        # Compute the minimum and maximum intensity of the batched signals
        non_batched_dims = list(range(1, inp.ndim))
        minimum_intensity = inp.amin(dim=non_batched_dims, keepdim=False)
        maximum_intensity = inp.amax(dim=non_batched_dims, keepdim=False)

        # Clone the signal to avoid input mutations
        inp = inp.clone()

        # The indices corresponding to the non-constant batched signals
        indices = maximum_intensity != minimum_intensity

        # Prepare the tensors for broadcasting
        shape = (-1,) + (1,) * len(non_batched_dims)
        minimum_intensity = minimum_intensity.view(*shape)
        maximum_intensity = maximum_intensity.view(*shape)

        # Rescale the non-constant batched signals between zero and one
        inp[indices] -= minimum_intensity[indices]
        inp[indices] /= maximum_intensity[indices] - minimum_intensity[indices]

        # The indices corresponding to the constant batched signals
        indices = torch.logical_not(indices)

        # Clamp constant batched signals between zero and one
        inp[indices] = inp[indices].clamp(min=0.0, max=1.0)
    elif mode == "clip":
        # Clamp every batched signal between zero and one
        inp = inp.clamp(min=0.0, max=1.0)
    else:  # pragma: no cover
        raise ValueError(
            f"Unsupported normalization mode: {mode}. Supported modes are 'min_max' and 'clip'."
        )

    return inp


def complex_abs(data: torch.Tensor | None, dim=1, keepdim=True):
    """
    Compute the absolute value of a complex valued input tensor.

    If data has length 2 in the channel dimension given by dim, assumes this represents Re and Im parts.
    If data is a ``torch.complex`` dtype, takes absolute directly.

    :param torch.Tensor data: A complex valued tensor.
    :param int dim: complex dimension
    :param bool keepdim: keep complex dimension after abs
    """
    if data is None:
        return data

    if data.is_complex():
        return torch.abs(data)
    else:
        assert data.size(dim) == 2
        return (data**2).sum(dim=dim, keepdim=keepdim).sqrt()
