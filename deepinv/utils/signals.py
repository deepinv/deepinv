"""Signal processing utilities"""

from __future__ import annotations
from warnings import warn

import torch


def normalize_signal(
    inp: torch.Tensor,
    *,
    mode: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> torch.Tensor:
    r"""
    Normalize a batch of signals between zero and one.

    :param torch.Tensor inp: the input signal to normalize, it should be of shape `(B, *)`.
    :param str mode: the normalization, either `'min_max'` for min-max normalization or `'clip'` for clipping.
        If ``clip`` is selected, the values of ``vmin`` and ``vmax`` are used as clipping bounds if provided,
        otherwise the default bounds of 0.0 and 1.0 are used.
        Note that min-max normalization of constant signals is ill-defined and here it amounts to mapping the constant
        value to the closest value between zero and one (which is equivalent to clipping).
    :return: the normalized batch of signals.

    """
    if mode != "clip":
        if vmin is not None or vmax is not None:
            warn(
                "The vmin and vmax arguments are used only when using 'clip' rescaling.",
                UserWarning,
                stacklevel=2,
            )
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError(
            f"vmin should be strictly less than vmax, got vmin={vmin} and vmax={vmax}."
        )
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
        if vmin is None:
            vmin = 0.0
        if vmax is None:
            vmax = 1.0
        inp = inp.clamp(min=vmin, max=vmax)
        inp = (inp - vmin) / (vmax - vmin + 1e-12)  # rescale to [0, 1]

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
        return torch.linalg.vector_norm(data, dim=dim, ord=2, keepdim=keepdim)
