r"""
NumPy-style histograms in PyTorch

Copy from: https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py
"""

import torch

from torch import Size, Tensor, BoolTensor
from typing import Union, Sequence


def ravel_multi_index(coords: Tensor, shape: Size) -> torch.Tensor:
    r"""
    Converts a tensor of coordinate vectors into a tensor of flat indices.

    This is a ``torch`` implementation of ``numpy.ravel_multi_index``.

    :param torch.Tensor coords: A tensor of coordinate vectors, ``(*, D)``.
    :param tuple shape: The source shape.
    :return: The raveled indices, ``(*,)``.

    """

    shape = coords.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return (coords * coefs).sum(dim=-1)


def unravel_index(indices: Tensor, shape: Size) -> torch.Tensor:
    r"""
    Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a ``torch`` implementation of ``numpy.unravel_index``.

    :param torch.Tensor indices: A tensor of flat indices, (*,).
    :param tuple shape: The target shape.
    :return: The unraveled coordinates, ``(*, D)``.
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]


def out_of_bounds(x: Tensor, low: Tensor, upp: Tensor) -> BoolTensor:
    r"""
    Returns a mask of out-of-bounds values in x.

    :param torch.Tensor x: A tensor, ``(*, D)``.
    :param torch.Tensor, float low: The lower bound in each dimension, scalar or ``(D,)``.
    :param torch.Tensor, float upp: The upper bound in each dimension, scalar or ``(D,)``.
    :return: the mask tensor, ``(*,)``.

    """

    a, b = x < low, x > upp

    if x.dim() > 1:
        a, b = torch.any(a, dim=-1), torch.any(b, dim=-1)

    return torch.logical_or(a, b)


def quantize(x: Tensor, bins: Tensor, low: Tensor, upp: Tensor) -> torch.Tensor:
    r"""
    Maps the values of x to integers.

    :param torch.Tensor x: A tensor, ``(*, D)``.
    :param torch.Tensor bins: The number of bins in each dimension, scalar or (D,).
    :param torch.Tensor, float low: The lower bound in each dimension, scalar or ``(D,)``.
    :param torch.Tensor, float upp: The upper bound in each dimension, scalar or ``(D,)``.
    :return: The quantized tensor, ``(*, D)``.

    """

    x = (x - low) / (upp - low)  # in [0.0, 1.0]
    x = (bins * x).long()  # in [0, bins]

    return x


def histogramdd(
    x: Tensor,
    bins: Union[int, Sequence[int]] = 10,
    low: Union[float, Sequence[float]] = None,
    upp: Union[float, Sequence[float]] = None,
    bounded: bool = False,
    weights: Tensor = None,
    sparse: bool = False,
    edges: Union[Tensor, Sequence[Tensor]] = None,
) -> torch.Tensor:
    r"""
    Computes the multidimensional histogram of a tensor.

    This is a ``torch`` implementation of ``numpy.histogramdd``.
    This function is borrowed from `torchist <https://github.com/francois-rozet/torchist/>`_.

    Note:
        Similar to ``numpy.histogram``, all bins are half-open except the last bin which
        also includes the upper bound.


    :param torch.Tensor x: A tensor, (\*, D).
    :param int, list[int] bins: The number of bins in each dimension, scalar or (D,).
    :param float, list[float] low: The lower bound in each dimension, scalar or (D,). If `low` is ``None``,
            the min of `x` is used instead.
    :param float, list[float] upp: The upper bound in each dimension, scalar or (D,). If `upp` is ``None``,
            the max of `x` is used instead.
    :param bool bounded: Whether `x` is bounded by `low` and `upp`, included.
            If `False`, out-of-bounds values are filtered out.
    :param torch.Tensor weights: A tensor of weights, ``(\*,)``. Each sample of `x` contributes
            its associated weight towards the bin count (instead of 1).
    :param bool sparse: Whether the histogram is returned as a sparse tensor or not.
    :param torch.Tensor, list[torch.Tensor] edges: The edges of the histogram. Either a vector or a list of vectors.
            If provided, ``bins``, ``low`` and ``upp`` are inferred from ``edges``.

    :return: (:class:`torch.Tensor`) : the histogram
    """

    # Preprocess
    D = x.size(-1)
    x = x.reshape(-1, D).squeeze(-1)

    if edges is None:
        bounded = bounded or (low is None and upp is None)

        if low is None:
            low = x.min(dim=0).values

        if upp is None:
            upp = x.max(dim=0).values
    elif torch.is_tensor(edges):
        edges = edges.flatten().to(x)
        bins = edges.numel() - 1
        low = edges[0]
        upp = edges[-1]
    else:
        edges = [e.flatten() for e in edges]
        bins = [e.numel() - 1 for e in edges]
        low = [e[0] for e in edges]
        upp = [e[-1] for e in edges]

        pack = x.new_full((D, max(bins) + 1), float("inf"))

        for i, e in enumerate(edges):
            pack[i, : e.numel()] = e.to(x)  # pad with inf

        edges = pack

    bins = torch.as_tensor(bins, dtype=torch.long, device=x.device).squeeze()
    low = torch.as_tensor(low, dtype=x.dtype, device=x.device).squeeze()
    upp = torch.as_tensor(upp, dtype=x.dtype, device=x.device).squeeze()

    assert torch.all(
        upp > low
    ), "The upper bound must be strictly larger than the lower bound"

    if weights is not None:
        weights = weights.flatten()

    # Filter out-of-bound values
    if not bounded:
        mask = ~out_of_bounds(x, low, upp)

        x = x[mask]

        if weights is not None:
            weights = weights[mask]

    # Indexing
    if edges is None:
        idx = quantize(x, bins, low, upp)
    elif edges.dim() > 1:
        idx = torch.searchsorted(edges, x.t().contiguous(), right=True).t() - 1
    else:
        idx = torch.bucketize(x, edges, right=True) - 1

    idx = torch.clip(idx, min=None, max=bins - 1)  # last bin includes upper bound

    # Histogram
    shape = torch.Size(bins.expand(D).tolist())

    if sparse:
        if weights is None:
            idx, values = torch.unique(idx, dim=0, return_counts=True)
        else:
            idx, inverse = torch.unique(idx, dim=0, return_inverse=True)
            values = weights.new_zeros(len(idx))
            values = values.scatter_add(dim=0, index=inverse, src=weights)

        hist = torch.sparse_coo_tensor(idx.t(), values, shape)
        hist._coalesced_(True)
    else:
        if D > 1:
            idx = ravel_multi_index(idx, shape)
        hist = idx.bincount(weights, minlength=shape.numel()).reshape(shape)

    return hist


def histogram(
    x: Tensor,
    bins: int = 10,
    low: float = None,
    upp: float = None,
    **kwargs,
) -> torch.Tensor:
    r"""Computes the histogram of a tensor.

    This is a `torch` implementation of `numpy.histogram`.


    :param torch.Tensor x: A tensor, ``(*,)``.
    :param int bins: The number of bins.
    :param float low: The lower bound. If `low` is ``None`` the min of `x` is used instead.
    :param float upp: The upper bound. If `upp` is ``None`` the max of `x` is used instead.
    :param kwargs: Keyword arguments passed to `histogramdd`.

    :return torch.Tensor: The histogram
    """

    return histogramdd(x.unsqueeze(-1), bins, low, upp, **kwargs)
