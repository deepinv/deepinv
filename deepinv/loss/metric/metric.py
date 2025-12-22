from __future__ import annotations
from types import ModuleType
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from deepinv.loss.metric.functional import norm
from deepinv.utils.signals import normalize_signal, complex_abs


def import_pyiqa() -> ModuleType:
    try:
        import pyiqa

        return pyiqa
    except ImportError:
        raise ImportError(
            "Metric not available. Please install the pyiqa package with `pip install pyiqa`."
        )


class Metric(Module):
    r"""
    Base class for metrics.

    See docs for :func:`forward <deepinv.loss.metric.Metric.forward>` below for more details.

    To create a new metric, inherit from this class, override the :func:`metric method <deepinv.loss.metric.Metric.metric>`,
    set ``lower_better`` attribute and optionally override the ``invert_metric`` method.

    You can also directly use this baseclass to wrap an existing metric function, e.g. from
    `torchmetrics <https://lightning.ai/docs/torchmetrics/stable>`_, to benefit from our preprocessing.
    The metric function must reduce over all dims except the batch dim (see example).

    :param Callable metric: metric function, it must reduce over all dims except batch dim. It must not reduce over batch dim.
        This is unused if the ``metric`` method is overridden. Takes as input `x_net` and `x` tensors and returns a tensor of metric scores.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: if higher is better, invert metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2`` normalizes by :math:`\ell_2` spatial norm, ``min_max`` normalizes by min and max of each input, ``clip`` clips to :math:`[0,1]`, ``standardize`` standardizes to same mean and std as ground truth, ``none`` or None no reduction will be applied (default).
    :param int, tuple[int], None center_crop: If not `None` (default), center crop the tensor(s) before computing the metrics.
        If an `int` is provided, the cropping is applied equally on all spatial dimensions (by default, all dimensions except the first two).
        If `tuple` of `int`, cropping is performed over the last `len(center_crop)` dimensions. If positive values are provided, a standard center crop is applied.
        If negative (or zero) values are passed, cropping will be done by removing `center_crop` pixels from the borders (useful when tensors vary in size across the dataset).

    |sep|

    Examples:

        Use ``Metric`` to wrap functional metrics such as from torchmetrics:

        >>> from functools import partial
        >>> from torchmetrics.functional.image import structural_similarity_index_measure
        >>> from deepinv.loss.metric import Metric
        >>> m = Metric(metric=partial(structural_similarity_index_measure, reduction='none'))
        >>> x = x_net = torch.ones(2, 3, 64, 64) # B,C,H,W
        >>> m(x_net - 0.1, x)
        tensor([0., 0.])

    """

    def __init__(
        self,
        metric: Callable[[Tensor, Tensor], Tensor] = None,
        complex_abs: bool = False,
        train_loss: bool = False,
        reduction: str | None = None,
        norm_inputs: str | None = None,
        center_crop: int | tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.train_loss = train_loss
        self.complex_abs = complex_abs  # NOTE assumes C in dim=1
        self._metric = metric
        self.center_crop = center_crop

        if isinstance(center_crop, tuple):
            if not (
                all(c > 0 for c in center_crop) or all(c <= 0 for c in center_crop)
            ):
                raise ValueError(
                    "If center_crop is a tuple, all values must be either positive or negative."
                )

        normalizer = lambda x: x
        if norm_inputs is not None:
            if not isinstance(norm_inputs, str):
                raise ValueError("norm_inputs must be str or None.")
            elif norm_inputs.lower() == "min_max":
                normalizer = lambda x: normalize_signal(x, mode="min_max")
            elif norm_inputs.lower() == "clip":
                normalizer = lambda x: normalize_signal(x, mode="clip")
            elif norm_inputs.lower() == "l2":
                normalizer = lambda x: x / norm(x)
            elif norm_inputs.lower() == "standardize":
                pass
            elif norm_inputs.lower() == "none":
                pass
            else:
                raise ValueError(
                    "norm_inputs must either be l2, min_max, none or None."
                )
        self.normalizer = lambda x: x if x is None else normalizer(x)
        self.norm_inputs = norm_inputs

        self.reducer = lambda x: x
        if reduction is not None:
            if callable(reduction):
                self.reducer = reduction
            elif not isinstance(reduction, str):
                raise ValueError("reduction must either be str, callable, or None.")
            elif reduction.lower() == "mean":
                self.reducer = lambda x: x.mean()
            elif reduction.lower() == "sum":
                self.reducer = lambda x: x.sum()
            elif reduction.lower() == "none":
                pass
            else:
                raise ValueError("reduction must either be mean, sum, none or None.")

        # Subclasses override this if higher is better (e.g. in SSIM)
        self.lower_better = True

    def _apply_center_crop(self, x: Tensor) -> Tensor:
        """Apply center crop to tensor.

        :param torch.Tensor x: input tensor of shape (B, C, ...)
        :return torch.Tensor: center cropped tensor
        """
        if self.center_crop is None or x is None:
            return x

        # Convert int to tuple for all spatial dimensions (all dims except first two)
        if isinstance(self.center_crop, int):
            n_spatial_dims = x.ndim - 2  # Exclude batch and channel dims
            crop_sizes = (self.center_crop,) * n_spatial_dims
        else:
            crop_sizes = self.center_crop

        # Number of spatial dimensions to crop
        n_crop_dims = len(crop_sizes)

        # Check if we have enough dimensions to crop
        if x.ndim < 2 + n_crop_dims:
            raise ValueError(
                f"Tensor has {x.ndim} dimensions but center_crop requires at least {2 + n_crop_dims} dimensions"
            )

        # Apply cropping to the last n_crop_dims dimensions
        slices = [slice(None)] * x.ndim
        for i, crop_size in enumerate(crop_sizes):
            dim_idx = x.ndim - n_crop_dims + i
            dim_size = x.shape[dim_idx]

            if crop_size > 0:
                # Standard center crop
                if crop_size > dim_size:
                    raise ValueError(
                        f"Crop size {crop_size} is larger than dimension size {dim_size} at dimension {dim_idx}"
                    )
                start = (dim_size - crop_size) // 2
                end = start + crop_size
            else:
                # Negative or zero: remove pixels from borders
                border_pixels = abs(crop_size)
                if 2 * border_pixels >= dim_size:
                    raise ValueError(
                        f"Border removal of {border_pixels} pixels on each side would remove entire dimension of size {dim_size}"
                    )
                start = border_pixels
                end = dim_size - border_pixels

            slices[dim_idx] = slice(start, end)

        return x[tuple(slices)]

    def metric(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Calculate metric on data.

        Override this function to implement your own metric. Always include ``args`` and ``kwargs`` arguments.
        Do not perform reduction.

        :param torch.Tensor x_net: Reconstructed image :math:`\hat{x}=\inverse{y}` of shape ``(B, ...)`` or ``(B, C, ...)``.
        :param torch.Tensor x: Reference image :math:`x` (optional) of shape ``(B, ...)`` or ``(B, C, ...)``.

        :return torch.Tensor: calculated unreduced metric of shape ``(B,)``.
        """
        return self._metric(x_net, x, *args, **kwargs)

    def invert_metric(self, m: Tensor):
        """Invert metric. Used where a higher=better metric is to be used in a training loss.

        :param torch.Tensor m: calculated metric
        """
        return -m

    def forward(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Metric forward pass.

        Usually, the data passed is ``x_net, x`` i.e. estimate and target or only ``x_net`` for no-reference metric.

        The forward pass also optionally calculates complex magnitude of images, performs normalisation,
        or inverts the metric to use it as a training loss (if by default higher is better).

        By default, no reduction is performed in the batch dimension, but mean or sum reduction can be performed too.

        All tensors should be of shape ``(B, ...)`` or ``(B, C, ...)`` where ``B`` is batch size and ``C`` is channels.

        .. note::

            If a full reference metric is used and a tensor is `None`, a tensor of NaN will be returned instead.

        :param torch.Tensor x_net: Reconstructed image :math:`\hat{x}=\inverse{y}` of shape ``(B, ...)`` or ``(B, C, ...)``.
        :param torch.Tensor x: Reference image :math:`x` (optional) of shape ``(B, ...)`` or ``(B, C, ...)``.

        :return torch.Tensor: calculated metric, the tensor size might be ``(1,)`` or ``(B,)``.
        """
        if isinstance(x_net, (list, tuple)):
            x_net = x_net[0] if x_net is not None else x_net
            x = x[0] if x is not None else x

        if self.complex_abs:
            x_net, x = complex_abs(x_net), complex_abs(x)

        # Apply center crop before normalization
        x_net = self._apply_center_crop(x_net)
        x = self._apply_center_crop(x)

        if self.norm_inputs == "standardize":
            if x_net is None or x is None:
                raise ValueError(
                    "Both x and x_net must not be None in order to use standardize."
                )
            x_net = (x_net - x_net.mean()) / x_net.std() * x.std() + x.mean()

        x_net = self.normalizer(x_net)
        x = self.normalizer(x) if x is not None else None

        if x_net is None:
            return torch.tensor([torch.nan])
        else:
            m = self.metric(x_net, x, *args, **kwargs)

        m = self.reducer(m)

        if self.train_loss and not self.lower_better:
            return self.invert_metric(m)
        else:
            return m

    def __add__(self, other: Metric):
        """Sums two metrics via the + operation.

        :param deepinv.loss.metric.Metric other: other metric
        :return: :class:`deepinv.loss.metric.Metric` summed metric.
        """
        return Metric(
            metric=lambda x_net, x, *args, **kwargs: self.forward(
                x_net, x, *args, **kwargs
            )
            + other.forward(x_net, x, *args, **kwargs)
        )
