from __future__ import annotations
from types import ModuleType
from typing import Optional, Callable

from torch import Tensor, ones
from torch.nn import Module

from deepinv.loss.metric.functional import complex_abs, norm
from deepinv.utils.plotting import rescale_img


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
        This is unused if the ``metric`` method is overridden.
    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: if higher is better, invert metric. If lower is better, does nothing.
    :param str reduction: a method to reduce metric score over individual batch scores. ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.

    |sep|

    Examples:

        Use ``Metric`` to wrap functional metrics such as from torchmetrics:

        >>> from functools import partial
        >>> from torchmetrics.functional.image import structural_similarity_index_measure
        >>> from deepinv.loss.metric import Metric
        >>> m = Metric(metric=partial(structural_similarity_index_measure, reduction='none'))
        >>> x = x_net = ones(2, 3, 64, 64) # B,C,H,W
        >>> m(x_net - 0.1, x)
        tensor([0., 0.])

    """

    def __init__(
        self,
        metric: Callable = None,
        complex_abs: bool = False,
        train_loss: bool = False,
        reduction: Optional[str] = None,
        norm_inputs: Optional[str] = None,
    ):
        super().__init__()
        self.train_loss = train_loss
        self.complex_abs = complex_abs  # NOTE assumes C in dim=1
        self._metric = metric
        normalizer = lambda x: x
        if norm_inputs is not None:
            if not isinstance(norm_inputs, str):
                raise ValueError(
                    "norm_inputs must either be l2, min_max, none or None."
                )
            elif norm_inputs.lower() == "min_max":
                normalizer = lambda x: rescale_img(x, rescale_mode="min_max")
            elif norm_inputs.lower() == "l2":
                normalizer = lambda x: x / norm(x)
            elif norm_inputs.lower() == "none":
                pass
            else:
                raise ValueError(
                    "norm_inputs must either be l2, min_max, none or None."
                )
        self.normalizer = lambda x: x if x is None else normalizer(x)
        self.reducer = lambda x: x
        if reduction is not None:
            if not isinstance(reduction, str):
                raise ValueError("reduction must either be mean, sum, none or None.")
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

    def metric(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Calculate metric on data.

        Override this function to implement your own metric. Always include ``args`` and ``kwargs`` arguments.

        :param torch.Tensor x_net: Reconstructed image :math:`\hat{x}=\inverse{y}` of shape ``(B, ...)`` or ``(B, C, ...)``.
        :param torch.Tensor x: Reference image :math:`x` (optional) of shape ``(B, ...)`` or ``(B, C, ...)``.

        :return torch.Tensor: calculated metric, the tensor size might be ``(1,)`` or ``(B,)``.
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

        :param torch.Tensor x_net: Reconstructed image :math:`\hat{x}=\inverse{y}` of shape ``(B, ...)`` or ``(B, C, ...)``.
        :param torch.Tensor x: Reference image :math:`x` (optional) of shape ``(B, ...)`` or ``(B, C, ...)``.

        :return torch.Tensor: calculated metric, the tensor size might be ``(1,)`` or ``(B,)``.
        """
        if isinstance(x_net, (list, tuple)):
            x_net = x_net[0] if x_net is not None else x_net
            x = x[0] if x is not None else x

        if self.complex_abs:
            x_net, x = complex_abs(x_net), complex_abs(x)

        m = self.metric(
            self.normalizer(x_net),
            self.normalizer(x),
            *args,
            **kwargs,
        )

        m = self.reducer(m)

        if self.train_loss and not self.lower_better:
            return self.invert_metric(m)
        else:
            return m
