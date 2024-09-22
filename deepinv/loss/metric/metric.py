from __future__ import annotations
from types import ModuleType
from typing import Optional, TYPE_CHECKING

from torch import Tensor
from torch.nn import Module

from deepinv.loss.loss import Loss
from deepinv.loss.metric.functional import complex_abs, norm
from deepinv.utils.plotting import rescale_img

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


def import_pyiqa() -> ModuleType:
    try:
        import pyiqa

        return pyiqa
    except ImportError:
        raise ImportError(
            "Metric not available. Please install the pyiqa package with `pip install pyiqa`."
        )


class Metric(Loss):
    r"""
    Base class for metrics.

    See docs for ``forward()`` below for more details.

    To create a new metric, inherit from this class and override the function :meth:`deepinv.loss.metric.Metric.metric`.

    :param bool complex_abs: perform complex magnitude before passing data to metric function. If ``True``,
        the data must either be of complex dtype or have size 2 in the channel dimension (usually the second dimension after batch).
    :param bool train_loss: use metric as a training loss, by returning one minus the metric. If lower is better, does nothing.
    :param str reduction: ``mean``: takes the mean, ``sum`` takes the sum, ``none`` or None no reduction will be applied (default).
    :param str norm_inputs: normalize images before passing to metric. ``l2``normalizes by L2 spatial norm, ``min_max`` normalizes by min and max of each input.
    """

    def __init__(
        self,
        complex_abs: bool = False,
        train_loss: bool = False,
        reduction: Optional[str] = None,
        norm_inputs: Optional[str] = None,
    ):
        super().__init__()
        self.lower_better = True
        self.train_loss = train_loss
        self.complex_abs = complex_abs  # NOTE assumes C in dim=1
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

    def metric(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        y: Tensor = None,
        physics: Physics = None,
        model: Module = None,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Calculate metric on data.

        Override this function to implement your own metric. Always include ``args`` and ``kwargs`` arguments.

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return torch.Tensor: calculated metric, the tensor size might be (1,) or (batch size,).
        """
        raise NotImplementedError()

    def forward(
        self,
        x_net: Tensor = None,
        x: Tensor = None,
        y: Tensor = None,
        physics: Physics = None,
        model: Module = None,
        *args,
        **kwargs,
    ) -> Tensor:
        r"""Metric forward pass.

        Usually, the data passed is ``x_net, x`` i.e. estimate and target or only ``x_net`` for no-reference metric.
        In general, the data can also include measurements ``y``, ``physics`` and the reconstruction ``model``.

        The forward pass also optionally calculates complex magnitude of images, performs normalisation,
        or inverts the metric to use it as a training loss.

        By default, no reduction is performed in the batch dimension, but mean or sum reduction can be performed too.

        All tensors should be of shape (B, ...) or (B, C, ...).

        :param torch.Tensor x_net: Reconstructed image :math:`\inverse{y}`.
        :param torch.Tensor x: Reference image.
        :param torch.Tensor y: Measurement.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.

        :return torch.Tensor: calculated metric, the tensor size might be (1,) or (batch size,).
        """
        if isinstance(x_net, (list, tuple)):
            x_net = x_net[0]
            x = x[0] if x is not None else x
            y = y[0] if y is not None else y

        if self.complex_abs:
            x_net, x, y = complex_abs(x_net), complex_abs(x), complex_abs(y)

        m = self.metric(
            x_net=self.normalizer(x_net),
            x=self.normalizer(x),
            y=self.normalizer(y),
            physics=physics,
            model=model,
            *args,
            **kwargs,
        )

        m = self.reducer(m)

        m = (1.0 - m) if (self.train_loss and not self.lower_better) else m
        return m
