from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from deepinv.loss.loss import Loss
from deepinv.loss.metric.metric import Metric
from deepinv.models.base import Reconstructor
from deepinv.physics.forward import Physics

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class SupLoss(Loss):
    r"""
    Standard supervised loss

    The supervised loss is defined as

    .. math::

        \frac{1}{n}\|x-\inverse{y}\|^2

    where :math:`\inverse{y}` is the reconstructed signal and :math:`x` is the ground truth target of :math:`n` elements.

    By default, the error is computed using the MSE metric, however any other metric (e.g., :math:`\ell_1`)
    can be used as well.
    If called with arguments ``x_net, x``, this is simply a wrapper for the metric ``metric``.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    """

    def __init__(self, metric: Metric | torch.nn.Module | None = None):
        if metric is None:
            metric = torch.nn.MSELoss()
        super().__init__()
        self.name = "supervised"
        self.metric = metric

    def forward(self, x_net, x, **kwargs):
        r"""
        Computes the loss.

        :param torch.Tensor x_net: Reconstructed image :math:\inverse{y}.
        :param torch.Tensor x: Target (ground-truth) image.
        :return: (:class:`torch.Tensor`) loss.
        """
        return self.metric(x_net, x)


class ReducedResolutionLoss(SupLoss):
    r"""
    Reduced resolution loss for blur and downsampling problems.

    The reduced resolution loss is defined as

    .. math::

        \frac{1}{n}\|y-\inverse{\forw{y}}\|^2

    where :math:`\forw{y}` is the reduced resolution measurement via further degrading, and the measurement :math:`y` is used a supervisory signal.

    .. note::

        Optionally initialize with physics to fix the reduced resolution operator. If not passed, the loss takes the physics from the forward pass during training.
        However, this should only be used with physics that can be used to meaningfully further degrade the measurements
        :math:`y`, such as blur or downsampling. The physics must be defined without an `img_size` so it can be applied
        to the measurements :math:`y`.

    At test time, the model does not perform the reduced resolution measurement.

    .. hint::

        During training, consider using the `disable_train_metrics` option in :class:`deepinv.Trainer` to prevent a shape
        mismatch during metric computation if the reduced resolution output will be smaller than ground truth.

    This loss was used in :footcite:t:`shocher2017zero-shot` for downsampling tasks, and is named Wald's protocol :footcite:p:`wald1997fusion`
    for pan-sharpening tasks.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency,
        which is set as the mean squared error by default.
    :param Physics physics: optional physics to perform reduced resolution measurement. If not specified, take the physics from the forward pass.
    """

    def __init__(
        self, metric: Metric | torch.nn.Module | None = None, physics: Physics = None
    ):
        super().__init__(metric=metric)
        self.physics = physics

    def forward(self, x_net: Tensor, y: Tensor, *args, **kwargs):
        r"""
        Computes the reduced resolution loss.

        :param torch.Tensor x_net: reconstructions.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module model: Reconstruction function.
        :return: (:class:`torch.Tensor`) loss.
        """
        try:
            return self.metric(x_net, y)
        except BaseException as e:
            raise RuntimeError(
                f"Metric error. Check that the reconstruction (of shape {x_net.shape}) and y (of shape {y.shape}) can be used to calculate the metric. Full error:",
                str(e),
            )

    def adapt_model(self, model: torch.nn.Module) -> ReducedResolutionModel:
        if isinstance(model, self.ReducedResolutionModel):
            return model
        else:
            return self.ReducedResolutionModel(model, self.physics)

    class ReducedResolutionModel(Reconstructor):
        def __init__(self, model: Reconstructor, physics: Physics | None):
            super().__init__()
            self.model = model
            self.physics = physics

        def forward(self, y: Tensor, physics: Physics, **kwargs):
            if self.training:
                phys = self.physics if self.physics is not None else physics
                try:
                    z = phys(y)
                except BaseException as e:
                    raise RuntimeError(
                        "Physics error. Check that the used physics can be applied to y to generate a further degraded y. Full error:",
                        str(e),
                    )
                try:
                    return self.model(z, phys)
                except BaseException as e:
                    raise RuntimeError(
                        "Model error. Check that the model can be used with a reduced-resolution input physics.A(y). Full error:",
                        str(e),
                    )
            else:
                return self.model(y, physics)
