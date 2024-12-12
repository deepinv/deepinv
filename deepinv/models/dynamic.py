from __future__ import annotations
from typing import TYPE_CHECKING

from torch import Tensor
from torch.nn import Module

from deepinv.models.base import Reconstructor
from deepinv.utils.mixin import TimeMixin
if TYPE_CHECKING:
    from deepinv.physics import Physics

class TimeAgnosticNet(Reconstructor, TimeMixin):
    r"""
    Time-agnostic network wrapper.

    Adapts a static image reconstruction network to process time-varying inputs.
    The image reconstruction network then processes the data independently frame-by-frame.

    Flattens time dimension into batch dimension at input, and unflattens at output.

    |sep|

    :Example:

    >>> from deepinv.models import UNet, TimeAgnosticNet
    >>> model = UNet(scales=2)
    >>> model = TimeAgnosticNet(model)
    >>> y = rand(1, 1, 4, 8, 8) # B,C,T,H,W
    >>> x_net = model(y, None)
    >>> x_net.shape == y.shape
    True

    :param Module backbone_net: Base network which can only take static inputs (B,C,H,W)
    :param torch.device device: cpu or gpu.
    """

    def __init__(self, backbone_net: Module):
        super().__init__()
        self.backbone_net = backbone_net

    def forward(self, y: Tensor, physics: Physics, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y

        :param Tensor y: measurements [B,C,T,H,W]
        :param Physics physics: forward operator acting on dynamic inputs
        """
        return self.unflatten(self.backbone_net(self.flatten(y), physics, **kwargs))


class TimeAveragingNet(Module, TimeMixin):
    r"""
    Time-averaging network wrapper.

    Network wrapper to flatten dynamic input.

    Adapts a static image reconstruction network for time-varying inputs to output static reconstructions.
    Average the data across the time dim before passing into network.

    .. note::

        The input physics is assumed to be a temporal physics which produced the temporal measurements y (potentially with temporal mask ``mask``).
        It must either implement a ``to_static`` method to remove the time dimension, or already be a static physics (e.g. :class:`deepinv.physics.MRI`).

    |sep|

    :Example:

    >>> from deepinv.models import UNet, TimeAveragingNet
    >>> model = UNet(scales=2)
    >>> model = TimeAveragingNet(model)
    >>> y = rand(1, 1, 4, 8, 8) # B,C,T,H,W
    >>> x_net = model(y, None)
    >>> x_net.shape # B,C,H,W
    torch.Size([1, 1, 8, 8])

    :param Module backbone_net: Base network which can only take static inputs (B,C,H,W)
    :param torch.device device: cpu or gpu.
    """

    def __init__(self, backbone_net):
        super().__init__()
        self.backbone_net = backbone_net

    def forward(self, y, physics: TimeMixin, **kwargs):
        return self.backbone_net(
            self.average(y, getattr(physics, "mask", None)),
            getattr(physics, "to_static", lambda: physics)(),
            **kwargs,
        )
