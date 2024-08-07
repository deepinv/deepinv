from torch import Tensor, rand
import torch.nn as nn
from deepinv.physics import Physics


class TimeAgnosticNet(nn.Module):
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

    :param torch.nn.Module backbone_net: Base network which can only take static inputs (B,C,H,W)
    :param torch.device device: cpu or gpu.
    """

    def __init__(self, backbone_net: nn.Module):
        super().__init__()
        self.backbone_net = backbone_net

    def flatten(self, a: Tensor) -> Tensor:
        B, C, T, H, W = a.shape
        return a.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    def unflatten(self, a: Tensor, batch_size=1) -> Tensor:
        BT, C, H, W = a.shape
        return a.reshape(batch_size, BT // batch_size, C, H, W).permute(0, 2, 1, 3, 4)

    def forward(self, y: Tensor, physics: Physics, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y

        :param Tensor y: measurements [B,C,T,H,W]
        :param deepinv.physics.Physics physics: forward operator acting on dynamic inputs
        """
        return self.unflatten(self.backbone_net(self.flatten(y), physics, **kwargs))
