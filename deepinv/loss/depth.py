from __future__ import annotations
from typing import TYPE_CHECKING

from torch import Tensor

from deepinv.loss.mc import MCLoss

if TYPE_CHECKING:
    from deepinv.physics.rendering import RenderingPhysics


class ViewConsistencyLoss(MCLoss):
    r"""
    View consistency loss

    This loss takes a batch of views, where the first view is assumed to be the model input
    and the rest are the reference views.

    This loss then enforces that the reference views are consistent with the
    rendered views from the reconstructed object (which could be a 2.5D or 3D representation).

    The view consistency loss is defined as

    .. math::

        \|y_\text{ref}-\forw{\inverse{y_\text{input},\Pi}}\|

    where :math:\forw{\cdot} is the rendering forward physics, :math:`\Pi` is the relative pose between the reference and input views,
    :math:`\inverse{y}` is the reconstructed representation, and :math:`\|\cdot\|` is the backbone metric.

    By default, the error is computed using the MSE metric, however any other metric (e.g. LPIPS or SSIM)
    can be used as well.

    :param Metric, torch.nn.Module metric: metric used for computing data consistency, which is set as the mean squared error by default.
    """

    def forward(self, y: Tensor, x_net: Tensor, physics: RenderingPhysics, **kwargs):
        r"""
        Computes the view consistency loss

        :param torch.Tensor y: batch of B views
        :param torch.Tensor x_net: Reconstructed 2.5D or 3D representation :math:`\inverse{y}`.
        :param deepinv.physics.Physics physics: rendering operator
        :return: (torch.Tensor) loss.
        """
        # Absolute poses of all (len B) measurements
        R, T = physics.pose_R, physics.pose_T

        # Split images into input (len 1) and references (len B-1)
        # Assumes model reconstructs from pose of first image in batch i.e. x_net = model(y_in)
        y_in, R_in, T_in = y[[0]], R[[0]], T[[0]]
        y_to, R_to, T_to = y[1:], R[1:], T[1:]

        # Calculate relative poses from input to references
        # R2^-1 (y2-T2) = R1^-1 (y1-T1)
        # y2 = R2R1^-1 (y1-T1) + T2 = R2R1^-1y1 - R2R1^-1T1 + T2
        # y2 = R_rel y_1 + T_rel
        R_rel = R_to @ R_in.inverse()
        T_rel = R_rel @ T_in + T_to

        # Render to (B-1) reference images
        y_hat = physics.A(x_net, pose_R=R_rel, pose_T=T_rel)

        # Reset physics parameters
        physics.update_parameters(pose_R=R, pose_T=T)

        return self.metric(y_hat, y_to)
