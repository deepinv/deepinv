from __future__ import annotations
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

from deepinv.models.base import Reconstructor, Denoiser

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class ArtifactRemoval(Reconstructor):
    r"""
    Artifact removal architecture.

    Transforms a denoiser :math:`\phi` into a reconstruction network :math:`R` by doing

    - Adjoint: :math:`\inversef{y}{A}=\phi(A^{\top}y)` with ``mode='adjoint'``.
    - Pseudoinverse: :math:`\inversef{y}{A}=\phi(A^{\dagger}y)` with ``mode='pinv'``.
    - Direct: :math:`\inversef{y}{A}=\phi(y)` with ``mode='direct'``.

    .. note::

        In the case of ``mode='pinv'``, the architecture is inspired by the FBPConvNet
        approach :footcite:t:`jin2017deep` where a deep network :math:`\phi`
        is used to improve the filtered back projection :math:`A^{\dagger}y`.

    .. deprecated:: 0.2.2

       The ``pinv`` parameter is deprecated and might be removed in future versions. Use ``mode`` instead.

    :param deepinv.models.Denoiser, torch.nn.Module backbone_net: Base denoiser network :math:`\phi`
        (see :ref:`denoisers` for available architectures).
    :param str mode: Reconstruction mode. Options are 'direct', 'adjoint' or 'pinv'.
    :param bool pinv: (deprecated) if ``True`` uses pseudo-inverse :math:`A^{\dagger}y` instead of the default transpose.
    :param torch.device device: cpu or gpu.
    """

    def __init__(
        self,
        backbone_net: Denoiser,
        mode: str = "adjoint",
        pinv: bool = False,
        ckpt_path: str = None,
        device: torch.device | str = None,
    ):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

        if self.pinv:
            mode = "pinv"
        self.mode = mode.lower()

        if ckpt_path is not None:
            self.backbone_net.load_state_dict(torch.load(ckpt_path), strict=True)
            self.backbone_net.eval()

        if device is not None:
            self.backbone_net = self.backbone_net.to(device)

    def backbone_inference(
        self, tensor_in: Tensor, physics: Physics, y: Tensor, **kwargs
    ) -> torch.Tensor:
        """Perform inference on the backbone network.

        By default, treats backbone network as a denoiser.
        Override for different inference e.g. for an unrolled network.

        :param torch.Tensor tensor_in: input tensor as dictated by ArtifactRemoval mode
        :param Physics physics: forward physics
        :param torch.Tensor y: input measurements y
        :return: reconstructed image
        """
        return self.backbone_net(
            tensor_in, getattr(physics.noise_model, "sigma", None), **kwargs
        )

    def forward(self, y: Tensor, physics: Physics, **kwargs) -> torch.Tensor:
        r"""
        Reconstructs a signal estimate from measurements y

        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param dict kwargs: additional keyword arguments for the backbone network.

        :return: reconstructed image
        """
        if isinstance(physics, nn.DataParallel):
            physics = physics.module

        if self.mode == "adjoint":
            x_hat = physics.A_adjoint(y)
        elif self.mode == "pinv":
            x_hat = physics.A_dagger(y)
        elif self.mode == "direct":
            x_hat = y
        else:
            raise ValueError(
                "Invalid ArtifactRemoval mode. Options are 'direct', 'adjoint' or 'pinv'."
            )

        return self.backbone_inference(x_hat, physics, y, **kwargs)
