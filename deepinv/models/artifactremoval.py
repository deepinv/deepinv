import torch
import torch.nn as nn
from .base import Reconstructor, Denoiser


class ArtifactRemoval(Reconstructor):
    r"""
    Artifact removal architecture.

    Transforms a denoiser :math:`\phi` into a reconstruction network :math:`R` by doing

    - Adjoint: :math:`\inversef{y}{A}=\phi(A^{\top}y)` with ``mode='adjoint'``.
    - Pseudoinverse: :math:`\inversef{y}{A}=\phi(A^{\dagger}y)` with ``mode='pinv'``.
    - Direct: :math:`\inversef{y}{A}=\phi(y)` with ``mode='direct'``.

    .. note::

        In the case of ``mode='pinv'``, the architecture is inspired by the FBPConvNet
        approach of https://arxiv.org/pdf/1611.03679 where a deep network :math:`\phi`
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
        mode="adjoint",
        pinv=False,
        ckpt_path=None,
        device=None,
    ):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

        if self.pinv:
            mode = "pinv"
        self.mode = mode

        if ckpt_path is not None:
            self.backbone_net.load_state_dict(torch.load(ckpt_path), strict=True)
            self.backbone_net.eval()

        if type(self.backbone_net).__name__ == "UNetRes":
            for _, v in self.backbone_net.named_parameters():
                v.requires_grad = False
            self.backbone_net = self.backbone_net.to(device)

    def forward(self, y, physics, **kwargs):
        r"""
        Reconstructs a signal estimate from measurements y

        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        """
        if isinstance(physics, nn.DataParallel):
            physics = physics.module

        if self.mode == "adjoint":
            y_in = physics.A_adjoint(y)
        elif self.mode == "pinv":
            y_in = physics.A_dagger(y)
        elif self.mode == "direct":
            y_in = y
        else:
            raise ValueError(
                "Invalid ArtifactRemoval mode. Options are 'direct', 'adjoint' or 'pinv'."
            )

        if type(self.backbone_net).__name__ == "UNetRes":
            noise_level_map = (
                torch.FloatTensor(y_in.size(0), 1, y_in.size(2), y_in.size(3))
                .fill_(kwargs["sigma"])
                .to(y_in.dtype)
            )
            y_in = torch.cat((y_in, noise_level_map), 1)

        if hasattr(physics.noise_model, "sigma"):
            sigma = physics.noise_model.sigma
        else:
            sigma = None

        return self.backbone_net(y_in, sigma)
