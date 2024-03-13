# import DeepInv
import torch
import torch.nn as nn


class ArtifactRemoval(nn.Module):
    r"""
    Artifact removal architecture :math:`\phi(A^{\top}y)`.

    The architecture is inspired by the FBPConvNet approach of https://arxiv.org/pdf/1611.03679
    where a deep network :math:`\phi` is used to improve the linear reconstruction :math:`A^{\top}y`.

    :param torch.nn.Module backbone_net: Base network :math:`\phi`, can be pretrained or not.
    :param bool pinv: If ``True`` uses pseudo-inverse :math:`A^{\dagger}y` instead of the default transpose.
    :param torch.device device: cpu or gpu.
    """

    def __init__(self, backbone_net, pinv=False, ckpt_path=None, device=None):
        super(ArtifactRemoval, self).__init__()
        self.pinv = pinv
        self.backbone_net = backbone_net

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

        y_in = physics.A_adjoint(y) if not self.pinv else physics.A_dagger(y)
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
