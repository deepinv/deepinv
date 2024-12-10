from __future__ import annotations
from typing import Union, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn

from deepinv.models.base import Denoiser
from deepinv.models.artifactremoval import ArtifactRemoval
from deepinv.models import DnCNN
from deepinv.physics.mri import MRIMixin

if TYPE_CHECKING:
    from deepinv.physics.forward import Physics


class VarNet(ArtifactRemoval, MRIMixin):
    """
    VarNet or E2E-VarNet model.

    These models are from the papers
    `Sriram et al., End-to-End Variational Networks for Accelerated MRI Reconstruction <https://arxiv.org/abs/2004.06688>`_
    and
    `Hammernik et al., Learning a variational network for reconstruction of accelerated MRI data <https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26977>`_.

    This performs unrolled iterations on the image estimate x (as per the original VarNet paper)
    or the kspace y (as per E2E-VarNet).

    Note that we do not currently support sensitivity-map estimation for multicoil MRI.

    Code loosely adapted from E2E-VarNet implementation from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/varnet.py.

    :param Denoiser, nn.Module denoiser: backbone network that parametrises the grad of the regulariser.
        If ``None``, a small DnCNN is used.
    :param int num_cascades: number of unrolled iterations ('cascades').
    :param str mode: if 'varnet', perform iterates on the images x as in original VarNet.
        If 'e2e-varnet', perform iterates on the kspace y as in the E2E-VarNet.
    """

    def __init__(
        self,
        denoiser: Union[Denoiser, nn.Module] = None,
        num_cascades: int = 12,
        mode: str = "varnet",
    ):
        if mode.lower() == "varnet":
            self.estimate_x = True
        elif mode.lower() == "e2e-varnet":
            self.estimate_x = False
        else:
            raise ValueError("mode must either be 'varnet' or 'e2e-varnet'.")

        denoiser = (
            denoiser
            if denoiser is not None
            else DnCNN(
                in_channels=2,
                out_channels=2,
                pretrained=None,
                depth=7,
            )
        )

        cascades = nn.Sequential(
            *[
                VarNetBlock(denoiser, estimate_x=self.estimate_x)
                for _ in range(num_cascades)
            ]
        )

        super().__init__(
            backbone_net=cascades,
            mode="adjoint" if self.estimate_x else "direct",
            device=None,
        )

    def backbone_inference(
        self, tensor_in: Tensor, physics: Physics, y: Tensor
    ) -> Tensor:
        """Perform inference on input tensor.

        Uses physics and y for data consistency.
        If necessary, perform fully-sampled MRI IFFT on model output.

        :param Tensor tensor_in: input tensor as dictated by VarNet mode (either k-space or image)
        :param Physics physics: forward physics for data consistency
        :param Tensor y: input measurements y for data consistency
        :return: Tensor: reconstructed image
        """
        hat, _, _ = self.backbone_net((tensor_in, physics, y))
        if self.estimate_x:
            return hat
        else:
            return self.from_torch_complex(
                self.ifft(self.to_torch_complex(hat), dim=(-2, -1))
            )


class VarNetBlock(nn.Module):
    """
    One unrolled iteration ("cascade") of VarNet or E2E-VarNet.
    See :class:`deepinv.models.VarNet` for details.

    :param Denoiser, nn.Module denoiser: backbone denoiser network.
    :param bool estimate_x: whether estimate images x, or kspaces y.
    """

    def __init__(self, denoiser: Union[Denoiser, nn.Module], estimate_x: bool = True):
        super().__init__()

        self.denoiser = denoiser
        self.estimate_x = estimate_x
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        args_in: tuple,
    ) -> tuple:
        """Forward pass of one VarNet block

        The following arguments should be passed in as a tuple ``args_in``.

        :param Tensor tensor_in: input tensor, either images ``x`` or kspaces ``y`` depending on ``self.estimate_x``.
        :param MRI physics: forward physics including updated mask
        :param Tensor y: input kspace measurements.
        :return: ``(tensor_out, physics, y)``, where tensor_out is either images ``x`` or kspaces ``y``.
        """
        tensor_in, physics, y = args_in

        y_in = tensor_in if not self.estimate_x else physics.A(tensor_in)

        dc = physics.mask * (y_in - y)

        if self.estimate_x:
            dc = physics.A_adjoint(dc)

        tensor_out = tensor_in - dc * self.dc_weight - self.denoiser(tensor_in)

        return (tensor_out, physics, y)
