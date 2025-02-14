from __future__ import annotations
from typing import Union, TYPE_CHECKING
from warnings import warn

import torch
from torch import Tensor
import torch.nn as nn

from deepinv.models.base import Denoiser
from deepinv.models.artifactremoval import ArtifactRemoval
from deepinv.models import DnCNN
from deepinv.physics.mri import MRIMixin, MRI, MultiCoilMRI

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

    .. note::

        For singlecoil MRI, either mode is valid.
        For multicoil MRI, the VarNet mode will simply sum over the coils (not preferred). Using E2E-VarNet is therefore preferred.
        For sensitivity-map estimation for multicoil MRI, pass in ``sensitivity_model``.

    Code loosely adapted from E2E-VarNet implementation from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/varnet.py.

    :param Denoiser, torch.nn.Module denoiser: backbone network that parametrises the grad of the regulariser.
        If ``None``, a small DnCNN is used.
    :param torch.nn.Module sensitivity_model: network to jointly estimate coil sensitivity maps for multi-coil MRI. If ``None``, do not perform any map estimation. For single-coil MRI, unused.
    :param int num_cascades: number of unrolled iterations ('cascades').
    :param str mode: if 'varnet', perform iterates on the images x as in original VarNet.
        If 'e2e-varnet', perform iterates on the kspace y as in the E2E-VarNet.
    """

    def __init__(
        self,
        denoiser: Union[Denoiser, nn.Module] = None,
        sensitivity_model: nn.Module = None,
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

        self.sensitivity_model = sensitivity_model

        super().__init__(
            backbone_net=cascades,
            mode="adjoint" if self.estimate_x else "direct",
            device=None,
        )

    def backbone_inference(
        self, tensor_in: Tensor, physics: Union[MRI, MultiCoilMRI], y: Tensor
    ) -> torch.Tensor:
        """Perform inference on input tensor.

        Uses physics and y for data consistency.
        If necessary, perform fully-sampled MRI IFFT on model output.

        :param torch.Tensor tensor_in: input tensor as dictated by VarNet mode (either k-space or image)
        :param Physics physics: forward physics for data consistency
        :param torch.Tensor y: input measurements y for data consistency
        :return: (:class:`torch.Tensor`) reconstructed image
        """
        if self.sensitivity_model is not None:
            if self.mode != "e2e-varnet":
                warn(
                    "sensitivity_model provided but will not be used when model is not e2e-varnet."
                )

            coil_maps = self.sensitivity_model(y, physics)
        else:
            coil_maps = None

        hat, _, _, _ = self.backbone_net((tensor_in, physics, y, coil_maps))
        mask = physics.mask

        if not self.estimate_x:
            # Convert estimate to image domain
            hat = physics.A_adjoint(
                hat, mask=torch.ones(hat.shape[-2:], dtype=hat.dtype, device=hat.device)
            )

        physics.update_parameters(mask=mask)

        return hat


class VarNetBlock(nn.Module):
    """
    One unrolled iteration ("cascade") of VarNet or E2E-VarNet.
    See :class:`deepinv.models.VarNet` for details.

    :param Denoiser, torch.nn.Module denoiser: backbone denoiser network.
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

        :param torch.Tensor tensor_in: input tensor, either images ``x`` or kspaces ``y`` depending on ``self.estimate_x``.
        :param MRI physics: forward physics including updated mask
        :param torch.Tensor y: input kspace measurements.
        :param Optional[torch.Tensor] coil_maps: if ``sensitivity_model is not None``, this will contain coil map estimates for E2E-VarNet. Otherwise, it will be ``None``.
        :return: ``(tensor_out, physics, y, coil_maps)``, where tensor_out is either images ``x`` or kspaces ``y``.
        """
        tensor_in, physics, y, coil_maps = args_in

        y_in = tensor_in if not self.estimate_x else physics.A(tensor_in)

        mask = physics.mask

        if len(mask.shape) == len(y.shape):
            dc = mask * (y_in - y)
        elif len(mask.shape) == len(y.shape) - 1:
            # y is multicoil, mask is not
            dc = mask.unsqueeze(2).expand_as(y) * (y_in - y)
        else:
            raise ValueError(
                "Measurements y should either be same shape as physics mask, or have one additional dimension for multicoil data."
            )

        if self.estimate_x:
            # DC term in image domain
            dc = physics.A_adjoint(dc, coil_maps=coil_maps)

            # Denoises images directly
            denoised = self.denoiser(tensor_in)
        else:
            # DC term in measurement domain
            # Denoiser in image domain so convert from measurements
            ones_mask = torch.ones_like(mask)
            denoised = physics.A(
                self.denoiser(
                    physics.A_adjoint(tensor_in, mask=ones_mask, coil_maps=coil_maps)
                ),
                mask=ones_mask,
                coil_maps=coil_maps,
            )

        tensor_out = tensor_in - dc * self.dc_weight - denoised

        # Reset physics back to original mask
        physics.update_parameters(mask=mask)

        return (tensor_out, physics, y, coil_maps)
