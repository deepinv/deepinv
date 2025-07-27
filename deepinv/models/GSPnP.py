import torch
import torch.nn as nn
from torch import Tensor

from typing import Union
from .utils import get_weights_url
from .base import Denoiser


class StudentGrad(nn.Module):
    def __init__(self, denoiser):
        super().__init__()
        self.model = denoiser

    def forward(self, x, sigma):
        return self.model(x, sigma)


class GSPnP(Denoiser):
    r"""
    Gradient Step module to use a denoiser architecture as a Gradient Step Denoiser.

    See :footcite:t:`hurault2021gradient`. Code from https://github.com/samuro95/GSPnP.

    :param torch.nn.Module denoiser: Denoiser model.
    :param float alpha: Relaxation parameter
    :param bool detach: If `True`, the denoiser output will be detached from the computation graph.
        Setting this to `False` allows one to compute the gradient of the denoiser output with respect to the input, it is necessary in training.
        Default is `True`.
    """

    def __init__(self, denoiser, alpha: float = 1.0, detach: bool = True):
        super().__init__()
        self.student_grad = StudentGrad(denoiser)
        self.alpha = alpha
        self.detach = detach

    def potential(
        self, x: Tensor, sigma: Union[float, torch.Tensor], *args, **kwargs
    ) -> Tensor:
        N = self.student_grad(x, sigma)
        return (
            0.5
            * self.alpha
            * torch.norm((x - N).view(x.shape[0], -1), p=2, dim=-1) ** 2
        )

    def potential_grad(
        self, x: Tensor, sigma: Union[float, torch.Tensor], *args, **kwargs
    ) -> Tensor:
        r"""
        Calculate :math:`\nabla g` the gradient of the regularizer :math:`g` at input :math:`x`.

        :param torch.Tensor x: Input image
        :param float sigma: Denoiser level :math:`\sigma` (std)
        """
        with torch.enable_grad():
            x = x.to(torch.float32)
            x = x.requires_grad_()
            N = self.student_grad(x, sigma)
            JN = torch.autograd.grad(
                N, x, grad_outputs=x - N, create_graph=False if self.detach else True
            )[0]
        if self.detach:
            x = x.detach()
            JN = JN.detach()

        Dg = x - N - JN
        return self.alpha * Dg

    def forward(self, x: Tensor, sigma: Union[float, torch.Tensor]) -> Tensor:
        r"""
        Denoising with Gradient Step Denoiser

        :param torch.Tensor x: Input image
        :param float sigma: Denoiser level (std)
        """
        Dg = self.potential_grad(x, sigma)
        x_hat = x - Dg
        return x_hat


def GSDRUNet(
    alpha=1.0,
    in_channels=3,
    out_channels=3,
    nb=2,
    nc=[64, 128, 256, 512],
    act_mode="E",
    pretrained=None,
    device=torch.device("cpu"),
):
    """
    Gradient Step Denoiser with DRUNet architecture.

    Based on the GSPnP method from :footcite:t:`hurault2021gradient`.

    :param float alpha: Relaxation parameter
    :param int in_channels: Number of input channels
    :param int out_channels: Number of output channels
    :param int nb: Number of blocks in the DRUNet
    :param list[int,int,int,int] nc: number of channels per convolutional layer in the DRUNet. The network has a fixed number of 4 scales with ``nb`` blocks per scale (default: ``[64,128,256,512]``).
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU "E" for ELU and "S" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelsuffle" for pixel
        shuffling, and "upconv" for nearest neighbour upsampling with additional convolution.
    :param bool download: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param str device: gpu or cpu.
    """
    from deepinv.models.drunet import DRUNet

    denoiser = DRUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        nb=nb,
        nc=nc,
        act_mode=act_mode,
        pretrained=None,
        device=device,
    )
    GSmodel = GSPnP(denoiser, alpha=alpha)
    if pretrained:
        if pretrained == "download":
            if in_channels == 3:
                file_name = "GSDRUNet_torch.ckpt"
            elif in_channels == 1:
                file_name = "GSDRUNet_grayscale_torch.ckpt"
            url = get_weights_url(model_name="gradientstep", file_name=file_name)
            ckpt = torch.hub.load_state_dict_from_url(
                url,
                map_location=lambda storage, loc: storage,
                file_name=file_name,
            )
        else:
            ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        GSmodel.load_state_dict(ckpt, strict=False)
        GSmodel.eval()
    return GSmodel
