from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor

from .utils import get_weights_url, load_state_dict_from_url
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
        self, x: Tensor, sigma: float | torch.Tensor, *args, **kwargs
    ) -> Tensor:
        N = self.student_grad(x, sigma)
        return (
            0.5
            * self.alpha
            * torch.linalg.vector_norm(x - N, dim=tuple(range(1, x.dim())), ord=2) ** 2
        )

    def potential_grad(
        self, x: Tensor, sigma: float | torch.Tensor, *args, **kwargs
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

    def forward(self, x: Tensor, sigma: float | torch.Tensor) -> Tensor:
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
    nc=(64, 128, 256, 512),
    act_mode="E",
    pretrained=None,
    device=torch.device("cpu"),
):
    """
    Gradient Step Denoiser with DRUNet architecture.

    Based on the GSPnP method from :footcite:t:`hurault2021gradient`.

    See :class:`deepinv.models.ProxDRUNet` for the proximal version of this denoiser from :footcite:t:`hurault2022proximal`.

    :param float alpha: Relaxation parameter
    :param int in_channels: Number of input channels
    :param int out_channels: Number of output channels
    :param int nb: Number of blocks in the DRUNet
    :param Sequence[int,int,int,int] nc: number of channels per convolutional layer in the DRUNet. The network has a fixed number of 4 scales with ``nb`` blocks per scale (default: ``[64,128,256,512]``).
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU, "E" for ELU and "s" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelshuffle" for pixel
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
        else:
            file_name = None
        _load_gs_weights(GSmodel, pretrained, file_name)
    return GSmodel


class ProxDRUNet(GSPnP):
    r"""
    Proximal Denoiser with DRUNet architecture.

    Proximal version of the Gradient Step Denoiser :class:`deepinv.models.GSDRUNet`, from :footcite:t:`hurault2022proximal`.
    Code from https://github.com/samuro95/Prox-PnP.

    The denoiser is a gradient step on a learned potential :math:`g_\sigma(x) = \frac{\alpha}{2} \|x - N_\sigma(x)\|^2`,
    where :math:`N_\sigma` is a DRUNet with Softplus activations, i.e.

    .. math::

        D_\sigma(x) = x - \nabla g_\sigma(x).

    The network is trained so that :math:`\nabla g_\sigma` has a Lipschitz constant smaller than 1, which makes
    :math:`D_\sigma` the proximal operator of a (weakly convex) regularization function, see :footcite:t:`hurault2022proximal`.
    The smooth Softplus activations make :math:`g_\sigma` twice differentiable, which is required for this property.

    :param float alpha: Relaxation parameter
    :param int in_channels: Number of input channels
    :param int out_channels: Number of output channels
    :param int nb: Number of blocks in the DRUNet
    :param Sequence[int,int,int,int] nc: number of channels per convolutional layer in the DRUNet. The network has a fixed number of 4 scales with ``nb`` blocks per scale (default: ``[64,128,256,512]``).
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU, "E" for ELU and "s" for Softplus (default).
    :param str pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the Prox-DRUNet weights will be downloaded from an
        online repository (only available for the default architecture with 3 input/output channels and Softplus activations).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param bool detach: If `True`, the denoiser output will be detached from the computation graph.
        Setting this to `False` allows one to compute the gradient of the denoiser output with respect to the input, it is necessary in training.
        Default is `True`.
    :param str device: gpu or cpu.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        in_channels: int = 3,
        out_channels: int = 3,
        nb: int = 2,
        nc=(64, 128, 256, 512),
        act_mode: str = "s",
        pretrained: str | None = None,
        detach: bool = True,
        device=torch.device("cpu"),
    ):
        from deepinv.models.drunet import DRUNet

        if pretrained == "download" and (
            in_channels != 3 or out_channels != 3 or act_mode != "s"
        ):
            raise ValueError(
                "The Prox-DRUNet pretrained weights are only available for color images "
                "(in_channels=out_channels=3) and Softplus activations (act_mode='s')."
            )

        denoiser = DRUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            nb=nb,
            nc=nc,
            act_mode=act_mode,
            pretrained=None,
            device=device,
        )
        super().__init__(denoiser, alpha=alpha, detach=detach)
        if pretrained:
            file_name = "Prox-DRUNet_torch.pth" if pretrained == "download" else None
            _load_gs_weights(self, pretrained, file_name)


def _load_gs_weights(model: GSPnP, pretrained: str, file_name: str | None) -> None:
    r"""
    Load pretrained weights into a :class:`deepinv.models.GSPnP` model and set it in eval mode.

    :param deepinv.models.GSPnP model: model to load the weights into.
    :param str pretrained: ``"download"`` to download ``file_name`` from the ``gradientstep`` Hugging Face repository,
        or a path to a local checkpoint.
    :param str file_name: name of the file to download when ``pretrained="download"``.
    """
    if pretrained == "download":
        url = get_weights_url(model_name="gradientstep", file_name=file_name)
        ckpt = load_state_dict_from_url(
            url,
            map_location=lambda storage, loc: storage,
            file_name=file_name,
        )
    else:
        ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)

    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(_convert_lightning_state_dict(ckpt), strict=True)
    model.eval()


def _convert_lightning_state_dict(state_dict: dict) -> dict:
    r"""
    Convert a state dict saved from the original Gradient Step / Prox-PnP Lightning code
    (keys ``student_grad.m_head.weight``, ...) to the key layout of :class:`deepinv.models.GSPnP`
    (keys ``student_grad.model.m_head.weight``, ...). Training metrics stored in the checkpoint are dropped.
    State dicts already in the deepinv layout are returned unchanged.
    """
    new_state_dict = {}
    for key, val in state_dict.items():
        if not key.startswith("student_grad."):
            continue  # e.g. train_PSNR.data_range
        if not key.startswith("student_grad.model."):
            key = "student_grad.model." + key[len("student_grad.") :]
        new_state_dict[key] = val
    return new_state_dict
