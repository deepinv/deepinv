from __future__ import annotations
import torch.nn as nn
import torch
from .utils import get_weights_url, conv_nd, fix_dim, initialize_3d_from_2d
from .base import Denoiser


class DnCNN(Denoiser):
    r"""
    DnCNN convolutional denoiser.

    The architecture was introduced by :footcite:t:`zhang2017beyond` and is composed of a series of
    convolutional layers with ReLU activation functions. The number of layers can be specified by the user. Unlike the
    original paper, this implementation does not include batch normalization layers.

    The network can be initialized with pretrained weights, which can be downloaded from an online repository. The
    pretrained weights are trained with the default parameters of the network, i.e. 20 layers, 64 channels and biases.

    :param int in_channels: input image channels
    :param int out_channels: output image channels
    :param int depth: number of convolutional layers
    :param bool bias: use bias in the convolutional layers
    :param int nf: number of channels per convolutional layer
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for architecture with depth 20, 64 channels and biases).
        It is possible to download weights trained via the regularization method in :footcite:t:`pesquet2021learning`, using ``pretrained='download_lipschitz'``.
        When building a 3D network, it is possible to initialize with 2D pretrained weights by using ``pretrained='download_2d'`` or ``pretrained='download_lipschitz_2d'``, which provides a good starting point for fine-tuning.
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param bool pretrained_2d_isotropic: when loading 2D pretrained weights into a 3D network, whether to initialize the 3D kernels isotropically. By default the weights are loaded axially, i.e., by initializing the central slice of the 3D kernels with the 2D weights.
    :param torch.device, str device: Device to put the model on.
    :param str, int dim: Whether to build 2D or 3D network (if str, can be "2", "2d", "3D", etc.)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        depth: int = 20,
        bias: bool = True,
        nf: int = 64,
        pretrained: str | None = "download",
        pretrained_2d_isotropic: bool = False,
        device: torch.device | str = "cpu",
        dim: int | str = 2,
    ):
        super(DnCNN, self).__init__()

        dim = fix_dim(dim)

        conv = conv_nd(dim)

        self.depth = depth

        self.in_conv = conv(
            in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.conv_list = nn.ModuleList(
            [
                conv(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias)
                for _ in range(self.depth - 2)
            ]
        )
        self.out_conv = conv(
            nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        if pretrained is not None:
            if pretrained.startswith("download"):
                if dim == 3 and pretrained in (
                    "download",
                    "download_lipschitz",
                ):  # pragma: no cover
                    raise ValueError(
                        "No 3D weights for DnCNN are available for download. You can either initialize with 2D weights by using `download_2d` or `download_lipschitz_2d`, which provides a good starting point for fine-tuning, or set pretrained to None or path to your own pretrained weights."
                    )
                name = ""
                if bias and depth == 20:
                    if pretrained.startswith("download_lipschitz"):
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_lipschitz_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_lipschitz_gray.pth"
                    else:
                        if in_channels == 3 and out_channels == 3:
                            name = "dncnn_sigma2_color.pth"
                        elif in_channels == 1 and out_channels == 1:
                            name = "dncnn_sigma2_gray.pth"

                if name == "":
                    raise ValueError(
                        "No pretrained weights were found online that match the chosen architecture"
                    )
                url = get_weights_url(model_name="dncnn", file_name=name)
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)

            if dim == 3 and pretrained in ("download_2d", "download_lipschitz_2d"):
                initialize_3d_from_2d(self, ckpt, isotropic=pretrained_2d_isotropic)
            else:
                self.load_state_dict(ckpt, strict=True)
            self.eval()
        else:
            self.apply(weights_init_kaiming)

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, sigma=None) -> torch.Tensor:
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level (not used)

        .. note::
            The argument ``sigma`` is included for compatibility with the base class :class:`Denoiser <deepinv.models.base.Denoiser>` but is not used in this model.
        """
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)

        return self.out_conv(x1) + x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
