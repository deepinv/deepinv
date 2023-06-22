import torch.nn as nn
import torch
from .denoiser import online_weights_path


class DnCNN(nn.Module):
    r"""
    DnCNN convolutional denoiser.

    The architecture was introduced by Zhang et al. in https://arxiv.org/abs/1608.03981 and is composed of a series of
    convolutional layers with ReLU activation functions. The number of layers can be specified by the user.

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
        It is possible to download weights trained via the regularization method in https://epubs.siam.org/doi/abs/10.1137/20M1387961
        using ``pretrained='download_lipschitz'``.
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
    :param bool train: training or testing mode
    :param str device: gpu or cpu
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=20,
        bias=True,
        nf=64,
        pretrained="download",
        train=False,
        device=None,
    ):
        super(DnCNN, self).__init__()

        self.depth = depth

        self.in_conv = nn.Conv2d(
            in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias)
                for _ in range(self.depth - 2)
            ]
        )
        self.out_conv = nn.Conv2d(
            nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

        # if pretrain and ckpt_path is not None:
        #    self.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage), strict=True)

        if pretrained is not None:
            if pretrained.startswith("download"):
                name = ""
                if bias and depth == 20:
                    if pretrained == "download_lipschitz":
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
                    raise Exception(
                        "No pretrained weights were found online that match the chosen architecture"
                    )
                url = online_weights_path() + name
                ckpt = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt = torch.load(pretrained, map_location=lambda storage, loc: storage)
            self.load_state_dict(ckpt, strict=True)

        if not train:
            self.eval()
            for _, v in self.named_parameters():
                v.requires_grad = False

        if device is not None:
            self.to(device)

    def forward(self, x, sigma=None):
        r"""
        Run the denoiser on noisy image. The noise level is not used in this denoiser.

        :param torch.Tensor x: noisy image
        :param float sigma: noise level (not used)
        """
        x1 = self.in_conv(x)
        x1 = self.nl_list[0](x1)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x1)
            x1 = self.nl_list[i + 1](x_l)

        return self.out_conv(x1) + x
