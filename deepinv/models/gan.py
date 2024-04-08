import torch.nn as nn
import numpy as np

# TODO all docstrings
class PatchGANDiscriminator(nn.Module):
    # Implementation taken from Kupyn et al., DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks https://openaccess.thecvf.com/content_cvpr_2018/papers/Kupyn_DeblurGAN_Blind_Motion_CVPR_2018_paper.pdf
    # Originally from pix2pix: Isola et al., Image-to-Image Translation with Conditional Adversarial Networks https://arxiv.org/abs/1611.07004
    def __init__(
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        use_sigmoid=False,
        batch_norm=True,
        bias=True,
    ):
        super().__init__()

        kw = 4  # kernel width
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=bias,
                ),
                nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=bias,
            ),
            nn.BatchNorm2d(ndf * nf_mult) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ESRGANDiscriminator(nn.Module):
    # Implementation taken from https://github.com/edongdongchen/EI/blob/main/models/discriminator.py
    # Originally from Wang et al., ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks https://arxiv.org/abs/1809.00219
    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
