import torch
import torch.nn as nn


class PrimalBlock(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, depth=3, bias=True, nf=5):
        """
        TODO: add doc
        """
        super(PrimalBlock, self).__init__()

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

        self.nl_list = nn.ModuleList([nn.PReLU() for _ in range(self.depth - 1)])

    def forward(self, x, Atu):
        x_in = torch.cat((x, Atu), dim=1)

        x_ = self.in_conv(x_in)
        x_ = self.nl_list[0](x_)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x_)
            x_ = self.nl_list[i + 1](x_l)

        return self.out_conv(x_) + x


class DualBlock(nn.Module):
    def __init__(self, in_channels=7, out_channels=3, depth=3, bias=True, nf=5):
        """
        TODO: add doc
        """
        super(DualBlock, self).__init__()

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

        self.nl_list = nn.ModuleList([nn.PReLU() for _ in range(self.depth - 1)])

    def forward(self, u, Ax_cur, y):
        x_in = torch.cat((u, Ax_cur, y), dim=1)

        x_ = self.in_conv(x_in)
        x_ = self.nl_list[0](x_)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x_)
            x_ = self.nl_list[i + 1](x_l)

        x_out = self.out_conv(x_) + Ax_cur

        return x_out
