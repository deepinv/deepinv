import torch
import torch.nn as nn


class PrimalBlock(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, depth=3, bias=True, nf=5):
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

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

    def forward(self, x, Atu, it):
        x_in = torch.cat((x, Atu), dim=1)

        x_ = self.in_conv(x_in)
        x_ = self.nl_list[0](x_)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x_)
            x_ = self.nl_list[i + 1](x_l)

        return self.out_conv(x_) + x


class PrimalBlock_list(nn.Module):
    def __init__(
        self, in_channels=2, out_channels=1, depth=3, bias=True, nf=5, max_it=5
    ):
        """
        TODO: add doc
        """
        super(PrimalBlock_list, self).__init__()

        self.depth = depth
        self.list_modules = nn.ModuleList(
            [
                PrimalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    bias=bias,
                    nf=nf,
                )
                for _ in range(max_it)
            ]
        )

    def forward(self, x, Atu, it):
        return self.list_modules[it](x, Atu, it)


class DualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=3, bias=True, nf=5):
        """
        TODO: add doc
        """
        super(DualBlock, self).__init__()

        self.depth = depth

        self.in_conv = nn.Conv1d(
            in_channels, nf, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(nf, nf, kernel_size=3, stride=1, padding=1, bias=bias)
                for _ in range(self.depth - 2)
            ]
        )
        self.out_conv = nn.Conv1d(
            nf, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

        self.nl_list = nn.ModuleList([nn.ReLU() for _ in range(self.depth - 1)])

    def forward(self, Ax_cur, u, y, denoise_level=None):
        Ax_cur = Ax_cur.unsqueeze(1)
        u = u.unsqueeze(1)
        y = y.unsqueeze(1)

        x_in = torch.cat((Ax_cur, u, y), dim=1)

        x_ = self.in_conv(x_in)
        x_ = self.nl_list[0](x_)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x_)
            x_ = self.nl_list[i + 1](x_l)

        x_out = self.out_conv(x_) + Ax_cur[:, 0:1, ...]

        return x_out[:, 0, ...]


class DualBlock_list(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1, depth=3, bias=True, nf=5, max_it=1
    ):
        """
        TODO: add doc
        """
        super(DualBlock_list, self).__init__()

        self.depth = depth
        self.list_modules = nn.ModuleList(
            [
                DualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    depth=depth,
                    bias=bias,
                    nf=nf,
                )
                for _ in range(max_it)
            ]
        )

    def forward(self, Ax_cur, u, y, it):
        return self.list_modules[it](Ax_cur, u, y)


class Toy(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=3, bias=True, nf=5):
        """
        TODO: add doc
        """
        super(Toy, self).__init__()

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

    def forward(self, x_in):
        x_ = self.in_conv(x_in)
        x_ = self.nl_list[0](x_)

        for i in range(self.depth - 2):
            x_l = self.conv_list[i](x_)
            x_ = self.nl_list[i + 1](x_l)

        return self.out_conv(x_) + x_in
