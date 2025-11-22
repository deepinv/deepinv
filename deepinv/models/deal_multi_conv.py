import torch
from torch import nn
import torch.nn.utils.parametrize as P

ms = torch


def symmetric_pad(tensor: ms.Tensor, pad_size) -> ms.Tensor:
    """
    Pads symmetrically the spatial dimensions (last two dimensions) of the input tensor.

    :param tensor: input of shape [..., H, W] to be padded
    :return: padded tensor of shape [..., H + pad_h, W + pad_w]
    """
    sz = tensor.shape
    sz = [sz[0], sz[1], sz[2], sz[3]]
    sz[-1] = sz[-1] + sum(pad_size[2::])
    sz[-2] = sz[-2] + sum(pad_size[0:2])
    out = ms.empty(sz, dtype=tensor.dtype).to(tensor.device)
    # Copy the original tensor to the central part
    out[
        ...,
        pad_size[0] : out.size(-2) - pad_size[1],
        pad_size[2] : out.size(-1) - pad_size[3],
    ] = tensor
    # Pad Top
    if pad_size[0] != 0:
        out[..., 0 : pad_size[0], :] = ms.flip(
            out[..., pad_size[0] : 2 * pad_size[0], :], (-2,)
        )
    # Pad Bottom
    if pad_size[1] != 0:
        out[..., out.size(-2) - pad_size[1] : :, :] = ms.flip(
            out[..., out.size(-2) - 2 * pad_size[1] : out.size(-2) - pad_size[1], :],
            (-2,),
        )
    # Pad Left
    if pad_size[2] != 0:
        out[..., :, 0 : pad_size[2]] = ms.flip(
            out[..., :, pad_size[2] : 2 * pad_size[2]], (-1,)
        )
    # Pad Right
    if pad_size[3] != 0:
        out[..., :, out.size(-1) - pad_size[3] : :] = ms.flip(
            out[..., :, out.size(-1) - 2 * pad_size[3] : out.size(-1) - pad_size[3]],
            (-1,),
        )
    return out


def symmetric_pad_transpose(tensor: ms.Tensor, pad_size) -> ms.Tensor:
    """
    Adjoint of the SymmetricPad2D operation which amounts to a special type of cropping.

    :param tensor: input of shape [..., H, W] to be cropped
    :return: cropped tensor of shape [..., H - pad_h, W - pad_w]
    """
    sz = list(tensor.size())
    out = tensor.clone()
    # Top
    if pad_size[0] != 0:
        out[..., pad_size[0] : 2 * pad_size[0], :] += ms.flip(
            out[..., 0 : pad_size[0], :], (-2,)
        )
    # Bottom
    if pad_size[1] != 0:
        out[..., -2 * pad_size[1] : -pad_size[1], :] += ms.flip(
            out[..., -pad_size[1] : :, :], (-2,)
        )
        # Left
    if pad_size[2] != 0:
        out[..., pad_size[2] : 2 * pad_size[2]] += ms.flip(
            out[..., 0 : pad_size[2]], (-1,)
        )
    # Right
    if pad_size[3] != 0:
        out[..., -2 * pad_size[3] : -pad_size[3]] += ms.flip(
            out[..., -pad_size[3] : :], (-1,)
        )
    if pad_size[1] == 0:
        end_h = sz[-2] + 1
    else:
        end_h = sz[-2] - pad_size[1]

    if pad_size[3] == 0:
        end_w = sz[-1] + 1
    else:
        end_w = sz[-1] - pad_size[3]
    out = out[..., pad_size[0] : end_h, pad_size[2] : end_w]
    return out


class MultiConv2d(nn.Module):
    class MultiConv2d(nn.Module):
        def __init__(
                self,
                num_channels=None,
                size_kernels=None,
                zero_mean=True,
                sn_size=256,
                color=True,
        ):
            if num_channels is None:
                num_channels = [1, 64]
            if size_kernels is None:
                size_kernels = [3]

        """ """

        super().__init__()
        # parameters and options
        self.size_kernels = size_kernels
        self.num_channels = num_channels
        self.sn_size = sn_size
        self.zero_mean = zero_mean
        self.padding = self.size_kernels[0] // 2
        self.color = color

        # list of convolutionnal layers
        self.conv_layers = nn.ModuleList()

        for j in range(len(num_channels) - 1):
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=num_channels[j],
                    out_channels=num_channels[j + 1],
                    kernel_size=size_kernels[j],
                    padding=size_kernels[j] // 2,
                    stride=1,
                    bias=False,
                )
            )
            # enforce zero mean filter for first conv
            if zero_mean and j == 0:
                P.register_parametrization(self.conv_layers[-1], "weight", ZeroMean())

        # cache the estimation of the spectral norm
        self.L = torch.tensor(1.0, requires_grad=True)
        # cache dirac impulse used to estimate the spectral norm
        self.padding_total = sum([kernel_size // 2 for kernel_size in size_kernels])

        if color:
            self.dirac = torch.zeros(
                (1, 3) + (4 * self.padding_total + 1, 4 * self.padding_total + 1)
            )
            self.dirac[0, 1, 2 * self.padding_total, 2 * self.padding_total] = 1
        else:
            self.dirac = torch.zeros(
                (1, 1) + (4 * self.padding_total + 1, 4 * self.padding_total + 1)
            )
            self.dirac[0, 0, 2 * self.padding_total, 2 * self.padding_total] = 1

    def forward(self, x):
        return self.convolution(x)

    def convolution(self, x):
        # normalized convolution, so that the spectral norm of the convolutional kernel is 1
        # nb the spectral norm of the convolution has to be upated before
        x = x / torch.sqrt(self.L)

        for conv in self.conv_layers:
            weight = conv.weight
            # x = symmetric_pad(x, (self.padding, self.padding, self.padding, self.padding))
            x = nn.functional.conv2d(
                x,
                weight,
                bias=None,
                dilation=conv.dilation,
                padding=self.padding,
                groups=conv.groups,
                stride=conv.stride,
            )

        return x

    def transpose(self, x):
        # normalized transpose convolution, so that the spectral norm of the convolutional kernel is 1
        # nb the spectral norm of the convolution has to be upated before
        x = x / torch.sqrt(self.L)

        for conv in reversed(self.conv_layers):
            weight = conv.weight
            x = nn.functional.conv_transpose2d(
                x,
                weight,
                bias=None,
                padding=self.padding,
                groups=conv.groups,
                dilation=conv.dilation,
                stride=conv.stride,
            )
            # x = symmetric_pad_transpose(x, (self.padding, self.padding, self.padding, self.padding))

        return x

    def spectral_norm(self, mode="Fourier", n_steps=50):
        """Compute the spectral norm of the convolutional layer
        Args:
            mode: "Fourier" or "power_method"
                - "Fourier" computes the spectral norm by computing the DFT of the equivalent convolutional kernel. This is only an estimate (boundary effects are not taken into account) but it is differentiable and fast
                - "power_method" computes the spectral norm by power iteration. This is more accurate and used before testing
            n_steps: number of steps for the power method
        """

        if mode == "Fourier":
            # temporary set L to 1 to get the spectral norm of the unnormalized filter
            self.L = torch.tensor([1.0], device=self.conv_layers[0].weight.device)
            # get the convolutional kernel corresponding to WtW
            kernel = self.get_kernel_WtW()
            # pad the kernel and compute its DFT. The spectral norm of WtW is the maximum of the absolute value of the DFT
            padding = (self.sn_size - 1) // 2 - self.padding_total
            if self.color:
                fft_kernel = torch.fft.fft2(
                    torch.nn.functional.pad(
                        kernel, (padding, padding, padding, padding)
                    )
                ).abs()
                self.L = (
                    fft_kernel[:, 0].max()
                    + fft_kernel[:, 1].max()
                    + fft_kernel[:, 2].max()
                )
            else:
                self.L = (
                    torch.fft.fft2(
                        torch.nn.functional.pad(
                            kernel, (padding, padding, padding, padding)
                        )
                    )
                    .abs()
                    .max()
                )

            return self.L

        elif mode == "power_method":
            if self.color:
                n = 3
            else:
                n = 1

            self.L = torch.tensor([1.0], device=self.conv_layers[0].weight.device)
            u = torch.empty(
                (1, n, self.sn_size, self.sn_size),
                device=self.conv_layers[0].weight.device,
            ).normal_()
            with torch.no_grad():
                for _ in range(n_steps):
                    u = self.transpose(self.convolution(u))
                    u = u / torch.linalg.norm(u)

                # The largest eigen value can now be estimated in a differentiable way
                sn = torch.linalg.norm(self.transpose(self.convolution(u)))
                self.L = sn
                return sn

    def check_tranpose(self):
        """
        Check that the convolutional layer is indeed the transpose of the convolutional layer
        """
        device = self.conv_layers[0].weight.device

        for i in range(1):
            x1 = torch.empty((1, 1, 40, 40), device=device).normal_()
            x2 = torch.empty(
                (1, self.num_channels[-1], 40, 40), device=device
            ).normal_()

            ps_1 = (self(x1) * x2).sum()
            ps_2 = (self.transpose(x2) * x1).sum()
            print(f"ps_1: {ps_1.item()}")
            print(f"ps_2: {ps_2.item()}")
            print(f"ratio: {ps_1.item() / ps_2.item()}")

    def spectrum(self):
        kernel = self.get_kernel_WtW()
        padding = (self.sn_size - 1) // 2 - self.padding_total
        return torch.fft.fft2(
            torch.nn.functional.pad(kernel, (padding, padding, padding, padding))
        )

    def get_filters(self):
        # we collapse the convolutions to get one kernel per channel
        # this done by computing the response of a dirac impulse
        self.dirac = self.dirac.to(self.conv_layers[0].weight.device)
        kernel = self.convolution(self.dirac)[
            :,
            :,
            self.padding_total : 3 * self.padding_total + 1,
            self.padding_total : 3 * self.padding_total + 1,
        ]
        return kernel

    def get_kernel_WtW(self):
        self.dirac = self.dirac.to(self.conv_layers[0].weight.device).type(
            self.conv_layers[0].weight.dtype
        )
        return self.transpose(self.convolution(self.dirac))


# enforce zero mean kernels for each output channel
class ZeroMean(nn.Module):
    def forward(self, X):
        Y = X - torch.mean(X, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return Y
