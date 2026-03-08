from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import Tensor, nn
from abc import ABC

from deepinv.physics import LinearPhysics
from .base import Reconstructor

ms = torch

class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """

    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
        # The value of the spline at any x is a combination
        # of at most two coefficients
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - step_size.item())

        floored_x = torch.floor((x_clamped - x_min) / step_size)  # left coefficient

        fracs = (x - x_min) / step_size - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = coefficients_vect[indexes + 1] * fracs + coefficients_vect[
            indexes
        ] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        # ctx.results = (fracs, coefficients_vect, indexes, grid)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients, indexes, step_size = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = (
            (coefficients_vect[indexes + 1] - coefficients_vect[indexes])
            / step_size
            * grad_out
        )

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(
            coefficients_vect, dtype=coefficients_vect.dtype
        )
        # right coefficients gradients

        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, (fracs * grad_out).view(-1)
        )
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1)
        )

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None


class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """

    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
        # The value of the spline at any x is a combination
        # of at most two coefficients
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - step_size.item())

        floored_x = torch.floor((x_clamped - x_min) / step_size)  # left coefficient

        fracs = (x - x_min) / step_size - floored_x  # distance to left coefficient

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)
        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        activation_output = (
            coefficients_vect[indexes + 1] - coefficients_vect[indexes]
        ) / step_size

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        fracs, coefficients, indexes, step_size = ctx.saved_tensors
        grad_x = 0 * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients.view(-1))
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, torch.ones_like(fracs).view(-1) / step_size
        )
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), -torch.ones_like(fracs).view(-1) / step_size
        )

        return grad_x, grad_coefficients_vect, None, None, None, None


class Quadratic_Spline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """

    @staticmethod
    def forward(ctx, x, coefficients, x_min, x_max, num_knots, zero_knot_indexes):
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - 2 * step_size.item())

        floored_x = torch.floor((x_clamped - x_min) / step_size)  # left

        # This gives the indexes (in coefficients_vect) of the left
        # coefficients
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        # B-Splines evaluation
        shift1 = (x - x_min) / step_size - floored_x

        frac1 = ((shift1 - 1) ** 2) / 2
        frac2 = (-2 * (shift1) ** 2 + 2 * shift1 + 1) / 2
        frac3 = (shift1) ** 2 / 2

        coefficients_vect = coefficients.view(-1)

        activation_output = (
            coefficients_vect[indexes + 2] * frac3
            + coefficients_vect[indexes + 1] * frac2
            + coefficients_vect[indexes] * frac1
        )

        grad_x = (
            coefficients_vect[indexes + 2] * (shift1)
            + coefficients_vect[indexes + 1] * (1 - 2 * shift1)
            + coefficients_vect[indexes] * ((shift1 - 1))
        )

        grad_x = grad_x / step_size

        ctx.save_for_backward(
            grad_x, frac1, frac2, frac3, coefficients, indexes, step_size
        )

        return activation_output

    @staticmethod
    def backward(ctx, grad_out):
        grad_x, frac1, frac2, frac3, coefficients, indexes, grid = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)

        grad_x = grad_x * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
        # coefficients gradients
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 2, (frac3 * grad_out).view(-1)
        )

        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, (frac2 * grad_out).view(-1)
        )

        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), (frac1 * grad_out).view(-1)
        )

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)

        return grad_x, grad_coefficients, None, None, None, None

class LinearSpline(ABC, nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_knots (int): number of knots of the spline
        num_activations (int) : number of activation functions
        x_min (float): position of left-most knot
        x_max (float): position of right-most knot
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
        antisymmetric (bool): Constrain the potential to be symmetric <=> activation antisymmetric
    """

    def __init__(
        self,
        num_activations,
        num_knots,
        x_min,
        x_max,
        init,
        slope_max=None,
        slope_min=None,
        clamp=True,
        **kwargs,
    ):

        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.x_min = torch.tensor([x_min])
        self.x_max = torch.tensor([x_max])
        self.slope_min = slope_min
        self.slope_max = slope_max

        self.step_size = (self.x_max - self.x_min) / (self.num_knots - 1)
        self.clamp = clamp
        self.no_constraints = slope_max is None and slope_min is None and not clamp

        # parameters
        coefficients = self.initialize_coeffs()  # spline coefficients
        self.coefficients = nn.Parameter(coefficients)

        self.projected_coefficients_cached = None
        self.register_buffer(
            "D2_filter", Tensor([1, -2, 1]).view(1, 1, 3).div(self.step_size)
        )

        self.init_zero_knot_indexes()

    def init_zero_knot_indexes(self):
        """Initialize indexes of zero knots of each activation."""
        # self.zero_knot_indexes[i] gives index of knot 0 for filter/neuron_i.
        # size: (num_activations,)
        activation_arange = torch.arange(0, self.num_activations)
        self.zero_knot_indexes = activation_arange * self.num_knots

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since B1 splines are interpolators)."""
        init = self.init
        grid_tensor = torch.linspace(
            self.x_min.item(), self.x_max.item(), self.num_knots
        ).expand((self.num_activations, self.num_knots))

        if isinstance(init, float):
            coefficients = torch.ones_like(grid_tensor) * init
        elif init == "gaussian":
            coefficients = torch.exp(-(grid_tensor**2))
        elif init == "identity":
            coefficients = grid_tensor
        elif init == "zero":
            coefficients = grid_tensor * 0
        else:
            raise ValueError("init should be in [identity, zero].")

        return coefficients

    @property
    def projected_coefficients(self):
        """B-spline coefficients projected to meet the constraint."""
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        else:
            return self.clipped_coefficients()

    def cached_projected_coefficients(self):
        """B-spline coefficients projected to meet the constraint."""
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self.clipped_coefficients()

    @property
    def slopes(self):
        """Get the slopes of the activations"""
        coeff = self.projected_coefficients
        slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.step_size

        return slopes

    @property
    def device(self):
        return self.coefficients.device

    def hyper_param_to_device(self):
        device = self.device
        self.x_min, self.x_max, self.step_size, self.zero_knot_indexes = (
            self.x_min.to(device),
            self.x_max.to(device),
            self.step_size.to(device),
            self.zero_knot_indexes.to(device),
        )

    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on weather the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        self.hyper_param_to_device()

        in_shape = x.shape
        in_channels = in_shape[1]

        if in_channels % self.num_activations != 0:
            raise ValueError(
                "Number of input channels must be divisible by number of activations."
            )

        x = x.view(
            x.shape[0],
            self.num_activations,
            in_channels // self.num_activations,
            *x.shape[2:],
        )

        x = LinearSpline_Func.apply(
            x,
            self.projected_coefficients,
            self.x_min,
            self.x_max,
            self.num_knots,
            self.zero_knot_indexes,
        )

        x = x.view(in_shape)

        return x

    def extra_repr(self):
        """repr for print(model)"""

        s = (
            "num_activations={num_activations}, "
            "init={init}, num_knots={num_knots}, range=[{x_min[0]:.3f}, {x_max[0]:.3f}], "
            "slope_max={slope_max}, "
            "slope_min={slope_min}."
        )

        return s.format(**self.__dict__)

    def clipped_coefficients(self):
        """Simple projection of the spline coefficients to enforce the constraints, for e.g. bounded slope"""

        device = self.device

        if self.no_constraints:
            return self.coefficients

        cs = self.coefficients

        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.step_size

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(new_slopes, self.slope_min, self.slope_max)

        # clamp extension
        if self.clamp:
            new_slopes[:, 0] = 0
            new_slopes[:, -1] = 0

        new_cs = torch.zeros(self.coefficients.shape, device=device, dtype=cs.dtype)

        new_cs[:, 1:] = torch.cumsum(new_slopes, dim=1) * self.step_size

        new_cs = new_cs + (cs - new_cs).mean(dim=1).unsqueeze(1)

        return new_cs

    @property
    def relu_slopes(self):
        """Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        return F.conv1d(
            self.projected_coefficients.unsqueeze(1), self.D2_filter
        ).squeeze(1)

    def tv2(self):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1).sum()


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
    def __init__(
        self,
        num_channels=None,
        size_kernels=None,
        zero_mean: bool = True,
        sn_size: int = 256,
        color: bool = False,
    ) -> None:
        """
        Multi-layer convolution with spectral norm control.

        Parameters
        ----------
        num_channels : list[int], optional
            Number of channels per layer. Defaults to [1, 64].
        size_kernels : list[int], optional
            Kernel sizes per layer. Defaults to [3].
        zero_mean : bool, optional
            If True, enforce zero-mean filters for the first conv layer.
        sn_size : int, optional
            Spatial size used for spectral-norm estimation.
        color : bool, optional
            If True, use 3-channel (color) version; otherwise grayscale.
        """
        super().__init__()

        if num_channels is None:
            num_channels = [1, 64]
        if size_kernels is None:
            size_kernels = [3]

        # parameters and options
        self.size_kernels = size_kernels
        self.num_channels = num_channels
        self.sn_size = sn_size
        self.zero_mean = zero_mean
        self.padding = self.size_kernels[0] // 2
        self.color = color

        # list of convolutional layers
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
        self.padding_total = sum(kernel_size // 2 for kernel_size in size_kernels)

        if color:
            self.dirac = torch.zeros(
                (1, 3, 4 * self.padding_total + 1, 4 * self.padding_total + 1)
            )
            self.dirac[0, 1, 2 * self.padding_total, 2 * self.padding_total] = 1
        else:
            self.dirac = torch.zeros(
                (1, 1, 4 * self.padding_total + 1, 4 * self.padding_total + 1)
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


class _DEALImpl(nn.Module):
    def __init__(self, color: bool) -> None:
        super().__init__()

        self.kernel_size = 9
        self.conv_pad = self.kernel_size // 2
        self.color = color

        if self.color:
            self.n = 3
            channels = [3, 12, 24, 128]
        else:
            self.n = 1
            channels = [1, 4, 8, 128]

        self.last_c = channels[-1]
        self.W1 = MultiConv2d(
            channels, [self.kernel_size] * (len(channels) - 1), color=self.color
        )

        self.M1 = MultiConv2d(
            channels, [self.kernel_size] * (len(channels) - 1), color=self.color
        )
        self.M2 = nn.Conv2d(
            self.last_c, self.last_c, kernel_size=3, padding=1, bias=False, groups=1
        )
        self.M3 = nn.Conv2d(
            self.last_c, self.last_c, kernel_size=3, padding=1, bias=False, groups=1
        )

        self.spline1 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="identity",
            clamp=False,
            slope_min=0,
        )
        self.spline2 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="identity",
            clamp=False,
            slope_min=0,
        )
        self.spline3 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="gaussian",
            clamp=False,
        )

        self.spline_lambda = LinearSpline(
            num_activations=1,
            num_knots=53,
            x_min=-1,
            x_max=51,
            init="identity",
            clamp=False,
        )
        self.spline_scaling = LinearSpline(
            num_activations=self.last_c,
            num_knots=14,
            x_min=-1,
            x_max=51,
            init=3.0,
            clamp=False,
        )

        self.number_of_cgs = 0
        self.last_cg_iter = 0
        self.max_iter = 1000


    def cal_lambda(self, sigma):
        self.lmbda = self.spline_lambda(sigma)

    def cal_scaling(self, sigma):
        sigma = torch.ones((sigma.size(0), self.last_c, 1, 1)).to(sigma.device) * sigma
        self.scaling = torch.exp(self.spline_scaling(sigma)) / (sigma + 1e-5)

    def last_act(self, x):
        x = torch.abs(x)
        x = self.spline3(self.scaling * x)
        return torch.clip(x, 1e-2, 1)

    def K(self, x, idx=None):
        return torch.sqrt(self.lmbda) * self.W1(x) * self.mask[idx]

    def Kt(self, y, idx=None):
        return torch.sqrt(self.lmbda) * self.W1.transpose(y * self.mask[idx])

    def KtK(self, x, idx=None):
        return self.Kt(self.K(x, idx), idx)

    def cal_mask(self, x):
        self.mask = self.last_act(
            self.M3(
                self.spline2(torch.abs(self.M2(self.spline1(torch.abs(self.M1(x))))))
            )
        )
        return self.mask

    def L(self, x, idx="None"):
        return self.W1(x) * self.mask[idx]

    def Lt(self, y, idx=None):
        return self.W1.transpose(y * self.mask[idx])

    def BtB(self, x, H, Ht, idx=None):
        BtBD = (Ht(H(x)) + self.lmbda[idx] * self.Lt(self.L(x, idx), idx)) / (
            1 + self.lmbda[idx]
        )
        return BtBD

    def cg_sample(self, b, x0, max_iter, eps=1e-5):
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        b = self.Kt(b, correct_idx_mask)
        r = b - self.KtK(x, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = list()
        idx_uniques_cont = list()
        idx_uniques_cont.append([i for i in range(x.size(0))])

        for i in range(max_iter):

            idx_cont = torch.where(r_norm.squeeze() > eps)[0].tolist()

            if i == max_iter - 1:
                idx_cont = []

            len_new = len(idx_cont)

            if len_new != len_old:
                idx_done = torch.where(r_norm.squeeze() <= eps)[0].tolist()

                if i == max_iter - 1:
                    idx_done = [h for h in range(x.size(0))]
                idx_uniques_done.append(idx_done)

                correct_idx = [idx_uniques_cont[-1][id] for id in idx_done]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx = [idx_uniques_cont[-j - 2][id] for id in correct_idx]

                correct_idx_mask = [idx_uniques_cont[-1][id] for id in idx_cont]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx_mask = [
                        idx_uniques_cont[-j - 2][id] for id in correct_idx_mask
                    ]

                output[correct_idx] = x[idx_done]
                idx_uniques_cont.append(idx_cont)
                r = r[idx_cont]
                p = p[idx_cont]
                x = x[idx_cont]
                r_norm = r_norm[idx_cont]
                len_old = len_new

            if len(idx_cont) == 0:
                break

            BTBp = self.KtK(p, correct_idx_mask)
            alpha = r_norm / ((p * BTBp).sum(dim=(1, 2, 3), keepdim=True))

            x = x + alpha * p
            r_norm_old = r_norm.clone()
            r = r - alpha * BTBp

            r_norm = (r**2).sum(dim=(1, 2, 3), keepdim=True)
            beta = r_norm / (r_norm_old)
            p = r + beta * p

        return output, i

    def cg(
        self, b, x0, max_iter, eps=1e-5, H=lambda x: x, Ht=lambda x: x, dims=(1, 2, 3)
    ):

        b = b / (1 + self.lmbda)
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        r = b - self.BtB(x, H, Ht, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = list()
        idx_uniques_cont = list()
        idx_uniques_cont.append([i for i in range(x.size(0))])

        for i in range(max_iter):

            idx_cont = torch.where(r_norm.squeeze() > eps)[0].tolist()

            if i == max_iter - 1:
                idx_cont = []

            len_new = len(idx_cont)

            if len_new != len_old:
                idx_done = torch.where(r_norm.squeeze() <= eps)[0].tolist()

                if i == max_iter - 1:
                    idx_done = [h for h in range(x.size(0))]
                idx_uniques_done.append(idx_done)

                correct_idx = [idx_uniques_cont[-1][id] for id in idx_done]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx = [idx_uniques_cont[-j - 2][id] for id in correct_idx]

                correct_idx_mask = [idx_uniques_cont[-1][id] for id in idx_cont]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx_mask = [
                        idx_uniques_cont[-j - 2][id] for id in correct_idx_mask
                    ]

                output[correct_idx] = x[idx_done]
                idx_uniques_cont.append(idx_cont)
                r = r[idx_cont]
                p = p[idx_cont]
                x = x[idx_cont]
                r_norm = r_norm[idx_cont]
                len_old = len_new

            if len(idx_cont) == 0:
                break

            BTBp = self.BtB(p, H, Ht, correct_idx_mask)
            alpha = r_norm / ((p * BTBp).sum(dim=(1, 2, 3), keepdim=True))

            x = x + alpha * p
            r_norm_old = r_norm.clone()
            r = r - alpha * BTBp

            r_norm = (r**2).sum(dim=(1, 2, 3), keepdim=True)
            beta = r_norm / (r_norm_old)
            p = r + beta * p

        return output, i

    def denoise(self, y, sigma):

        self.W1.spectral_norm()
        self.cal_lambda(sigma)
        self.cal_scaling(sigma)

        if self.training:
            self.c_k_list = list()
            grad_steps = 1
            n_out = torch.randint(14, 59, (1, 1))
            n_in = 50
            eps_in = 1e-4
            eps_out = 1e-4
            eps_bck = 1e-4

        else:
            grad_steps = 0
            n_out = 60
            n_in = 200
            eps_in = 1e-6
            eps_out = 1e-5

        c_k = torch.zeros_like(y)
        c_k_old = c_k.clone()

        with torch.no_grad():
            for id in range(n_out):
                self.cal_mask(c_k)
                c_k, self.last_cg_iter = self.cg(y, c_k_old, n_in, eps=eps_in)
                res = torch.linalg.norm(c_k - c_k_old) / torch.linalg.norm(c_k_old)
                c_k_old = c_k.clone()
                if (res < eps_out).all():
                    break

        def backward_hook1(grad):
            self.cal_mask(d_k)
            g, _ = self.cg(grad, grad, n_in, eps=eps_bck)
            return g

        if self.training:
            d_k = c_k
            self.cal_mask(d_k)
            self.c_k_list.append(d_k)
            with torch.no_grad():
                c_k, self.last_cg_iter = self.cg(y, c_k, n_in, eps=eps_bck)
            idx = [i for i in range(y.size(0))]
            c_k1 = y - self.lmbda * self.Lt(self.L(c_k.detach(), idx), idx)
            c_k1.register_hook(backward_hook1)
            self.c_k_list.append(c_k1)

        else:
            c_k1 = c_k

        self.number_of_cgs = id + grad_steps

        return c_k1

    def solve_inverse_problem(
        self,
        y,
        H,
        Ht,
        sigma,
        lmbda,
        eps_in=1e-8,
        eps_out=1e-5,
        path=False,
        x_init=None,
        verbose=False,
    ):

        self.W1.spectral_norm()
        self.cal_scaling(torch.tensor([[sigma]]).to(y.device))
        self.lmbda = torch.tensor([[lmbda]]).to(y.device)
        if path:
            c_ks = list()

        # NEW: use self.max_iter to control iterations
        max_iters = getattr(self, "max_iter", 1000)
        max_cg_iters = getattr(self, "max_iter", 1000)

        with torch.no_grad():
            if x_init is not None:
                c_k = x_init
            else:
                c_k = Ht(y) * 0
            c_k_old = c_k.clone()

            for m in range(max_iters):  # instead of range(1000)
                if path:
                    c_ks.append(c_k)

                self.cal_mask(c_k)
                c_k, cg_iters = self.cg(
                    Ht(y),
                    c_k_old,
                    max_cg_iters,  # instead of 1000
                    eps=eps_in,
                    H=H,
                    Ht=Ht,
                )

                res = torch.linalg.norm(c_k - c_k_old) / torch.linalg.norm(c_k_old)

                c_k_old = c_k.clone()

                if verbose:
                    print(
                        "CG Number:",
                        m,
                        "CG iterations:",
                        cg_iters,
                        "Outer residual:",
                        res,
                    )

                if (res < eps_out).all():
                    break

        if path:
            c_ks.append(c_k)
            return torch.clip(c_k, 0, 1), c_ks
        else:
            return torch.clip(c_k, 0, 1)


class DEAL(Reconstructor):
    """
    Deep Equilibrium Attention Least Squares (DEAL) reconstruction model.

    This model solves linear inverse problems using a learned equilibrium-based
    regularizer combined with conjugate gradient iterations. It can be used for
    image restoration and reconstruction tasks such as denoising, deblurring,
    and computed tomography reconstruction.

    This implementation is adapted from the official DEAL repository:
    https://github.com/mehrsapo/DEAL

    For the original method, see :footcite:t:`pourya2025dealing`.

    Parameters
    ----------
    pretrained : str, optional
        Path to a pretrained DEAL checkpoint file or "download" to automatically
        download the official pretrained weights.
    sigma : float, optional
        Noise level parameter expected by the DEAL model (default: 25.0).
    lam : float, optional
        Regularisation strength (lambda) used by the DEAL solver (default: 10.0).
    max_iter : int, optional
        Maximum number of outer iterations in the inverse solver (default: 50).
    auto_scale : bool, optional
        If True, rescale the measurements ``y`` so that their standard
        deviation is close to ``target_y_std`` (default: False).
    target_y_std : float, optional
        Target standard deviation used for automatic scaling (default: 25.0).
    color : bool, optional
        If True, use the color version of DEAL (three channels). If False,
        use the grayscale version (one channel). This must match the
        pretrained and the data (default: False).
    device : {"cuda", "cpu"} or None, optional
        Device used for computations. If None, "cuda" is used when available,
        otherwise "cpu".
    clamp_output : bool, optional
        If True, clamp the reconstructed image to the range [0, 1]
        before returning (default: True).
    """

    def __init__(
        self,
        pretrained: str,
        sigma: float = 25.0,
        lam: float = 10.0,
        max_iter: int = 50,
        auto_scale: bool = False,
        target_y_std: float = 25.0,
        color: bool = False,
        device: str | None = None,
        clamp_output: bool = True,
    ) -> None:

        super().__init__()

        # Device selection
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Solver parameters
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.auto_scale = bool(auto_scale)
        self.target_y_std = float(target_y_std)
        self.clamp_output = bool(clamp_output)

        # Underlying DEAL model from the official package
        self.model = _DEALImpl(color=color).to(self.device).eval()

        
        # Load pretrained weights
        if pretrained == "download":
            if color:
                url = "https://raw.githubusercontent.com/mehrsapo/DEAL/main/trained_models/deal_color.pth"
            else:
                url = "https://raw.githubusercontent.com/mehrsapo/DEAL/main/trained_models/deal_gray.pth"
            state = torch.hub.load_state_dict_from_url(
                url,
                map_location=self.device,
                file_name=url.split("/")[-1],
            )
        else:
            try:
                state = torch.load(
                    pretrained, map_location=self.device, weights_only=True
                )
            except TypeError:
                state = torch.load(pretrained, map_location=self.device)
                
        self.model.load_state_dict(state["state_dict"])
            

    @torch.no_grad()
    def forward(self, y: torch.Tensor, physics: LinearPhysics) -> torch.Tensor:
        """
        Run the DEAL reconstruction.

        Parameters
        ----------
        y : torch.Tensor
            Measurements (e.g. sinogram).
        physics : deepinv.physics.LinearPhysics
            DeepInverse physics operator with ``__call__`` and ``A_adjoint``.

        Returns
        -------
        torch.Tensor
            Reconstructed image with the same spatial shape as ``H^T y``.
        """
        # Move data to the correct device
        y = y.to(self.device)

        # Forward and adjoint operators as callables
        H = lambda z: physics(z)
        Ht = physics.A_adjoint

        # Optional automatic scaling of y
        if self.auto_scale:
            y_std = float(y.std().detach().cpu())
            if 0.0 < y_std < 5.0:
                scale = self.target_y_std / (y_std + 1e-12)
                y = y * scale

        # Zero initialisation
        x_init = torch.zeros_like(Ht(y))

        # Set number of outer iterations on the underlying DEAL model
        if hasattr(self.model, "max_iter"):
            self.model.max_iter = max(int(self.max_iter), 1)

        # Call the official DEAL solver
        x_hat = self.model.solve_inverse_problem(
            y,
            H=H,
            Ht=Ht,
            sigma=self.sigma,
            lmbda=self.lam,
            x_init=x_init,
            verbose=False,
            path=False,
        )

        return x_hat.clamp(0.0, 1.0) if self.clamp_output else x_hat
