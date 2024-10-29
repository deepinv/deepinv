import torch
from deepinv.models.splines.spline_activation import WeaklyConvexSplineActivation
import numpy as np


class RidgeRegularizer(torch.nn.Module):
    def __init__(
        self,
        channel_sequence=[1, 4, 8, 80],
        kernel_size=5,
        max_noise_level=30.0 / 255.0,
        rho_wconvex=1.0,
        spline_knots=[11, 101],
    ):
        r"""
        Weakly Convex Ridge Regularizer

        Implementation of the `weakly convex ridge regularizer <https://epubs.siam.org/doi/10.1137/23M1565243>`_ based on the `official code <https://github.com/axgoujon/weakly_convex_ridge_regularizer>`_. The regularizer is defined as

        .. math::

            R(x)=\sum_{i=1}^N \psi_i(W_i)

        where :math:`W_i` are some convolutions and :math:`\psi_i` are some weakly convex activation functions parameterized by splines. In practice, the :math:`W_i` are realized by a concatenation of several convolutions without non-linearities, where the number of channels of these convolutions can be specified in the constructor.


        :param list of ints channel_sequence: number of channels for the convolutions
        :param int kernel size: kernel sizes for the convolutions
        :param float max_noise_level: maximum noise level where the model can be trained
        :param float rho_convex: modulus of weak convexity
        :param list of int spline_knots: spline_knots[0] is the number of knots of the scaling splines and spline_knots[1] is the number of knots for the potentials
        """
        super().__init__()
        # initialize splines
        self.potential = WeaklyConvexSplineActivation(
            channel_sequence[-1],
            scaling_knots=spline_knots[0],
            spline_knots=spline_knots[1],
            max_noise_level=max_noise_level * 255.0,
            rho_wconvex=rho_wconvex,
        )
        # initialize convolutions
        self.W = MultiConv2d(
            num_channels=channel_sequence,
            size_kernels=[kernel_size] * (len(channel_sequence) - 1),
        )

    def forward(self, x, sigma, tol=1e-4, max_iter=500):
        r"""
        Solve the denoising problem for the Weakly Convex Ridge Regularizer

        via an accelerated gradient descent. When called without torch.no_grad() this might require a large amount of memory.
        """
        x_noisy = x.clone()
        # initial value: noisy image
        z = torch.clone(x)
        t = 1
        # the index of the images that have not converged yet
        # relative change in the estimate
        res = 100000

        mu = torch.exp(
            self.potential.mu_spline(torch.tensor([[[[sigma * 255]]]], device=x.device))
        )
        step_size = 1 / (1 + mu)
        for i in range(max_iter):
            x_old = torch.clone(x)
            grad = self.grad(z, sigma) + (z - x_noisy)
            grad = grad * step_size

            x = z - grad

            t_old = t
            t = 0.5 * (1 + np.sqrt(1 + 4 * t**2))
            z = x + (t_old - 1) / t * (x - x_old)

            if i > 0:
                res_vec = torch.sqrt(
                    torch.sum((x - x_old).view(x.shape[0], -1) ** 2)
                    / torch.sum(x.view(x.shape[0], -1) ** 2)
                )
                res = torch.max(res_vec)

            if res < tol:
                break
        return x

    def cost(self, x, sigma):
        r"""
        Evaluates the regularizer itself.

        It is not efficient to use autograd on this function. Use the grad function instead.
        """
        if isinstance(sigma, float):
            sigma = sigma * torch.ones((x.size(0),), device=x.device) * 255

        return self.potential(self.W(x), sigma).sum(dim=tuple(range(1, len(x.shape))))

    def grad(self, x, sigma):
        r"""
        Evaluates the gradient of the regularizer
        """
        if isinstance(sigma, float):
            sigma = sigma * torch.ones((x.size(0),), device=x.device) * 255

        return self.W.transpose(self.potential.derivative(self.W(x), sigma))

    def load_state_dict(self, state_dict, **kwargs):
        super().load_state_dict(state_dict, **kwargs)
        self.potential.phi_plus.hyper_param_to_device()
        self.potential.phi_minus.hyper_param_to_device()
        self.W.spectral_norm(mode="power_method", n_steps=100)


class MultiConv2d(torch.nn.Module):
    def __init__(
        self, num_channels=[1, 64], size_kernels=[3], zero_mean=True, sn_size=256
    ):
        r"""
        The multiconv module for the ridge regularizer

        This module concatinates a sequence of convolutions without non-linearities in between

        The implementation is taken from `this paper <https://epubs.siam.org/doi/10.1137/23M1565243>`_ and can be found `here <https://github.com/axgoujon/weakly_convex_ridge_regularizer>`_.

        :param list of ints num_channels: num_channels[0]: number of input channels, num_channels[i>0]: number of output channels of the i-th convolution layer
        :param list of ints size_kernels: kernerl sizes for each convolution layer (len(size_kernels) = len(num_channels) - 1)
        :param bool zero_mean: the filters of convolutions are constrained to be of zero mean if true
        :param int sn_size: input image size for spectral normalization (required for training)
        """

        super().__init__()
        # parameters and options
        self.size_kernels = size_kernels
        self.num_channels = num_channels
        self.sn_size = sn_size
        self.zero_mean = zero_mean

        # list of convolutionnal layers
        self.conv_layers = torch.nn.ModuleList()

        for j in range(len(num_channels) - 1):
            self.conv_layers.append(
                torch.nn.Conv2d(
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
                torch.nn.utils.parametrize.register_parametrization(
                    self.conv_layers[-1], "weight", ZeroMean()
                )

        # cache the estimation of the spectral norm
        self.L = torch.tensor(1.0, requires_grad=True)
        # cache dirac impulse used to estimate the spectral norm
        self.padding_total = sum([kernel_size // 2 for kernel_size in size_kernels])
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
            x = torch.nn.functional.conv2d(
                x,
                weight,
                bias=None,
                dilation=conv.dilation,
                padding=conv.padding,
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
            x = torch.nn.functional.conv_transpose2d(
                x,
                weight,
                bias=None,
                padding=conv.padding,
                groups=conv.groups,
                dilation=conv.dilation,
                stride=conv.stride,
            )

        return x

    def spectral_norm(self, mode="Fourier", n_steps=1000):
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
            self.L = torch.tensor([1.0], device=self.conv_layers[0].weight.device)
            u = torch.empty(
                (1, 1, self.sn_size, self.sn_size),
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
        self.dirac = self.dirac.to(self.conv_layers[0].weight.device)
        return self.transpose(self.convolution(self.dirac))


# enforce zero mean kernels for each output channel
class ZeroMean(torch.nn.Module):
    def forward(self, X):
        Y = X - torch.mean(X, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return Y
