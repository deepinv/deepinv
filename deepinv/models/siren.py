import math
import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm
from .base import Reconstructor
from deepinv.utils.decorators import _deprecated_alias
from deepinv.loss.mc import MCLoss

# Adapted from https://github.com/vsitzmann/siren by V. Sitzmann, J. N.P. Martel and A. W. Bergman, 
# and https://github.com/TheoHanon/Spherical-Implicit-Neural-Representation by T. Hanon.


def get_mgrid(shape):
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    :param tuple shape: The shape of the grid to generate. E.g., (H, W) for a 2D grid.
    """
    tensors = tuple([torch.linspace(-1, 1, steps=steps) for steps in shape])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(shape))
    return mgrid


def nabla(
    y: torch.Tensor, x: torch.Tensor, grad_outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Computes the vector-Jacobian-product (VJP) of y w.r.t. x on the direction of grad_outputs.

    Each element of the VJP is computed as :math:`\sum_{i=1}^m \frac{\partial y_i}{\partial x_k} \cdot v_i`, where :math:`x_k` is the k-th element of `x`, :math:`y_i` is the i-th element of `y` and :math:`v_i` is the i-th element of `grad_outputs`.

    By default, `grad_outputs` is a tensor of ones with the same shape as `y`. The default behavior on a scalar `y` thus computes the nabla (gradient) of `y` w.r.t. `x`.

    :param torch.Tensor y: The output tensor.
    :param torch.Tensor x: The input tensor.
    :param Optional[torch.Tensor] grad_outputs: The direction of the VJP. If None, a tensor of ones with the same shape as `y` is used.
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class TVPrior(nn.Module):
    r"""
    Total variation (TV) prior :math:`\reg{x} = \| D x \|_{1}`.

    The TV prior is computed using the continuous definition of the gradient, i.e., :math:`D x = \nabla x`.
    """

    def forward(self, y, x):
        y = torch.mean(nabla(y, x).abs())
        return y


class FourierPE(nn.Module):
    r"""
    Fourier Positional Encoding.

    Maps the input :math:`z`onto the Fourier domain by applying a learnable linear map followed by a sinusoidal activation function.

    For an input :math:`z`, the encoding :math:`\psi(z)` is given by

    .. math::
        \psi(z) = \sin\bigl(\omega_0\,\Omega\,z\bigr),

    where :math:`\Omega:\mathbb{R}^{\text{input_dim}} \rightarrow \mathbb{R}^{\text{output_dim}}` is a learnable weight matrix encoding frequencies and :math:`\omega_0` is a fixed frequency factor.

    :param int input_dim: Dimensionality of the input.
    :param int output_dim: Number of output features (atoms).
    :param bool bias: If True, the linear mapping includes a bias term. Default is True.
    :param float omega0: Frequency factor applied to the sinusoidal activation. Default is 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super().__init__()
        self.omega0 = omega0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias)

        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return torch.sin(self.omega0 * x)


class Sin(nn.Module):
    r"""
    Sine activation function with frequency scaling.

    Applies the sine activation function with a frequency scaling factor :math:`\omega_0`:

    .. math::
        \text{sin}(x) = \sin(\omega_0 \, x)

    :param float omega0: Frequency scaling factor.
    """

    def __init__(self, omega0: float = 1.0) -> None:
        super().__init__()
        self.omega0 = omega0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x)


class SinMLP(nn.Module):
    r"""
    Multi-layer perceptron with sinusoidal activation functions.

    If :math:`z` is the input, then the network computes

    .. math::
        f(z) = W_L\,\sin\Bigl(\omega_0\,W_{L-1}\,\sin\bigl(\cdots\,\sin(\omega_0\,W_1\,z+b_1)\bigr)+b_{L-1}\Bigr)+b_L,

    where :math:`W_i` and :math:`b_i` are the weight matrices and bias vectors of each layer, respectively.

    :param int input_dim: Dimensionality of the input.
    :param int output_dim: Dimensionality of the output.
    :param List[int] hidden_dims: The widths of hidden layers.
    :param bool bias: If True, each linear layer includes a bias term. Default is True.
    :param float omega0: Frequency scaling factor :math:`\omega_0` for the sinusoidal activation. Default is 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        bias: bool = False,
        omega0: float = 1.0,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        dims = [input_dim] + hidden_dims + [output_dim]

        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=bias) for i in range(len(dims) - 1)
        )
        self.activation = Sin(omega0=omega0)

    def init_weights(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                nn.init.uniform_(
                    layer.weight,
                    -math.sqrt(6 / layer.in_features) / self.layer.omega_0,
                    math.sqrt(6 / layer.in_features) / self.layer.omega_0,
                )
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class SIREN(nn.Module):
    r"""
    Sinusoidal Implicit Representation Network (SIREN) proposed by :footcite:t:`sitzmann2020implicit`.

    The network is composed of a Fourier positional encoding followed by a SinMLP.

    .. note::

        The frequency factors :math:`\omega_0` for the encoding and the SIREN are set to the default values 
        in the original paper. In practice, we recommend experimenting with different values.

    :param int input_dim: Input dimension for the positional encoding. E.g., 2 for a 2D image.
    :param int encoding_dim: Output dimension of the positional encoding.
    :param int out_channels: Number of channels for the output image. 1 for grayscale, 3 for RGB.
    :param List[int] siren_dims: Hidden‐layer sizes for the SIREN.
    :param dict omega0: Frequency factors for the positional encoding and SIREN, respectively. If None, defaults to {"encoding": 30.0, "siren": 1.0}.
    :param dict bias: Whether the positional encoding and SIREN include a bias term, respectively. If None, defaults to {"encoding": False, "siren": False}.
    :param str device: Device to run the model on. Default is "cpu".
    """

    def __init__(
        self,
        input_dim: int,
        encoding_dim: int,
        out_channels: int,
        siren_dims: list[int],
        omega0: Optional[dict] = None,
        bias: Optional[dict] = None,
        device: str = "cpu",
    ) -> None:

        super().__init__()

        if omega0 is None:
            omega0 = {"encoding": 30.0, "siren": 1.0}
        if bias is None:
            bias = {"encoding": True, "siren": False}

        self.pe = FourierPE(
            input_dim=input_dim,
            output_dim=encoding_dim,
            omega0=omega0["encoding"],
            bias=bias["encoding"],
        ).to(device)
        self.siren = SinMLP(
            input_dim=self.pe.output_dim,
            hidden_dims=siren_dims,
            output_dim=out_channels,
            omega0=omega0["siren"],
            bias=bias["siren"],
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        x = self.siren(x)
        return x


class ImplicitNeuralRepresentation(Reconstructor):
    r"""

    Implicit Neural Representation reconstruction.

    This method reconstructs an image by minimizing the loss function

    .. math::

        \min_{\theta}  \|y-Af_{\theta}(z)\|_2^2 + \lambda \|\nabla f_\theta(z)\|_1

    where :math:`z` is an input grid of pixels, and :math:`f_{\theta}` is a SIREN model with parameters
    :math:`\theta`. The minimization should be stopped early to avoid overfitting. The method uses the Adam
    optimizer.

    .. note::

        To use this method, you need to instanciate the SIREN model :math:`f_\theta(z)` independently.


    .. note::

        The learning rate provided by default is a typical value when training the model on a large image but 
        it needs to be tuned as it may be not optimal.  

    :param torch.nn.Module siren_net: SIREN network.
    :param list, tuple img_size: Size `(C,H,W)` of the input grid of pixels :math:`z`.
    :param int iterations: Number of optimization iterations.
    :param float learning_rate: Learning rate of the Adam optimizer.
    :param float regul_param: Regularization parameter :math:`\lambda`for the TV prior.
    :param bool verbose: If ``True``, print progress.
    :param bool re_init: If ``True``, re-initialize the network parameters before each reconstruction.
    """

    @_deprecated_alias(input_size="img_size")
    def __init__(
        self,
        siren_net,
        img_size,
        iterations=2500,
        learning_rate=1e-4,
        verbose=False,
        re_init=False,
        regul_param=None,
    ):
        super().__init__()
        self.siren_net = siren_net
        self.max_iter = int(iterations)
        self.lr = learning_rate
        self.verbose = verbose
        self.re_init = re_init
        self.img_size = img_size

        self.loss = MCLoss()
        self.prior = TVPrior()
        self.regul_param = regul_param

    def forward(self, y, physics, z=None, shape=None, **kwargs):
        r"""
        Reconstruct an image from the measurement :math:`y`. The reconstruction is performed by solving a minimization
        problem.

        :param torch.Tensor y: Measurement.
        :param torch.Tensor physics: Physics model.
        :param torch.Tensor z: Input grid of pixels on which the SIREN network is trained.
        :param tuple shape: If provided, the output is reshaped to this shape.
        """
        if self.re_init:
            for layer in self.siren_net.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

        self.siren_net.requires_grad_(True)
        z.requires_grad_(True)
        optimizer = torch.optim.Adam(self.siren_net.parameters(), lr=self.lr)
        for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
            x = self.siren_net(z)
            if shape is not None:
                x = x.view(shape)
            error = self.loss(y=y, x_net=x, physics=physics)
            if self.regul_param is not None:
                error += self.regul_param * self.prior(x, z)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()

        out = self.siren_net(z)
        if shape is not None:
            return out.view(shape)
        return out
