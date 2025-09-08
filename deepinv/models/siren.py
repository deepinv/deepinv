import math
import torch
import torch.nn as nn
from typing import Optional

def get_mgrid(shape) -> torch.Tensor:
    """
    Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.

    :param tuple shape: The shape of the grid to generate.

    :return: A tensor of shape `(2, shape.prod())` containing the grid coordinates
    """
    tensors = tuple([torch.linspace(-1, 1, steps=steps) for steps in shape])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(shape))
    return mgrid

class FourierPE(nn.Module):
    r"""
    Fourier Positional Encoding.

    Computes the positional encoding :math:`\psi(x)` by applying a learnable linear transformation followed by a sinusoidal activation.
    For an input :math:`x`, the encoding is given by

    .. math::
        z = \Omega(x),
        \quad
        \psi(x) = \sin\bigl(\omega_0\,z\bigr),

    where :math:`\Omega:\mathbb{R}^{\text{input_dim}} \rightarrow \mathbb{R}^{\text{output_dim}}` is a linear mapping and :math:`\omega_0` is a frequency factor.

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
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))
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
        \text{Sin}(x) = \sin(\omega_0 \, x)
    
    :param float omega0: Frequency scaling factor.
    """
    def __init__(self, omega0: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x)

class SinMLP(nn.Module):
    r"""
    Multi-layer perceptron with sinusoidal activations.

    Defines a feedforward neural network that computes a mapping
    :math:`f: \mathbb{R}^{\text{input\_dim}} \to \mathbb{R}^{\text{output\_dim}}` via a series of fully-connected layers interleaved with a sinusoidal activation function.

    If :math:`x` is the input, then the network computes

    .. math::
        f(x) = W_L\,\phi\Bigl(W_{L-1}\,\phi\bigl(\cdots\,\phi(W_1\,x+b_1)\bigr)+b_{L-1}\Bigr)+b_L,

    where :math:`\phi` denotes the sinusoidal activation function and :math:`W_i` and :math:`b_i` are the weight matrices
    and bias vectors of each layer, respectively.

    :param int input_dim: Dimensionality of the input.
    :param int output_dim: Dimensionality of the output.
    :param List[int] hidden_dims: The widths of hidden layers.
    :param bool bias: If True, each linear layer includes a bias term. Default is True.
    :param float omega0: Frequency scaling factor for the sinusoidal activation. Default is 1.0.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        bias: bool = True,
        omega0: float = 1.0,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim, bias=bias)
            for in_dim, out_dim in zip(
                [input_dim] + hidden_dims[:-1],
                hidden_dims[1:] + [output_dim],
                strict=True,
            )
        )
        self.activation = Sin(omega0=omega0)

    def init_weights(self) -> None:
        r"""Initialize the weights of the SIREN."""
        with torch.no_grad():
            for layer in self.layers:
                nn.init.uniform_(layer.weight, -math.sqrt(6 / layer.in_features)  / self.layer.omega_0, math.sqrt(6 / layer.in_features)  / self.layer.omega_0)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class SIREN(nn.Module):
    r"""
    Sinusoidal Implicit Representation Network (SIREN).

    Maps inputs through a Fourier positional encoding onto a SinMLP.

    :param int input_dim: Input dimension for the positional encoding. E.g., 2 for a 2D image.
    :param int encoding_dim: Output dimension of the positional encoding.
    :param int out_channels: Number of channels for the output image. 1 for grayscale, 3 for RGB.
    :param List[int] siren_dims: Hidden‐layer sizes for the SIREN.
    :param Optional[tuple[int, ...]] output_shape: If provided, the output is reshaped to this shape.
    :param dict omega0: Frequency factors for the positional encoding and SIREN, respectively. Default is {"pe": 20.0, "siren": 1.0}.
    :param bool bias: If True, the encoding and SinMLP include a bias term for each layer. Default is True.
    :param str device: Device to run the model on. Default is "cpu".
    """

    def __init__(
        self,
        input_dim: int, 
        encoding_dim: int,
        out_channels: int,
        siren_dims: list[int],
        output_shape: Optional[tuple[int, ...]] = None,
        omega0: dict = {"encoding": 20.0, "siren": 1.0},
        bias: bool = True,
        device: str = "cpu"
    ) -> None:

        super().__init__()
        
        self.module = nn.ModuleDict()   
        self.output_shape = output_shape

        self.module["PE"] = self.pe = FourierPE(
            input_dim=input_dim,
            output_dim=encoding_dim,
            omega0=omega0["encoding"],
            bias=bias,
        ).to(device)
        self.module["SIREN"] = self.siren = SinMLP(
            input_dim=self.pe.output_dim,
            hidden_dims=siren_dims,
            output_dim=out_channels,
            omega0=omega0['siren'],
            bias=bias,
        ).to(device)

    def forward(self, x: torch.Tensor, output_shape=None) -> torch.Tensor:
        if output_shape is None:
            output_shape = self.output_shape
        x = self.pe(x)
        x = self.siren(x)
        return x.view(output_shape)