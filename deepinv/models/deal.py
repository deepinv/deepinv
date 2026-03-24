from __future__ import annotations

from abc import ABC
from typing import Any, Callable

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import Tensor, nn

from deepinv.physics import LinearPhysics
from deepinv.optim.linear import conjugate_gradient
from deepinv.optim import Prior
from .base import Reconstructor

ms = torch

class DEAL(Reconstructor):
    """
    Deep Equilibrium Attention Least Squares (DEAL) reconstruction model.

    This model solves linear inverse problems using a learned equilibrium-based
    regularizer combined with iterative conjugate gradient least-squares updates. It can be used for
    image restoration and reconstruction tasks such as denoising, deblurring,
    and computed tomography reconstruction.

    This implementation is adapted from the official DEAL repository:
    https://github.com/mehrsapo/DEAL

    For the original method, see :footcite:t:`pourya2025dealing`.

    A pretrained network can be loaded by setting ``pretrained='download'``.

    The reconstruction is obtained by solving a regularized least-squares problem

    .. math::

        \hat{x} = \arg\min_x \frac{1}{2}\|Ax - y\|^2 + \lambda R_\theta(x),

    where :math:`A` is the forward operator, :math:`y` the measurements,
    and :math:`R_\theta(x)` a learned, spatially adaptive regularizer.

    The optimization is performed iteratively using a fixed-point scheme.
    At each outer iteration, the algorithm updates the reconstruction by solving
    a linearized least-squares subproblem using conjugate gradient:

    .. math::

        x^{(k+1)} \approx \arg\min_x \frac{1}{2}\|Ax - y\|^2 + \lambda \nabla R_\theta(x^{(k)})^\top x.

    The regularizer is parameterized by a neural network which produces
    spatially varying weights, allowing the model to adapt to local image structure.

    :param pretrained: checkpoint path or ``'download'``.
    :type pretrained: str

    :param sigma: noise-level parameter used by DEAL.
    :type sigma: float

    :param lam: regularization strength used by the DEAL solver.
    :type lam: float

    :param max_iter: maximum number of outer fixed-point iterations.
    :type max_iter: int

    :param auto_scale: if ``True``, rescales measurements based on their std.
    :type auto_scale: bool

    :param target_y_std: target std for auto-scaling when enabled.
    :type target_y_std: float

    :param color: if ``True``, use the color DEAL variant; otherwise grayscale.
    :type color: bool

    :param device: compute device. If ``None``, use CUDA if available.
    :type device: str or None

    :param clamp_output: if ``True``, clamp output to ``[0, 1]``.
    :type clamp_output: bool
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

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.sigma = float(sigma)
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.auto_scale = bool(auto_scale)
        self.target_y_std = float(target_y_std)
        self.clamp_output = bool(clamp_output)

        self.model = _DEALImpl(color=color).to(self.device).eval()

        if pretrained == "download":
            if color:
                url = (
                    "https://raw.githubusercontent.com/mehrsapo/DEAL/main/"
                    "trained_models/deal_color.pth"
                )

            else:
                url = (
                    "https://raw.githubusercontent.com/mehrsapo/DEAL/main/"
                    "trained_models/deal_gray.pth"
                )

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

        raw_state_dict = state.get("state_dict", state)

        model_state_dict = self.model.state_dict()

        # Older DEAL checkpoints do not contain newly introduced buffers.
        # Merge them with the current defaults while keeping pretrained weights.
        merged_state_dict = model_state_dict.copy()
        merged_state_dict.update(
            {
                key: value
                for key, value in raw_state_dict.items()
                if key in model_state_dict
            }
        )

        self.model.load_state_dict(merged_state_dict, strict=True)

    @torch.no_grad()
    def forward(self, y: torch.Tensor, physics: LinearPhysics) -> torch.Tensor:
        """
        Run the DEAL reconstruction.

        Parameters
        ----------
        y : torch.Tensor
            Measurements (e.g. sinogram).
        physics : deepinv.physics.LinearPhysics
            DeepInverse linear physics operator with ``__call__`` and ``A_adjoint``.

        Returns
        -------
        torch.Tensor
            Reconstructed image with the same spatial shape as ``H^T y``.
        """
        y = y.to(self.device)

        if physics.__class__.__name__ == "Denoising":
            sigma = torch.tensor([[self.sigma]], device=self.device)
            x_hat = self.model.denoise(y, sigma)
            return x_hat.clamp(0.0, 1.0) if self.clamp_output else x_hat

        def H(z: torch.Tensor) -> torch.Tensor:
            return physics.A(z)

        Ht = physics.A_adjoint

        if self.auto_scale:
            y_std = float(y.std().detach().cpu())
            if 0.0 < y_std < 5.0:
                scale = self.target_y_std / (y_std + 1e-12)
                y = y * scale

        x_init = torch.zeros_like(Ht(y))

        if hasattr(self.model, "max_iter"):
            self.model.max_iter = max(int(self.max_iter), 1)

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



class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the input.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        coefficients: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        num_knots: int,
        zero_knot_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the linear spline activation."""
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - step_size.item())
        floored_x = torch.floor((x_clamped - x_min) / step_size)
        fracs = (x - x_min) / step_size - floored_x
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)
        activation_output = coefficients_vect[indexes + 1] * fracs + coefficients_vect[
            indexes
        ] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        return activation_output

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        """Backpropagate through the active linear spline coefficients."""
        fracs, coefficients, indexes, step_size = ctx.saved_tensors
        coefficients_vect = coefficients.view(-1)

        grad_x = (
            (coefficients_vect[indexes + 1] - coefficients_vect[indexes])
            / step_size
            * grad_out
        )

        grad_coefficients_vect = torch.zeros_like(
            coefficients_vect, dtype=coefficients_vect.dtype
        )
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, (fracs * grad_out).view(-1)
        )
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), ((1 - fracs) * grad_out).view(-1)
        )

        grad_coefficients = grad_coefficients_vect.view(coefficients.shape)
        return grad_x, grad_coefficients, None, None, None, None


class LinearSplineDerivative_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the input.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        coefficients: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        num_knots: int,
        zero_knot_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the derivative of the linear spline activation."""
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - step_size.item())
        floored_x = torch.floor((x_clamped - x_min) / step_size)
        fracs = (x - x_min) / step_size - floored_x
        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

        coefficients_vect = coefficients.view(-1)
        activation_output = (
            coefficients_vect[indexes + 1] - coefficients_vect[indexes]
        ) / step_size

        ctx.save_for_backward(fracs, coefficients, indexes, step_size)
        return activation_output

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        """Backpropagate through the active spline-derivative coefficients."""
        fracs, coefficients, indexes, step_size = ctx.saved_tensors
        grad_x = 0 * grad_out

        grad_coefficients_vect = torch.zeros_like(coefficients.view(-1))
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1) + 1, torch.ones_like(fracs).view(-1) / step_size
        )
        grad_coefficients_vect.scatter_add_(
            0, indexes.view(-1), -torch.ones_like(fracs).view(-1) / step_size
        )

        return grad_x, grad_coefficients_vect, None, None, None, None


class Quadratic_Spline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the input.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        coefficients: torch.Tensor,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
        num_knots: int,
        zero_knot_indexes: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the quadratic spline activation."""
        step_size = (x_max - x_min) / (num_knots - 1)
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - 2 * step_size.item())
        floored_x = torch.floor((x_clamped - x_min) / step_size)

        indexes = (zero_knot_indexes.view(1, -1, 1, 1, 1) + floored_x).long()

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
            coefficients_vect[indexes + 2] * shift1
            + coefficients_vect[indexes + 1] * (1 - 2 * shift1)
            + coefficients_vect[indexes] * (shift1 - 1)
        )
        grad_x = grad_x / step_size

        ctx.save_for_backward(
            grad_x, frac1, frac2, frac3, coefficients, indexes, step_size
        )
        return activation_output

    @staticmethod
    def backward(
        ctx: Any, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
        """Backpropagate through the active quadratic spline coefficients."""
        grad_x, frac1, frac2, frac3, coefficients, indexes, grid = ctx.saved_tensors

        coefficients_vect = coefficients.view(-1)
        grad_x = grad_x * grad_out

        grad_coefficients_vect = torch.zeros_like(coefficients_vect)
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
    Class for LinearSpline activation functions.

    Args:
        num_knots: Number of knots of the spline.
        num_activations: Number of activation functions.
        x_min: Position of left-most knot.
        x_max: Position of right-most knot.
        slope_min: Minimum slope of the activation.
        slope_max: Maximum slope of the activation.
    """

    def __init__(
        self,
        num_activations: int,
        num_knots: int,
        x_min: float,
        x_max: float,
        init: str | float,
        slope_max: float | None = None,
        slope_min: float | None = None,
        clamp: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.num_knots = int(num_knots)
        self.num_activations = int(num_activations)
        self.init = init
        self.slope_min = slope_min
        self.slope_max = slope_max

        self.register_buffer("x_min", torch.tensor([x_min]))
        self.register_buffer("x_max", torch.tensor([x_max]))
        self.register_buffer(
            "step_size", (self.x_max - self.x_min) / (self.num_knots - 1)
        )
        self.clamp = clamp
        self.no_constraints = slope_max is None and slope_min is None and not clamp

        coefficients = self.initialize_coeffs()
        self.coefficients = nn.Parameter(coefficients)

        self.projected_coefficients_cached = None
        self.register_buffer(
            "D2_filter", Tensor([1, -2, 1]).view(1, 1, 3).div(self.step_size)
        )

        self.init_zero_knot_indexes()

    def init_zero_knot_indexes(self) -> None:
        """Initialize indexes of zero knots of each activation."""
        activation_arange = torch.arange(0, self.num_activations)
        self.register_buffer("zero_knot_indexes", activation_arange * self.num_knots)

    def initialize_coeffs(self) -> torch.Tensor:
        """Initialize spline coefficients."""
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
    def projected_coefficients(self) -> torch.Tensor:
        """B-spline coefficients projected to meet the constraint."""
        if self.projected_coefficients_cached is not None:
            return self.projected_coefficients_cached
        return self.clipped_coefficients()

    def cached_projected_coefficients(self) -> None:
        """Cache projected coefficients."""
        if self.projected_coefficients_cached is None:
            self.projected_coefficients_cached = self.clipped_coefficients()

    @property
    def slopes(self) -> torch.Tensor:
        """Get the slopes of the activations."""
        coeff = self.projected_coefficients
        slopes = (coeff[:, 1:] - coeff[:, :-1]) / self.step_size
        return slopes

    @property
    def device(self) -> torch.device:
        """Return parameter device."""
        return self.coefficients.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear spline activation.
        """
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

    def extra_repr(self) -> str:
        """Representation for print(model)."""
        s = (
            "num_activations={num_activations}, "
            "init={init}, num_knots={num_knots}, "
            "range=[{x_min[0]:.3f}, {x_max[0]:.3f}], "
            "slope_max={slope_max}, "
            "slope_min={slope_min}."
        )
        return s.format(**self.__dict__)

    def clipped_coefficients(self) -> torch.Tensor:
        """Project spline coefficients to satisfy constraints."""
        device = self.device

        if self.no_constraints:
            return self.coefficients

        cs = self.coefficients
        new_slopes = (cs[:, 1:] - cs[:, :-1]) / self.step_size

        if self.slope_min is not None or self.slope_max is not None:
            new_slopes = torch.clamp(new_slopes, self.slope_min, self.slope_max)

        if self.clamp:
            new_slopes[:, 0] = 0
            new_slopes[:, -1] = 0

        new_cs = torch.zeros(self.coefficients.shape, device=device, dtype=cs.dtype)
        new_cs[:, 1:] = torch.cumsum(new_slopes, dim=1) * self.step_size
        new_cs = new_cs + (cs - new_cs).mean(dim=1).unsqueeze(1)
        return new_cs

    @property
    def relu_slopes(self) -> torch.Tensor:
        """
        Get the activation ReLU slopes by convolving coefficients with [1, -2, 1].
        """
        return F.conv1d(
            self.projected_coefficients.unsqueeze(1), self.D2_filter
        ).squeeze(1)

    def tv2(self) -> torch.Tensor:
        """
        Compute the second-order total-variation regularization.
        """
        return self.relu_slopes.norm(1, dim=1).sum()


def symmetric_pad(tensor: ms.Tensor, pad_size: tuple[int, int, int, int]) -> ms.Tensor:
    """
    Pad symmetrically the spatial dimensions of the input tensor.
    """
    sz = tensor.shape
    sz = [sz[0], sz[1], sz[2], sz[3]]
    sz[-1] = sz[-1] + sum(pad_size[2::])
    sz[-2] = sz[-2] + sum(pad_size[0:2])
    out = ms.empty(sz, dtype=tensor.dtype).to(tensor.device)

    out[
        ...,
        pad_size[0] : out.size(-2) - pad_size[1],
        pad_size[2] : out.size(-1) - pad_size[3],
    ] = tensor

    if pad_size[0] != 0:
        out[..., 0 : pad_size[0], :] = ms.flip(
            out[..., pad_size[0] : 2 * pad_size[0], :], (-2,)
        )
    if pad_size[1] != 0:
        out[..., out.size(-2) - pad_size[1] :, :] = ms.flip(
            out[..., out.size(-2) - 2 * pad_size[1] : out.size(-2) - pad_size[1], :],
            (-2,),
        )
    if pad_size[2] != 0:
        out[..., :, 0 : pad_size[2]] = ms.flip(
            out[..., :, pad_size[2] : 2 * pad_size[2]], (-1,)
        )
    if pad_size[3] != 0:
        out[..., :, out.size(-1) - pad_size[3] :] = ms.flip(
            out[..., :, out.size(-1) - 2 * pad_size[3] : out.size(-1) - pad_size[3]],
            (-1,),
        )
    return out


def symmetric_pad_transpose(
    tensor: ms.Tensor, pad_size: tuple[int, int, int, int]
) -> ms.Tensor:
    """
    Adjoint of the symmetric padding operation.
    """
    sz = list(tensor.size())
    out = tensor.clone()

    if pad_size[0] != 0:
        out[..., pad_size[0] : 2 * pad_size[0], :] += ms.flip(
            out[..., 0 : pad_size[0], :], (-2,)
        )
    if pad_size[1] != 0:
        out[..., -2 * pad_size[1] : -pad_size[1], :] += ms.flip(
            out[..., -pad_size[1] :, :], (-2,)
        )
    if pad_size[2] != 0:
        out[..., pad_size[2] : 2 * pad_size[2]] += ms.flip(
            out[..., 0 : pad_size[2]], (-1,)
        )
    if pad_size[3] != 0:
        out[..., -2 * pad_size[3] : -pad_size[3]] += ms.flip(
            out[..., -pad_size[3] :], (-1,)
        )

    end_h = sz[-2] + 1 if pad_size[1] == 0 else sz[-2] - pad_size[1]
    end_w = sz[-1] + 1 if pad_size[3] == 0 else sz[-1] - pad_size[3]
    out = out[..., pad_size[0] : end_h, pad_size[2] : end_w]
    return out


class MultiConv2d(nn.Module):
    """
    Multi-layer convolution with spectral norm control.
    """

    def __init__(
        self,
        num_channels: list[int] | None = None,
        size_kernels: list[int] | None = None,
        zero_mean: bool = True,
        sn_size: int = 256,
        color: bool = False,
    ) -> None:
        super().__init__()

        if num_channels is None:
            num_channels = [1, 64]
        if size_kernels is None:
            size_kernels = [3]

        self.size_kernels = size_kernels
        self.num_channels = num_channels
        self.sn_size = sn_size
        self.zero_mean = zero_mean
        self.padding = self.size_kernels[0] // 2
        self.color = color

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
            if zero_mean and j == 0:
                P.register_parametrization(self.conv_layers[-1], "weight", ZeroMean())

        self.L = torch.tensor(1.0, requires_grad=True)
        self.padding_total = sum(kernel_size // 2 for kernel_size in size_kernels)

        if color:
            dirac = torch.zeros(
                (1, 3, 4 * self.padding_total + 1, 4 * self.padding_total + 1)
            )
            dirac[0, 1, 2 * self.padding_total, 2 * self.padding_total] = 1
        else:
            dirac = torch.zeros(
                (1, 1, 4 * self.padding_total + 1, 4 * self.padding_total + 1)
            )
            dirac[0, 0, 2 * self.padding_total, 2 * self.padding_total] = 1

        self.register_buffer("dirac", dirac)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stacked convolution operator."""
        return self.convolution(x)

    def convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the normalized forward convolution."""
        x = x / torch.sqrt(self.L)

        for conv in self.conv_layers:
            weight = conv.weight
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

    def transpose(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the adjoint stacked convolution operator."""
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
        return x

    def spectral_norm(self, mode: str = "Fourier", n_steps: int = 50) -> torch.Tensor:
        """
        Compute the spectral norm of the convolutional layer.
        """
        if mode == "Fourier":
            self.L = torch.tensor([1.0], device=self.conv_layers[0].weight.device)
            kernel = self.get_kernel_WtW()
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

            sn = torch.linalg.norm(self.transpose(self.convolution(u)))
            self.L = sn
            return sn

    def check_tranpose(self) -> None:
        """
        Check that the convolutional layer is indeed the transpose.
        """
        device = self.conv_layers[0].weight.device

        for _ in range(1):
            x1 = torch.empty((1, 1, 40, 40), device=device).normal_()
            x2 = torch.empty(
                (1, self.num_channels[-1], 40, 40), device=device
            ).normal_()

            ps_1 = (self(x1) * x2).sum()
            ps_2 = (self.transpose(x2) * x1).sum()
            print(f"ps_1: {ps_1.item()}")
            print(f"ps_2: {ps_2.item()}")
            print(f"ratio: {ps_1.item() / ps_2.item()}")

    def spectrum(self) -> torch.Tensor:
        """Return the Fourier spectrum of the normal operator kernel."""
        kernel = self.get_kernel_WtW()
        padding = (self.sn_size - 1) // 2 - self.padding_total
        return torch.fft.fft2(
            torch.nn.functional.pad(kernel, (padding, padding, padding, padding))
        )

    def get_filters(self) -> torch.Tensor:
        """Return the effective filters of the stacked convolution."""
        kernel = self.convolution(self.dirac)[
            :,
            :,
            self.padding_total : 3 * self.padding_total + 1,
            self.padding_total : 3 * self.padding_total + 1,
        ]
        return kernel

    def get_kernel_WtW(self) -> torch.Tensor:
        """Return the kernel of the normal operator W^T W."""
        dirac = self.dirac.to(dtype=self.conv_layers[0].weight.dtype)
        return self.transpose(self.convolution(dirac))


class ZeroMean(nn.Module):
    """Enforce zero-mean kernels for each output channel."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project kernels to have zero mean per output channel."""
        return x - torch.mean(x, dim=(1, 2, 3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)

class DEALRegularizer(Prior):
    """
    DEAL adaptive regularizer as a standalone Prior.

    This class exposes the learned regularization term of DEAL,
    allowing it to be used independently in optimization algorithms.
    """

    def __init__(self, model: "_DEALImpl"):
        super().__init__()
        self.model = model

    def grad(self, x: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
        """
        Compute the gradient of the DEAL regularizer.

        :param x: input image
        :param sigma: noise level
        :return: gradient of the regularizer at x
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([[sigma]], device=x.device, dtype=x.dtype)

        # initialize internal parameters
        self.model.cal_lambda(sigma)
        self.model.cal_scaling(sigma)

        # compute mask based on current x
        self.model.cal_mask(x)

        idx = [i for i in range(x.size(0))]

        # gradient ≈ λ * L^T L x
        return self.model.lmbda * self.model.Lt(self.model.L(x, idx), idx)

class _DEALImpl(nn.Module):
    """
    Internal implementation of the original DEAL solver.

    This class contains the learned regularizer, mask computation, and
    conjugate-gradient-based inverse-problem solver used by the public
    :class:`DEAL` wrapper.
    """

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

    def cal_lambda(self, sigma: torch.Tensor) -> None:
        """Compute the regularization parameter from the noise level."""
        self.lmbda = self.spline_lambda(sigma)

    def cal_scaling(self, sigma: torch.Tensor) -> None:
        """Compute channel-wise scaling factors from the noise level."""
        sigma = torch.ones((sigma.size(0), self.last_c, 1, 1)).to(sigma.device) * sigma
        self.scaling = torch.exp(self.spline_scaling(sigma)) / (sigma + 1e-5)

    def last_act(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the final mask activation."""
        x = torch.abs(x)
        x = self.spline3(self.scaling * x)
        return torch.clip(x, 1e-2, 1)

    def K(self, x: torch.Tensor, idx: list[int] | None = None) -> torch.Tensor:
        """Apply the weighted regularization operator."""
        return torch.sqrt(self.lmbda) * self.W1(x) * self.mask[idx]

    def Kt(self, y: torch.Tensor, idx: list[int] | None = None) -> torch.Tensor:
        """Apply the adjoint of the weighted regularization operator."""
        return torch.sqrt(self.lmbda) * self.W1.transpose(y * self.mask[idx])

    def KtK(self, x: torch.Tensor, idx: list[int] | None = None) -> torch.Tensor:
        """Apply the normal operator associated with the regularizer."""
        return self.Kt(self.K(x, idx), idx)

    def cal_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the spatially varying DEAL mask."""
        self.mask = self.last_act(
            self.M3(
                self.spline2(torch.abs(self.M2(self.spline1(torch.abs(self.M1(x))))))
            )
        )
        return self.mask

    def L(self, x: torch.Tensor, idx: list[int] | None = None) -> torch.Tensor:
        """Apply the masked analysis operator."""
        return self.W1(x) * self.mask[idx]

    def Lt(self, y: torch.Tensor, idx: list[int] | None = None) -> torch.Tensor:
        """Apply the adjoint masked analysis operator."""
        return self.W1.transpose(y * self.mask[idx])

    def BtB(
        self,
        x: torch.Tensor,
        H: Callable[[torch.Tensor], torch.Tensor],
        Ht: Callable[[torch.Tensor], torch.Tensor],
        idx: list[int] | None = None,
    ) -> torch.Tensor:
        """Apply the linear system operator solved inside conjugate gradients."""
        BtBD = (Ht(H(x)) + self.lmbda[idx] * self.Lt(self.L(x, idx), idx)) / (
            1 + self.lmbda[idx]
        )
        return BtBD

    def cg_sample(
        self,
        b: torch.Tensor,
        x0: torch.Tensor,
        max_iter: int,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, int]:
        """Run conjugate gradients for the denoising subproblem."""
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        b = self.Kt(b, correct_idx_mask)
        r = b - self.KtK(x, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = []
        idx_uniques_cont = []
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
            beta = r_norm / r_norm_old
            p = r + beta * p

        return output, i

    def cg(
        self,
        b: torch.Tensor,
        x0: torch.Tensor,
        max_iter: int,
        eps: float = 1e-5,
        H: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        Ht: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        dims: tuple[int, int, int] = (1, 2, 3),
    ) -> tuple[torch.Tensor, int]:
        """Run conjugate gradients for the current linear inverse-problem iterate."""
        b = b / (1 + self.lmbda)
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        r = b - self.BtB(x, H, Ht, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = []
        idx_uniques_cont = []
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
            beta = r_norm / r_norm_old
            p = r + beta * p

        return output, i


    def denoise(self, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Run DEAL in denoising mode."""
        self.W1.spectral_norm()
        self.cal_lambda(sigma)
        self.cal_scaling(sigma)

        if self.training:
            self.c_k_list = []
            grad_steps = 1
            n_out = int(torch.randint(14, 59, (1, 1)).item())
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
            for i in range(n_out):
                self.cal_mask(c_k)
                c_k, self.last_cg_iter = self.cg(y, c_k_old, n_in, eps=eps_in)
                res = torch.linalg.norm(c_k - c_k_old) / torch.linalg.norm(c_k_old)
                c_k_old = c_k.clone()
                if (res < eps_out).all():
                    break

        def backward_hook1(grad: torch.Tensor) -> torch.Tensor:
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

        self.number_of_cgs = i + grad_steps
        return c_k1

    def solve_inverse_problem(
        self,
        y: torch.Tensor,
        H: Callable[[torch.Tensor], torch.Tensor],
        Ht: Callable[[torch.Tensor], torch.Tensor],
        sigma: float,
        lmbda: float,
        eps_in: float = 1e-8,
        eps_out: float = 1e-5,
        path: bool = False,
        x_init: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Solve a linear inverse problem with the DEAL equilibrium solver."""
        self.W1.spectral_norm()
        self.cal_scaling(torch.tensor([[sigma]]).to(y.device))
        self.lmbda = torch.tensor([[lmbda]]).to(y.device)
        if path:
            c_ks: list[torch.Tensor] = []

        max_iters = getattr(self, "max_iter", 1000)
        max_cg_iters = getattr(self, "max_iter", 1000)

        with torch.no_grad():
            if x_init is not None:
                c_k = x_init
            else:
                c_k = Ht(y) * 0
            c_k_old = c_k.clone()

            for m in range(max_iters):
                if path:
                    c_ks.append(c_k)

                self.cal_mask(c_k)
                b = Ht(y)
                A_op = lambda x: self.BtB(x, H, Ht, [i for i in range(x.size(0))])
                c_k = conjugate_gradient(
                    A=A_op,
                    b=b,
                    init=c_k_old,
                    max_iter=max_cg_iters,
                    tol=eps_in,
                    eps=1e-8,
                )
                cg_iters = max_cg_iters

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
        return torch.clip(c_k, 0, 1)
