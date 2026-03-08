from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch import Tensor, nn
from abc import ABC

from deepinv.physics import LinearPhysics
from .base import Reconstructor

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
        self.model = deal_lib.DEAL(color=color).to(self.device).eval()

        
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
