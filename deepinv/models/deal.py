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
