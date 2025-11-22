import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from abc import ABC, abstractproperty, abstractmethod
from . import deal_spline_autograd_func as spline_autograd_func


import time


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

        x = spline_autograd_func.LinearSpline_Func.apply(
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
