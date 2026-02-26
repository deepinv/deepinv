import torch


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
