"""Regularization losses for jointly learning physics and reconstruction models."""

from __future__ import annotations

from torch import Tensor

from deepinv.loss.loss import Loss


class CodesignRegularization(Loss):
    r"""
    Base class for regularizing trainable parameters of a physics operator.

    Codesign regularizations differ from image or network regularizations in
    that they act on a parameter of ``physics`` rather than on the reconstructed
    image or the reconstruction network. They can therefore be included
    directly in the ``losses`` list passed to :class:`deepinv.Trainer`.

    A subclass should implement :meth:`regularization`. The default
    :meth:`forward` method adapts that function to the interface expected by
    :class:`deepinv.loss.Loss`.
    """

    def regularization(self, physics, **kwargs) -> Tensor:
        """Compute the regularization term for ``physics``."""
        raise NotImplementedError

    def forward(
        self,
        x_net: Tensor | None = None,
        x: Tensor | None = None,
        y: Tensor | None = None,
        physics=None,
        model=None,
        **kwargs,
    ) -> Tensor:
        """
        Evaluate the codesign regularization.

        The arguments ``x_net``, ``x``, ``y`` and ``model`` are accepted for
        compatibility with the standard loss interface. Codesign
        regularizations use the physics operator instead.

        :param deepinv.physics.Physics physics: Physics operator containing the
            trainable parameter to regularize.
        :return: Scalar regularization value or a tensor accepted by
            :class:`deepinv.Trainer`.
        """
        if physics is None:
            raise ValueError("CodesignRegularization requires a physics operator.")

        return self.regularization(physics=physics, **kwargs)


class BinaryRegularization(CodesignRegularization):
    r"""
    Binary regularizer to encourage sensing matrix coefficients to converge to
    :math:`\pm 1 / \sqrt{m}`.

    The loss was first introduced by Higham et al. :footcite:t:`higham2018deep`.
    The current implementation with the learning schedule was obtained from
    Bacca et al. :footcite:t:`bacca2021deep`.

    For a compressed-sensing matrix :math:`A` with :math:`m` measurements, the
    penalty is

    .. math::

        \left(A_{i,j} - \frac{1}{\sqrt{m}}\right)^2
        \left(A_{i,j} + \frac{1}{\sqrt{m}}\right)^2,

    which is minimized when :math:`A_{i,j}=\pm 1/\sqrt{m}`. The penalty is
    averaged over the columns of each sensing row, and the resulting tensor is
    compatible with :class:`deepinv.Trainer`.

    The parameter itself must be included in the optimizer by the caller.

    :param int m: Number of measurements used to define the binary target
        :math:`1 / \sqrt{m}`.
    :param str parameter_name: Name of the physics parameter to regularize.
        Defaults to ``"_A"`` for :class:`deepinv.physics.CompressedSensing`.
    :param float weight: Initial regularization weight. Defaults to ``1.0``.
    :param float weight_increase_factor: Factor to increase the weight after
        each forward pass. Defaults to ``1.0``.
    :param float max_weight: Maximum weight threshold to clip. Use ``None`` for
        no upper bound. Defaults to ``1e3``.
    """

    def __init__(
        self,
        m: int,
        parameter_name: str = "_A",
        weight: float = 1.0,
        weight_increase_factor: float = 1.0,
        max_weight: float | None = 1e3,
    ):
        super().__init__()

        if m <= 0:
            raise ValueError("m must be a positive integer.")
        if not isinstance(parameter_name, str) or not parameter_name:
            raise ValueError("parameter_name must be a non-empty string.")
        if weight < 0:
            raise ValueError("weight must be non-negative.")
        if weight_increase_factor < 0:
            raise ValueError("weight_increase_factor must be non-negative.")
        if max_weight is not None and max_weight < 0:
            raise ValueError("max_weight must be non-negative or None.")

        self.m = m
        self.parameter_name = parameter_name
        self.weight = weight
        self.weight_increase_factor = weight_increase_factor
        self.max_weight = max_weight
        self.target_value = 1.0 / (m**0.5)

    def regularization(self, physics, **kwargs) -> Tensor:
        """Compute the binary regularization term for ``physics``."""
        if not hasattr(physics, self.parameter_name):
            raise AttributeError(
                f"Physics operator {physics.__class__.__name__!r} has no "
                f"parameter {self.parameter_name!r}."
            )

        matrix = getattr(physics, self.parameter_name)
        if not isinstance(matrix, Tensor):
            raise TypeError(
                f"Physics parameter {self.parameter_name!r} must be a torch.Tensor."
            )

        # Penalty: (x - 1/√m)^2*(x + 1/√m)^2 is minimized when x = ±1/√m.
        # This encourages coefficients to converge to either +1/√m or -1/√m.
        # Formula: sum of penalties toward both targets (original paper formulation).
        target = matrix.new_tensor(self.target_value)
        penalty = ((matrix - target).square()) * ((matrix + target).square())

        # Increase weight if factor > 1.0 (dynamic regularization schedule).
        self.weight = self.weight * self.weight_increase_factor

        # Clip to max_weight if specified (prevent runaway regularization).
        if self.max_weight is not None:
            self.weight = min(self.weight, self.max_weight)

        return self.weight * penalty.mean(dim=1)
