"""Backward-compatible re-exports for the smooth hypergradient path.

Prefer :mod:`deepinv.optim.bilevel.smooth` and
:class:`~deepinv.optim.bilevel.oracle.HypergradientOracle` for new code.
"""

from .smooth import (
    hypergradient_error_bound,
    inexact_gradient,
    smooth_hypergradient_error_bound,
)

__all__ = [
    "hypergradient_error_bound",
    "inexact_gradient",
    "smooth_hypergradient_error_bound",
]
