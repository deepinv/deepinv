"""Hypergradient oracle interface and the strong-convexity distance fact.

MAID (outer loop) is solver-agnostic. Everything that depends on how the
lower level is solved, and on which a posteriori bound certifies the
hypergradient error, belongs behind :class:`HypergradientOracle`.

Unifying fact
-------------
Lemma 1 of Bogensperger, Ehrhardt, Pock, Salehi and Wong (arXiv 2412.06436),
and the same elementary fact behind every bound in Salehi et al.
(SIAM J. Math. Data Sci. 2025): if ``Phi`` is ``mu``-strongly convex with
minimiser ``x_star``, then for every ``x``

    ||x_star - x|| <= (1 / mu) * ||grad Phi(x)||.

Every a posteriori certificate used by an oracle is a variant of this.
It is what lets a user reason about a lower-level solver that has not been
wired in yet: produce a residual that controls ``||grad Phi||`` (or an
equivalent dual residual), divide by a known ``mu``, and the distance to
the minimiser is controlled.

Certification rule
------------------
``certified`` is ``True`` only for a bound proven in a citable paper, with
every constant (in particular every strong-convexity modulus) known rather
than estimated. An estimated ``mu`` that is too large makes every bound too
small, which is the dangerous direction: MAID then accepts a direction that
is not a descent direction while the line search still appears to succeed.

Non-certified bounds are opt-in only (``allow_uncertified=True`` at
construction). They must be measured by under-estimation rate against a
certified bound on problems where both are computable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


def strong_convexity_distance_bound(grad_norm: float, mu: float) -> float:
    r"""Lemma 1: :math:`\|x_\star - x\| \le \|\nabla\Phi(x)\| / \mu`.

    :param float grad_norm: :math:`\|\nabla\Phi(x)\|`.
    :param float mu: strong-convexity modulus of :math:`\Phi`, must be known
        and positive. An estimated ``mu`` must not be passed here if the
        caller intends a certified bound.
    :return: upper bound on :math:`\|x_\star - x\|`.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    return float(grad_norm) / float(mu)


@dataclass
class LowerLevelState:
    """Inexact lower-level solution with a certified (or claimed) distance.

    :param x: primal reconstruction used by the upper-level loss.
    :param eps: claimed bound on ``||x - xhat||``.
    :param extras: solver-specific payload (dual variable, residuals, ...).
    """

    x: torch.Tensor
    eps: float
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class HypergradientState:
    """Inexact hypergradient and the data needed to evaluate its error bound.

    :param z: inexact hypergradient.
    :param delta: adjoint / linear-solve accuracy parameter(s).
    :param extras: quantities consumed by :meth:`HypergradientOracle.error_bound`.
    """

    z: torch.Tensor
    delta: float
    extras: dict[str, Any] = field(default_factory=dict)


class HypergradientOracle(ABC):
    """Solver-specific lower level, hypergradient and error certificate.

    MAID only sees this interface. It never inspects which DeepInverse
    optimiser produced ``z``.
    """

    @property
    @abstractmethod
    def certified(self) -> bool:
        """True only for a bound proven in a citable paper with known constants."""

    @property
    @abstractmethod
    def citation(self) -> str:
        """Short bibliographic key for the bound (empty if not certified)."""

    @property
    @abstractmethod
    def L_g(self) -> float:
        """Lipschitz constant of the upper-level loss gradient in x."""

    def require_certified_or_opt_in(self, allow_uncertified: bool) -> None:
        """Enforce the certification rule at construction time.

        Raises ``ValueError`` when the oracle is not certified and the
        caller has not explicitly opted in.
        """
        if not self.certified and not allow_uncertified:
            raise ValueError(
                f"{type(self).__name__} is not certified "
                f"(citation={self.citation!r}). Non-certified a posteriori "
                "bounds can under-estimate the hypergradient error and break "
                "MAID's descent guarantee without any runtime failure. Pass "
                "allow_uncertified=True only when you accept that convergence "
                "is proven only for certified bounds, and measure the "
                "under-estimation rate against a certified oracle."
            )

    @abstractmethod
    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        """Solve the lower level so that ``||x - xhat|| <= eps``."""

    @abstractmethod
    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        """Form an inexact hypergradient at the given lower-level state."""

    @abstractmethod
    def error_bound(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        hyper: HypergradientState,
        eps: float,
        delta: float,
    ) -> float:
        r"""Return ``omega`` such that ``||z - grad f(theta)|| <= omega``.

        For a certified oracle this is a theorem. For a non-certified oracle
        this is a claim that must be measured empirically.
        """

    @abstractmethod
    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Upper-level loss ``g(x)`` (scalar tensor)."""

    @abstractmethod
    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient of the upper-level loss in ``x``."""

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        """Exact upper-level value when a closed form is available.

        Default: not implemented. Used only for diagnostics and tests.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a closed-form upper level."
        )

    def update_lipschitz_estimates(
        self, lower: LowerLevelState, theta: torch.Tensor
    ) -> None:
        """Optional running-max update of Lipschitz constants. Default no-op."""
        return None
