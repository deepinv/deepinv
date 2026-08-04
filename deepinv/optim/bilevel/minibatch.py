"""Minibatch accumulation of inexact hypergradients for MAID.

The paper's upper level is already a mean over ``m`` samples

    f(theta) = (1/m) sum_{i=1}^m g_i(xhat_i(theta)) + r(theta).

This module evaluates the inexact hypergradient of that mean by walking the
dataset in fixed-order chunks sized to fit available memory, without
weakening the a posteriori error bound that Algorithm 3.2 consumes.

Reduction order (determinism contract)
--------------------------------------
Samples are always processed in index order ``0, 1, ..., m-1``. Accumulators
are updated by sequential floating-point addition of per-sample quantities,
then divided by ``m``:

    z_acc     <- z_acc + z_i
    omega_acc <- omega_acc + omega_i
    g_acc     <- g_acc + g_i
    ng_acc    <- ng_acc + ||grad g_i||

Chunk size controls only how many samples are held in working memory during
the solve phase. It does not change the reduction order. With a fixed seed
and a deterministic single-sample oracle, the accumulated tensors are
therefore bitwise identical across runs and bitwise invariant to chunk size.

Error bound for the mean (requirement 3)
----------------------------------------
Let ``z_i`` be an inexact hypergradient for sample ``i`` with

    ||z_i - grad f_i(theta)|| <= omega_i.

The mean hypergradient and its bound are

    z = (1/m) sum_i z_i,
    ||z - grad f(theta)||
        = ||(1/m) sum_i (z_i - grad f_i)||
        <= (1/m) sum_i ||z_i - grad f_i||
        <= (1/m) sum_i omega_i.

So ``omega_mean = (1/m) * sum_i omega_i`` is a valid certificate for the
mean. This is what the descent test in Algorithm 3.2 line 8 must use.
Using ``max_i omega_i`` or a single-sample ``omega`` would either be loose
or incorrect; using a mean of the ``z_i`` with an un-averaged ``omega``
would silently break the guarantee.

Line-search quantities for the mean
-----------------------------------
With each sample satisfying ``||x_i - xhat_i|| <= eps``,

    |g_i(x_i) - g_i(xhat_i)|
        <= ||grad g_i(x_i)|| * eps + (L_{g,i}/2) * eps^2,

so the mean satisfies the same form with

    g_bar = (1/m) sum g_i,
    ||grad g||_bar = (1/m) sum ||grad g_i||,
    L_g_bar = (1/m) sum L_{g,i}.

:meth:`MinibatchOracle.g` returns ``g_bar``. :meth:`MinibatchOracle.grad_g`
returns a one-element tensor whose absolute value is ``||grad g||_bar``, so
that ``grad_g(...).norm()`` is the quantity MAID's line search needs.
:attr:`L_g` is ``L_g_bar``.

Goal-oriented estimator and Krylov recycling
--------------------------------------------
Each sample has its own Hessian ``H_i``. The main adjoint CG and any
dual-weighted residual recycling live inside that sample: recycling cannot
span samples, and the three DWR adjoint solves reset per sample. Chunking
does not create a cross-sample Krylov space. Cost therefore scales as
``m`` times the per-sample DWR cost, independent of chunk size (chunk size
only bounds concurrent memory).

Memory
------
Working memory for intermediates (CG state, temporary residuals) is
proportional to ``chunk_size``, not ``m``. A list of length-``m``
reconstructions is kept for warm starts across outer iterations; that
storage is ``O(m * dim)`` by design of warm-starting and is reported
separately from peak working memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .oracle import HypergradientOracle, HypergradientState, LowerLevelState


def mean_error_bound(omegas: Sequence[float]) -> float:
    r"""Aggregate per-sample hypergradient error bounds for the mean.

    If ``||z_i - grad f_i|| <= omega_i`` for each sample, then

    .. math::

        \Bigl\|
            \tfrac1m\sum_i z_i - \tfrac1m\sum_i \nabla f_i
        \Bigr\|
        \le \tfrac1m\sum_i \omega_i.

    :param omegas: per-sample bounds ``omega_i``, length ``m >= 1``.
    :return: ``omega_mean = (1/m) * sum omega_i``.
    """
    if len(omegas) == 0:
        raise ValueError("omegas must be non-empty")
    total = 0.0
    for w in omegas:
        total = total + float(w)
    return total / float(len(omegas))


def mean_accumulate(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sequential sum-then-divide of tensors (fixed reduction order).

    The reduction is left-to-right floating-point addition, then a single
    division by ``len(values)``. This is the documented contract for
    bitwise reproducibility and chunk-size invariance.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    acc = values[0].detach().clone()
    for v in values[1:]:
        acc = acc + v
    return acc / float(len(values))


@dataclass
class MinibatchOracle(HypergradientOracle):
    """Hypergradient oracle for a mean over independent sample oracles.

    :param sample_oracles: one :class:`HypergradientOracle` per dataset
        sample, in the fixed index order used for reduction.
    :param chunk_size: number of samples held in working memory at once.
        Must be in ``1 .. m``. Does not change the mathematical trajectory
        under the fixed reduction order above.
    """

    sample_oracles: list[HypergradientOracle]
    chunk_size: int = 1
    # Instrumentation (tests / demos).
    n_lower_solves: int = 0
    n_hypergradients: int = 0
    n_sample_lower_solves: int = 0
    n_sample_hypergradients: int = 0
    peak_working_bytes: int = 0
    last_omega_parts: list[float] = field(default_factory=list)
    # Cached line-search quantities from the last full pass.
    _cached_g: float | None = field(default=None, repr=False)
    _cached_grad_g_norm: float | None = field(default=None, repr=False)
    _cached_x_list: list[torch.Tensor] | None = field(default=None, repr=False)
    _cached_omega_mean: float | None = field(default=None, repr=False)

    def __init__(
        self,
        sample_oracles: Sequence[HypergradientOracle],
        chunk_size: int = 1,
    ):
        if len(sample_oracles) == 0:
            raise ValueError("sample_oracles must be non-empty")
        self.sample_oracles = list(sample_oracles)
        self.m = len(self.sample_oracles)
        if not (1 <= int(chunk_size) <= self.m):
            raise ValueError(
                f"chunk_size must lie in 1..{self.m}, got {chunk_size}"
            )
        self.chunk_size = int(chunk_size)
        # Certification: the mean is certified only if every sample is.
        self._certified = all(o.certified for o in self.sample_oracles)
        citations = sorted({o.citation for o in self.sample_oracles if o.citation})
        self._citation = (
            "mean of: " + "; ".join(citations) if citations else ""
        )
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.n_sample_lower_solves = 0
        self.n_sample_hypergradients = 0
        self.peak_working_bytes = 0
        self.last_omega_parts: list[float] = []
        self._cached_g = None
        self._cached_grad_g_norm = None
        self._cached_x_list = None
        self._cached_omega_mean = None

    # ------------------------------------------------------------------
    # HypergradientOracle API
    # ------------------------------------------------------------------
    @property
    def certified(self) -> bool:
        return self._certified

    @property
    def citation(self) -> str:
        return self._citation

    @property
    def L_g(self) -> float:
        # Mean of per-sample Lipschitz constants (see module docstring).
        total = 0.0
        for o in self.sample_oracles:
            total = total + float(o.L_g)
        return total / float(self.m)

    def reset_counters(self) -> None:
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.n_sample_lower_solves = 0
        self.n_sample_hypergradients = 0
        self.peak_working_bytes = 0
        self.last_omega_parts = []
        for o in self.sample_oracles:
            if hasattr(o, "reset_counters"):
                o.reset_counters()
            # Reset per-sample GD counters when present.
            prob = getattr(o, "problem", None)
            if prob is not None and hasattr(prob, "n_gd_iters"):
                prob.n_gd_iters = 0

    @property
    def n_gd_iters(self) -> int:
        total = 0
        for o in self.sample_oracles:
            prob = getattr(o, "problem", None)
            if prob is not None and hasattr(prob, "n_gd_iters"):
                total += int(prob.n_gd_iters)
        return total

    def _chunk_ranges(self):
        start = 0
        while start < self.m:
            end = min(start + self.chunk_size, self.m)
            yield range(start, end)
            start = end

    def _note_working(self, tensors: Sequence[torch.Tensor]) -> None:
        nbytes = 0
        for t in tensors:
            if t is None:
                continue
            nbytes += int(t.numel()) * int(t.element_size())
        if nbytes > self.peak_working_bytes:
            self.peak_working_bytes = nbytes

    def solve_lower_level(
        self,
        theta: torch.Tensor,
        eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        """Solve every sample lower level; return stacked reconstructions.

        Warm start expects ``warm_start.extras['x_list']`` (preferred) or a
        stacked ``warm_start.x`` of shape ``(m, ...)``.
        """
        x_list: list[torch.Tensor] = []
        grad_norms: list[float] = []
        warm_list = None
        if warm_start is not None:
            warm_list = warm_start.extras.get("x_list")
            if warm_list is None and warm_start.x is not None:
                # Stacked (m, d) form.
                warm_list = [warm_start.x[i] for i in range(self.m)]

        g_acc = 0.0
        ng_acc = 0.0

        for idx_range in self._chunk_ranges():
            chunk_tensors: list[torch.Tensor] = []
            for i in idx_range:
                warm_i = None
                if warm_list is not None:
                    warm_i = LowerLevelState(x=warm_list[i], eps=eps)
                lower_i = self.sample_oracles[i].solve_lower_level(
                    theta, eps=eps, warm_start=warm_i
                )
                self.n_sample_lower_solves += 1
                x_list.append(lower_i.x.detach())
                chunk_tensors.append(lower_i.x)
                g_i = float(self.sample_oracles[i].g(lower_i.x).item())
                ng_i = float(self.sample_oracles[i].grad_g(lower_i.x).norm().item())
                g_acc = g_acc + g_i
                ng_acc = ng_acc + ng_i
                gn = float(lower_i.extras.get("grad_norm", float("nan")))
                grad_norms.append(gn)
            self._note_working(chunk_tensors)

        self.n_lower_solves += 1
        g_bar = g_acc / float(self.m)
        ng_bar = ng_acc / float(self.m)
        self._cached_g = g_bar
        self._cached_grad_g_norm = ng_bar
        self._cached_x_list = x_list

        # Stack for a single tensor view; keep the list for fixed-order access.
        x_stack = torch.stack(x_list, dim=0)
        return LowerLevelState(
            x=x_stack,
            eps=eps,
            extras={
                "x_list": x_list,
                "grad_norms": grad_norms,
                "g_bar": g_bar,
                "grad_g_norm_bar": ng_bar,
                "m": self.m,
            },
        )

    def hypergradient(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        delta: float,
    ) -> HypergradientState:
        """Accumulate per-sample hypergradients with fixed reduction order."""
        x_list = lower.extras.get("x_list")
        if x_list is None:
            x_list = [lower.x[i] for i in range(self.m)]

        z_acc: torch.Tensor | None = None
        omega_parts: list[float] = []
        # Recompute g / grad_g norms for the line search if needed.
        g_acc = 0.0
        ng_acc = 0.0

        for idx_range in self._chunk_ranges():
            chunk_tensors: list[torch.Tensor] = []
            for i in idx_range:
                lower_i = LowerLevelState(x=x_list[i], eps=lower.eps)
                hyper_i = self.sample_oracles[i].hypergradient(
                    theta, lower_i, delta=delta
                )
                self.n_sample_hypergradients += 1
                z_i = hyper_i.z
                chunk_tensors.append(z_i)
                if z_acc is None:
                    z_acc = z_i.detach().clone()
                else:
                    z_acc = z_acc + z_i

                # Per-sample omega for the mean bound. Always form it here so
                # error_bound does not need to re-solve.
                omega_i = self.sample_oracles[i].error_bound(
                    theta, lower_i, hyper_i, eps=lower.eps, delta=delta
                )
                omega_parts.append(float(omega_i))

                g_i = float(self.sample_oracles[i].g(x_list[i]).item())
                ng_i = float(
                    self.sample_oracles[i].grad_g(x_list[i]).norm().item()
                )
                g_acc = g_acc + g_i
                ng_acc = ng_acc + ng_i
            self._note_working(chunk_tensors)

        assert z_acc is not None
        z_mean = z_acc / float(self.m)
        omega_mean = mean_error_bound(omega_parts)
        self.last_omega_parts = list(omega_parts)
        self._cached_omega_mean = omega_mean
        self._cached_g = g_acc / float(self.m)
        self._cached_grad_g_norm = ng_acc / float(self.m)
        self.n_hypergradients += 1

        return HypergradientState(
            z=z_mean,
            delta=delta,
            extras={
                "omega_mean": omega_mean,
                "omega_parts": list(omega_parts),
                "m": self.m,
                "chunk_size": self.chunk_size,
                "g_bar": self._cached_g,
                "grad_g_norm_bar": self._cached_grad_g_norm,
                "x_list": x_list,
            },
        )

    def error_bound(
        self,
        theta: torch.Tensor,
        lower: LowerLevelState,
        hyper: HypergradientState,
        eps: float,
        delta: float,
    ) -> float:
        """Return ``(1/m) sum omega_i`` (see module docstring)."""
        if "omega_mean" in hyper.extras:
            return float(hyper.extras["omega_mean"])
        # Fallback: recompute from parts if present.
        parts = hyper.extras.get("omega_parts")
        if parts is not None:
            return mean_error_bound(parts)
        raise RuntimeError(
            "MinibatchOracle.error_bound requires omega_mean in "
            "hyper.extras; call hypergradient first."
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        """Mean upper-level loss over samples.

        ``x`` may be a stacked ``(m, ...)`` tensor or ignored when a cached
        pass has just run (MAID always passes the lower-level state from
        the preceding solve).
        """
        if self._cached_g is not None and (
            self._cached_x_list is not None
            and isinstance(x, torch.Tensor)
            and x.shape[0] == self.m
        ):
            # Prefer evaluating from x so line-search trial points work.
            pass
        x_list: list[torch.Tensor]
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == self.m:
            x_list = [x[i] for i in range(self.m)]
        elif self._cached_x_list is not None:
            x_list = self._cached_x_list
        else:
            raise ValueError(
                "MinibatchOracle.g expects stacked x of shape (m, ...) "
                "or a preceding solve_lower_level call."
            )
        g_acc = 0.0
        for i, x_i in enumerate(x_list):
            g_acc = g_acc + float(self.sample_oracles[i].g(x_i).item())
        return torch.tensor(
            g_acc / float(self.m),
            dtype=x_list[0].dtype,
            device=x_list[0].device,
        )

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        """Return a one-element tensor holding ``(1/m) sum ||grad g_i||``.

        MAID only uses ``grad_g(x).norm()`` in the line search. The mean of
        the per-sample gradient norms is the correct coefficient for the
        multi-sample ``U_upper`` / ``U_lower`` bounds (module docstring).
        """
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == self.m:
            x_list = [x[i] for i in range(self.m)]
        elif self._cached_x_list is not None:
            x_list = self._cached_x_list
        else:
            raise ValueError(
                "MinibatchOracle.grad_g expects stacked x of shape (m, ...)."
            )
        ng_acc = 0.0
        for i, x_i in enumerate(x_list):
            ng_acc = ng_acc + float(
                self.sample_oracles[i].grad_g(x_i).norm().item()
            )
        mean_ng = ng_acc / float(self.m)
        return torch.tensor(
            [mean_ng], dtype=x_list[0].dtype, device=x_list[0].device
        )

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        total = None
        for o in self.sample_oracles:
            fi = o.f_closed_form(theta)
            total = fi if total is None else total + fi
        assert total is not None
        return total / float(self.m)

    def update_lipschitz_estimates(
        self, lower: LowerLevelState, theta: torch.Tensor
    ) -> None:
        x_list = lower.extras.get("x_list")
        if x_list is None:
            return
        for i, o in enumerate(self.sample_oracles):
            lower_i = LowerLevelState(x=x_list[i], eps=lower.eps)
            o.update_lipschitz_estimates(lower_i, theta)


def make_quadratic_dataset(
    m: int,
    cond: float,
    n: int = 40,
    d: int = 4,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> list:
    """Build ``m`` independent section-4.1 problems with given condition number.

    Used by tests and the minibatch demo. Each sample has its own design
    matrices; the lower-level Hessian condition number is approximately
    ``cond`` for every sample.
    """
    from .quadratic_ls import QuadraticBilevelLS

    samples = []
    for i in range(m):
        gen = torch.Generator().manual_seed(int(seed) + i)

        def mat(gen=gen):
            G = torch.randn(n, d, generator=gen, dtype=dtype)
            Q, _ = torch.linalg.qr(G, mode="reduced")
            s = torch.logspace(
                0, torch.log10(torch.tensor(float(cond))), d, dtype=dtype
            )
            return Q * s

        samples.append(
            QuadraticBilevelLS(
                mat(),
                mat(),
                mat(),
                torch.randn(n, generator=gen, dtype=dtype),
                torch.randn(n, generator=gen, dtype=dtype),
            )
        )
    return samples


def wrap_smooth_dataset(
    problems: Sequence,
    *,
    goal_oriented: bool = False,
    safety_factor: float = 1.25,
    cg_budget: int = 5,
    recycle_krylov: bool = True,
) -> list[HypergradientOracle]:
    """Wrap each problem as a smooth (or goal-oriented) oracle."""
    from .smooth import GoalOrientedSmoothOracle, SmoothHypergradientOracle
    from .estimators import GoalOrientedEstimator

    oracles: list[HypergradientOracle] = []
    for p in problems:
        if goal_oriented:
            oracles.append(
                GoalOrientedSmoothOracle(
                    p,
                    estimator=GoalOrientedEstimator(
                        safety_factor=safety_factor,
                        cg_budget=cg_budget,
                        recycle_krylov=recycle_krylov,
                    ),
                )
            )
        else:
            oracles.append(SmoothHypergradientOracle(p))
    return oracles
