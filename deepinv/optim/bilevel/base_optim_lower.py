"""Drive DeepInverse :class:`~deepinv.optim.BaseOptim` with residual stopping.

MAID needs a distance bound ``||xbar - xhat|| <= eps``. That comes from a
residual that controls the optimality gap of the lower level, not from
``BaseOptim``'s default fixed-point residual
``||x_{k+1} - x_k|| / ||x_k||``.

Residual families
-----------------
* **Gradient residual** (smooth objectives: ``GD``, and smooth-prior ``PGD`` /
  ``FISTA`` when the prior has a gradient): stop when
  ``||grad_x h(x)|| <= eps * mu``. Strong convexity then gives
  ``||x - xhat|| <= eps``.
* **Proximal residual** (proximal methods: ``PGD``, ``FISTA``): stop when
  ``||x - prox_{gamma g}(x - gamma grad f(x))|| / gamma <= eps * mu``.
  This is the natural stationarity measure for composite
  ``f + lambda g`` and reduces to the gradient residual when ``g`` is
  smooth and the prox is a gradient step.

Solvers that cannot express either residual are not wired here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

ResidualKind = Literal["gradient", "proximal"]


@dataclass
class LowerLevelSolveResult:
    """Outcome of a residual-stopped BaseOptim run."""

    x: torch.Tensor
    residual: float
    residual_kind: ResidualKind
    n_iters: int
    converged: bool


def gradient_residual(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    physics,
) -> float:
    r"""``||grad data_fidelity + lambda * grad prior||`` at ``x``."""
    data_fidelity = model.update_data_fidelity_fn(0)
    prior = model.update_prior_fn(0)
    params = model.update_params_fn(0)
    grad_f = data_fidelity.grad(x, y, physics)
    lam = params.get("lambda", 1.0)
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam, dtype=x.dtype, device=x.device)
    grad_g = prior.grad(x, params.get("g_param"))
    return float((grad_f + lam * grad_g).flatten().norm().item())


def proximal_residual(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    physics,
) -> float:
    r"""``||x - prox_{gamma lambda g}(x - gamma grad f)|| / gamma``.

    This is the fixed-point residual of proximal gradient. For smooth ``g``
    with ``prox`` a gradient step it coincides with the gradient residual
    up to scaling by the stepsize.
    """
    data_fidelity = model.update_data_fidelity_fn(0)
    prior = model.update_prior_fn(0)
    params = model.update_params_fn(0)
    stepsize = params.get("stepsize", 1.0)
    if torch.is_tensor(stepsize):
        gamma = float(stepsize.detach().cpu().item())
    else:
        gamma = float(stepsize)
    if gamma <= 0.0:
        raise ValueError(f"stepsize must be positive, got {gamma}")
    lam = params.get("lambda", 1.0)
    if torch.is_tensor(lam):
        lam = float(lam.detach().cpu().item())
    g_param = params.get("g_param")
    grad_f = data_fidelity.grad(x, y, physics)
    z = x - gamma * grad_f
    # Match PGD: prox of (lambda * stepsize) * g at z.
    prox = prior.prox(z, g_param, gamma=lam * gamma)
    return float((x - prox).flatten().norm().item() / gamma)


def residual_kind_for_solver(solver_name: str) -> ResidualKind:
    """Default residual family for a DeepInverse optimiser name."""
    name = solver_name.upper()
    if name in {"GD", "MD"}:
        return "gradient"
    if name in {"PGD", "FISTA", "HQS"}:
        return "proximal"
    raise ValueError(
        f"No residual criterion wired for solver {solver_name!r}. "
        "Supported: GD (gradient residual), PGD and FISTA (proximal residual)."
    )


def solve_base_optim(
    model: Any,
    y: torch.Tensor,
    physics,
    *,
    residual_tol: float,
    residual_kind: ResidualKind | None = None,
    x_init: torch.Tensor | None = None,
    max_iter: int | None = None,
) -> LowerLevelSolveResult:
    r"""Run ``model`` until the residual falls below ``residual_tol``.

    Warm-starts from ``x_init`` when provided via the ``init`` argument of
    :meth:`deepinv.optim.BaseOptim.forward`, so MAID can reuse the previous
    lower-level solution.

    :param model: a :class:`~deepinv.optim.BaseOptim` instance (``GD``,
        ``PGD``, ``FISTA``, ...).
    :param y: measurement.
    :param physics: forward operator.
    :param residual_tol: absolute residual threshold (use ``eps * mu`` for
        the strong-convexity distance bound).
    :param residual_kind: ``"gradient"`` or ``"proximal"``. If None, inferred
        from the model class.
    :param x_init: warm start for the primal variable.
    :param max_iter: override ``model.max_iter`` for this call only.
    """
    if residual_kind is None:
        residual_kind = residual_kind_for_solver(type(model).__name__)
    if residual_kind not in ("gradient", "proximal"):
        raise ValueError(f"Unknown residual_kind {residual_kind!r}")

    max_it = int(max_iter if max_iter is not None else model.max_iter)
    residual_fn: Callable = (
        gradient_residual if residual_kind == "gradient" else proximal_residual
    )

    # Initialise the fixed-point state, optionally warm-started.
    # FISTA stores (x, z) with z the extrapolated point; warm-start both.
    if x_init is not None and type(model).__name__ in {"FISTA"}:
        init = (x_init, x_init.clone())
    else:
        init = x_init
    X = model.init_iterate_fn(y, physics, init=init)
    model.has_converged = False

    residual = residual_fn(model, model.get_output(X), y, physics)
    if residual <= residual_tol:
        return LowerLevelSolveResult(
            x=model.get_output(X),
            residual=residual,
            residual_kind=residual_kind,
            n_iters=0,
            converged=True,
        )

    n_iters = 0
    for it in range(max_it):
        cur_params = model.update_params_fn(it)
        cur_prior = model.update_prior_fn(it)
        cur_data_fidelity = model.update_data_fidelity_fn(it)
        X = model.fixed_point.iterator(
            X, cur_data_fidelity, cur_prior, cur_params, y, physics
        )
        n_iters += 1
        residual = residual_fn(model, model.get_output(X), y, physics)
        if residual <= residual_tol:
            model.has_converged = True
            return LowerLevelSolveResult(
                x=model.get_output(X),
                residual=residual,
                residual_kind=residual_kind,
                n_iters=n_iters,
                converged=True,
            )

    return LowerLevelSolveResult(
        x=model.get_output(X),
        residual=residual,
        residual_kind=residual_kind,
        n_iters=n_iters,
        converged=False,
    )


def build_solver(
    name: str,
    *,
    data_fidelity,
    prior,
    lambda_reg: float,
    stepsize: float,
    max_iter: int = 10_000,
    **kwargs,
) -> Any:
    """Construct a DeepInverse optimiser by name."""
    # Late import: optim/__init__ loads bilevel before GD/PGD/FISTA are bound.
    from deepinv.optim.optimizers import FISTA, GD, PGD

    name_u = name.upper()
    common = dict(
        data_fidelity=data_fidelity,
        prior=prior,
        lambda_reg=lambda_reg,
        stepsize=stepsize,
        max_iter=max_iter,
        early_stop=False,  # residual stopping is handled externally
        verbose=False,
        show_progress_bar=False,
    )
    common.update(kwargs)
    if name_u == "GD":
        return GD(**common)
    if name_u == "PGD":
        return PGD(**common)
    if name_u == "FISTA":
        # FISTA needs the momentum parameter ``a`` (default 3).
        if "params_algo" not in common:
            common["params_algo"] = {
                "lambda": lambda_reg,
                "stepsize": stepsize,
                "g_param": common.get("g_param"),
                "a": 3,
            }
        return FISTA(**common)
    raise ValueError(
        f"Unsupported solver {name!r}. Supported: 'GD', 'PGD', 'FISTA'."
    )
