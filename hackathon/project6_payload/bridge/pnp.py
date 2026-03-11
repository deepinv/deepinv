from __future__ import annotations

from typing import Callable

import torch


def _call_denoiser(denoiser: Callable | torch.nn.Module | None, x: torch.Tensor, strength: float):
    if denoiser is None:
        return x
    for kwargs in ({"ths": strength}, {"sigma": strength}, {}):
        try:
            return denoiser(x, **kwargs)
        except TypeError:
            continue
    return denoiser(x)


def _least_squares_gradient(physics, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return physics.A_adjoint(physics.A(x) - y)


def _poisson_gradient(
    physics,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    prediction = physics.A(x).clamp_min(epsilon)
    return physics.A_adjoint(1.0 - y / prediction)


def pnp_reconstruct(
    physics,
    y: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    denoiser: Callable | torch.nn.Module | None = None,
    num_iterations: int = 25,
    step_size: float = 1e-4,
    denoiser_strength: float = 0.02,
    data_fidelity: str = "least_squares",
    positivity: bool = True,
):
    """Simple plug-and-play gradient descent for SIRF-backed operators."""
    if x0 is None:
        x = physics.A_adjoint(y).detach()
    else:
        x = x0.detach().clone()
    history = []

    gradient_fn = _least_squares_gradient
    if data_fidelity == "poisson":
        gradient_fn = _poisson_gradient

    for iteration in range(num_iterations):
        gradient = gradient_fn(physics, x, y)
        x = x - step_size * gradient
        x = _call_denoiser(denoiser, x, denoiser_strength)
        if positivity:
            x = x.clamp_min(0)
        residual = torch.linalg.norm((physics.A(x) - y).reshape(-1)).item()
        history.append(
            {
                "iteration": iteration + 1,
                "residual_l2": residual,
                "estimate_norm": torch.linalg.norm(x.reshape(-1)).item(),
            }
        )
    return x.detach(), history

