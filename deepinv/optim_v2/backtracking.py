import torch
import torch.nn as nn


class BacktrackingError(RuntimeError):
    r"""Raised when a backtracking strategy fails to find an acceptable stepsize."""


class Backtracking(nn.Module):
    r"""
    Base class for backtracking algorithms.
    """

    def forward(self, x, stepsize, step, objective, objective_x=None):
        r"""
        :param torch.Tensor x: current iterate.
        :param stepsize: initial stepsize.
        :param Callable step: returns the next iteration candidate given a stepsize: ``step(stepsize) -> candidate``.
        :param Callable objective: returns the objective value for a given candidate: ``objective(candidate) -> value``.
        :param torch.Tensor objective_x: previous objective value at ``x``. If not provided (`None`), it will be computed as ``objective(x)``.
        :return: accepted candidate, its stepsize, and its objective value.
        """
        raise NotImplementedError


class Armijo(Backtracking):
    r"""
    Armijo backtracking: accept the candidate when
    :math:`F(x) - F(x^+) \geq \frac{c}{\gamma}\|x^+ - x\|^2`, otherwise :math:`\gamma \leftarrow \rho\gamma`.

    The decrease is averaged over the batch, so a single stepsize is accepted for the whole batch.

    :param float decrease_factor: :math:`c`. Default: ``0.1``.
    :param float reduction_factor: :math:`\rho`. Default: ``0.5``.
    :param int max_backtracking_steps: maximum number of backtracking steps. Default: ``10``.
    :param bool verbose: whether to print the stepsize reduction. Default: ``False``.
    """

    def __init__(
        self,
        decrease_factor=0.1,
        reduction_factor=0.5,
        max_backtracking_steps=10,
        verbose=False,
    ):
        super().__init__()
        self.decrease_factor = decrease_factor
        self.reduction_factor = reduction_factor
        self.max_backtracking_steps = max_backtracking_steps
        self.verbose = verbose

    def forward(self, x, stepsize, step, objective, objective_x=None):
        r"""
        :param torch.Tensor x: current iterate.
        :param stepsize: initial stepsize.
        :param Callable step: returns the next iteration candidate given a stepsize: ``step(stepsize) -> candidate``.
        :param Callable objective: returns the objective value for a given candidate: ``objective(candidate) -> value``.
        :param torch.Tensor objective_x: previous objective value at ``x``. If not provided (`None`), it will be computed as ``objective(x)``.
        :return: accepted candidate, its stepsize, and its objective value.
        """
        if objective_x is None:
            with torch.no_grad():
                objective_x = objective(x)
        for _ in range(self.max_backtracking_steps):
            candidate = step(stepsize)
            with torch.no_grad():
                objective_candidate = objective(candidate)
                squared_norm = (candidate - x).flatten(1).square().sum(dim=1).mean()
                if (objective_x - objective_candidate).mean() >= (
                    self.decrease_factor / stepsize
                ) * squared_norm:
                    return candidate, stepsize, objective_candidate
            stepsize = stepsize * self.reduction_factor
            if self.verbose:
                print(f"Backtracking: stepsize reduced to {float(stepsize):.3e}")
        raise BacktrackingError(
            f"Armijo backtracking failed after {self.max_backtracking_steps} steps."
        )
