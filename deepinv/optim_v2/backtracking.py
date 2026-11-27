class Backtracking:
    r"""
    Base class for backtracking rules on the stepsize.

    A rule decides whether a candidate is accepted and how to update the stepsize otherwise.
    The search itself is performed by the algorithm.

    :param int max_steps: maximum number of candidates tried at each iteration.
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps

    def accept(self, x, candidate, stepsize, **kwargs) -> bool:
        r"""Whether ``candidate``, obtained from ``x`` with ``stepsize``, is accepted."""
        raise NotImplementedError

    def update(self, stepsize):
        r"""Stepsize to try after a rejected candidate."""
        raise NotImplementedError


class Armijo(Backtracking):
    r"""
    Armijo rule: accept the candidate when
    :math:`F(x) - F(x^+) \geq \frac{c}{\gamma}\|x^+ - x\|^2`, otherwise :math:`\gamma \leftarrow \rho\gamma`.

    The decrease is averaged over the batch, so a single stepsize is accepted for the whole batch.

    :param float decrease_factor: :math:`c`. Default: ``0.1``.
    :param float reduction_factor: :math:`\rho`. Default: ``0.5``.
    :param int max_steps: maximum number of candidates tried at each iteration. Default: ``10``.
    """

    def __init__(self, decrease_factor=0.1, reduction_factor=0.5, max_steps=10):
        super().__init__(max_steps)
        self.decrease_factor = decrease_factor
        self.reduction_factor = reduction_factor

    def accept(self, x, candidate, stepsize, objective_x, objective_candidate, **kwargs):
        squared_norm = (candidate - x).flatten(1).square().sum(dim=1).mean()
        decrease = (objective_x - objective_candidate).mean()
        return bool(decrease >= self.decrease_factor / stepsize * squared_norm)

    def update(self, stepsize):
        return stepsize * self.reduction_factor
