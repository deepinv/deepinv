import torch
import warnings


def power_method(
    operator,
    x0: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False,
    rng: torch.Generator | None = None,
    **kwargs,
):
    r"""
    Runs the power iteration method to estimate the largest singular value of a linear operator.

    :param Callable operator: function that applies the linear operator.
    :param torch.Tensor x0: initial vector for the power iteration.
    :param int max_iter: maximum number of iterations. Default: ``100``.
    :param float tol: tolerance for convergence. Default: ``1e-6``.
    :param bool verbose: if ``True``, prints convergence information. Default: ``False``.
    :param torch.Generator rng: random number generator for the initial vector. If ``None``, the global generator is used. Default: ``None``.
    :param kwargs: additional arguments to be passed to the operator.
    :returns: (:class:`torch.Tensor`) Estimated largest singular value of the operator.
    """
    x = torch.randn(x0.shape, generator=rng, dtype=x0.dtype, device=x0.device)
    x /= torch.linalg.vector_norm(x)
    zold = torch.zeros_like(x)

    for it in range(max_iter):
        y = operator(x, **kwargs)
        z = torch.vdot(x.flatten(), y.flatten()) / torch.linalg.vector_norm(x) ** 2

        rel_var = torch.linalg.vector_norm(z - zold)
        if rel_var < tol:
            if verbose:
                print(
                    f"Power iteration converged at iteration {it}, ||A^T A||_2={z.real.item():.2f}"
                )
            break
        zold = z
        x = y / torch.linalg.vector_norm(y)
    else:
        warnings.warn("Power iteration: convergence not reached")

    return z.real
