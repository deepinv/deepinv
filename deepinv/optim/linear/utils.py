from __future__ import annotations
from typing import Callable
import torch
from deepinv.utils.tensorlist import TensorList, zeros_like, ones_like


def _dot(a: torch.Tensor | TensorList, b: torch.Tensor, dim: int) -> torch.Tensor:
    r"""
    Computes the batched dot product between two tensors or two TensorLists along specified dimensions.
    Used in `'minres'`.

    :param torch.Tensor, deepinv.utils.TensorList a: First input tensor or TensorList.
    :param torch.Tensor, deepinv.utils.TensorList b: Second input tensor or TensorList.
    :param int dim: Dimensions along which to compute the dot product.
    :return: (:class:`torch.Tensor`) The batched dot product of a and b along the specified dimensions.
    """
    if isinstance(a, TensorList):
        aux = 0
        for ai, bi in zip(a.x, b.x, strict=True):
            aux += (ai.conj() * bi).sum(
                dim=dim, keepdim=True
            )  # performs batched dot product
        dot = aux
    else:
        dot = (a.conj() * b).sum(dim=dim, keepdim=True)  # performs batched dot product
    return dot


def _batched_dot(
    a: torch.Tensor | TensorList,
    b: torch.Tensor | TensorList,
    parallel_dim: None | int | list[int] = 0,
) -> torch.Tensor:
    """
    Computes the batched dot product between two tensors or two TensorLists along specified dimensions.
    Solves issue ``_dot`` has with singleton dimensions on TensorLists.
    Used in solvers with TensorList support `'CG'`, `'BiCGStab'`, `'lsqr'`, and `'lsmr'`.

    :param torch.Tensor, deepinv.utils.TensorList a: First input tensor or TensorList.
    :param torch.Tensor, deepinv.utils.TensorList b: Second input tensor or TensorList.
    :param None, int, list[int] parallel: Dimensions to consider as batch dimensions, along which.
    :return: (:class:`torch.Tensor`) The batched dot product of a and b along the specified dimensions.
    """

    def _one(ai, bi):
        """
        Dot product over non-batch dimensions of two tensors.

        :param torch.Tensor ai: First input tensor.
        :param torch.Tensor bi: Second input tensor.
        :return: (:class:`torch.Tensor`) Dot product of ``ai`` and ``bi``.
        """
        prod = ai.conj() * bi
        dims = _reduce_dims(ai, parallel_dim)  # each block uses its own dims
        return prod.sum(dim=dims) if dims else prod

    if isinstance(a, TensorList):
        total = 0
        for ai, bi in zip(a, b, strict=True):
            total = total + _one(ai, bi)  # out-of-place, common batch shape
        return total
    return _one(a, b)


def _resolve_stagtol(stagtol: float | None, b: torch.Tensor | TensorList) -> float:
    """
    Provide the stagnation tolerance in `'CG'`, `'BiCGStab'`, `'minres'`, `'lsqr'`, and `'lsmr'` if no explicit tolerance is passed as an argument.

    :param None, float stagtol: Stagnation tolerance given as an argument to the solver.
    :param torch.Tensor b: Tensor to provide the used dtype.
    :return: (:class:`float`) Stagnation tolerance to be used by the solver.
    """
    if stagtol is None:
        return 8.0 * torch.finfo(b.dtype).eps
        # the 8 approximates stagtol to 1e-6 which has performed well in testing
    return stagtol


def _as_dim_list(parallel_dim: None | int | list[int]) -> list[int]:
    """
    Normalise the ``parallel_dim`` argument to a list of batch dimensions.

    :param None, int, list[int] parallel_dim: Dimension to be considered as batch dimensions.
    :return: (:class:`list`) Input as a list.
    """
    if isinstance(parallel_dim, int):
        return [parallel_dim]
    if parallel_dim is None:
        return []
    return list(parallel_dim)


def _reduce_dims(t: torch.Tensor | TensorList, parallel_dim: list[int]) -> list[int]:
    """
    List of dimensions of a tensor or TensorList which are not batch dimensions.

    :param torch.Tensor, TensorList t: Input Tensor or TensorList.
    :param list[int] parallel_dim: Dimensions of ``t`` to be considered as batch dimensions.
    :return: (:class:`list`) Non-batch dimensions of ``t``.
    """
    ref = t[0] if isinstance(t, TensorList) else t
    return [i for i in range(ref.ndim) if i not in parallel_dim]


def _all_zero(x: torch.Tensor | TensorList) -> torch.Tensor | bool:
    """Whether every entry of a given tensor or TensorList is zero in `'lsqr'` and `'lsmr'`.

    :param torch.Tensor, TensorList x: Input tensor or TensorList.
    :return: (:class:`bool`) Whether or not ``x`` is all zeroes.
    """
    if isinstance(x, TensorList):
        return all(torch.all(xi == 0) for xi in x)
    return torch.all(x == 0)


def _batched_norm(
    u: torch.Tensor | TensorList, parallel_dim: list[int]
) -> torch.Tensor:
    """
    Euclidean norm reduced over every non-batch dimension with batch dimensions kept.

    :param torch.Tensor, TensorList u: Input Tensor or TensorList.
    :param list[int] parallel_dim: Dimensions of ``u`` to be considered as batch dimensions.
    :return: (:class:`torch.Tensor`) The norm of ``u`` with batch dimensions kept intact.
    """
    if isinstance(u, TensorList):
        block_norms = torch.stack(
            [
                torch.linalg.vector_norm(
                    uk, dim=_reduce_dims(uk, parallel_dim), keepdim=False
                )
                for uk in u
            ],
            dim=0,
        )  # shape (K, B): one norm per block per sample
        return torch.linalg.vector_norm(block_norms, dim=0)  # (B,)
    return torch.linalg.vector_norm(u, dim=_reduce_dims(u, parallel_dim), keepdim=False)


def _sample_shape(t: torch.Tensor | TensorList, parallel_dim: list[int]) -> list:
    """
    Shape of ``t`` with every non-batch dimension collapsed to 1.

    :param torch.Tensor, TensorList t: Input Tensor or TensorList.
    :param list[int] parallel_dim: Dimensions of ``t`` to be considered as batch dimensions.
    :return: (:class:`list`) Dimensions of ``t`` with non-batch dimensions as 1.
    """
    if isinstance(t, TensorList):
        return [
            [s if i in parallel_dim else 1 for i, s in enumerate(tk.shape)] for tk in t
        ]
    return [s if i in parallel_dim else 1 for i, s in enumerate(t.shape)]


def _scalar(v: torch.Tensor | TensorList, alpha: torch.Tensor, shape: list):
    """
    Scalar product used in `'CG'`, `'BiCGStab'`, `'lsqr'`, and `'lsmr'`. ``alpha`` is broadcast onto domain of measurement ``b``.

    :param torch.Tensor, Tensorlist v: Input to be scaled. If TensorList, each element is scaled independently.
    :param torch.Tensor alpha: Per-sample scalars.
    :param list shape: Shape which ``alpha`` is broadcast onto.
    :return: (:class:`torch.Tensor` or :class:`TensorList`) Scaled ``v`` keeping its shape and type.
    """

    if isinstance(v, TensorList):
        return TensorList(
            [vi * alpha.view(si_shape) for vi, si_shape in zip(v, shape, strict=True)]
        )
    return v * alpha.view(shape)


def _init_lsq_solution(
    x0: None | float | torch.Tensor | TensorList,
    b: torch.Tensor | TensorList,
    AT: Callable,
) -> tuple:
    r"""
    Set up the initial iterate shared by `'lsqr'` and `'lsmr'` from the ``x0`` argument.

    :param None, float, torch.Tensor, TensorList x0: Initial guess given to the solver as argument.
    :param torch.Tensor, TensorList b: Input tensor or TensorList.
    :param Callable: Adjoint operator as a callable function.
    :return: (:class:`tuple) ``(x, xt, x_ref)`` where ``x`` is the initial solution, ``xt`` is
        the pre-computed :math:`A^\top b` (``None`` when ``x0`` is an explicit tensor,
        as it is then not needed), and ``x_ref`` is a signal-domain tensor whose
        shape is used to broadcast per-sample scalars.
    """
    if x0 is None:
        xt = AT(b)
        return zeros_like(xt), xt, xt
    if isinstance(x0, float):
        xt = AT(b)
        return x0 * ones_like(xt), xt, xt
    x = x0.clone()
    return x, None, x


def _validate_eta(
    eta: None | float | torch.Tensor,
    b: torch.Tensor | TensorList,
    device: torch.device,
) -> torch.Tensor:
    r"""
    Validate and normalise the damping parameter :math:`\eta \geq 0` for `'lsqr'` and `'lsmr`'.

    ``None`` is set to 0 and non-tensors are converted to tensors.
    Raises ``ValueError`` if a batched ``eta`` does not match the batch size of ``b``, or if ``eta`` is negative.

    :param float, None, torch.Tensor eta: Damping parameter.
    :param torch.Tensor, deepinv.utils.TensorList b: Input tensor or TensorList.
    :param torch.device device: Device on which ``eta`` is created if it is not a tensor.
    :return: (:class:`torch.Tensor`)  Damping parameter ``eta`` as tensor.
    """
    if eta is None:
        eta = 0.0
    if not isinstance(eta, torch.Tensor):
        eta = torch.tensor(eta, device=device)
    if eta.ndim > 0:  # if batched eta
        batch_size = b[0].size(0) if isinstance(b, TensorList) else b.size(0)
        if eta.size(0) != batch_size:
            raise ValueError(
                "If eta is batched, its batch size must match the one of b."
            )
        eta = eta.squeeze()  # ensure eta has ndim as b
    if torch.any(eta < 0):
        raise ValueError(
            "Damping parameter eta must be non-negative. The least squares solver "
            "cannot be applied to problems with negative eta."
        )
    return eta


def _sym_ortho(a: torch.Tensor, b: torch.Tensor) -> tuple:
    """
    Batched Givens rotation.

    :param torch.Tensor a: First input, broadcast with ``b``.
    :param torch.Tensor b: Second input, broadcast with ``a``.
    :return: (:class:`tuple`) Rotated Tensors.
    """
    a, b = torch.broadcast_tensors(a, b)

    big_b = b.abs() > a.abs()
    p = torch.where(big_b, b, a)  # larger magnitude
    q = torch.where(big_b, a, b)  # smaller magnitude

    safe_p = torch.where(p == 0, 1.0, p)  # p == 0 only when both are 0
    tau = q / safe_p
    root = torch.sqrt(1 + tau * tau)

    w = torch.sign(p) / root
    t = w * tau

    c = torch.where(big_b, t, w)
    s = torch.where(big_b, w, t)
    r = p.abs() * root
    return c, s, r
