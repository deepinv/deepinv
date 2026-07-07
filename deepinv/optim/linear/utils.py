from __future__ import annotations
import torch
from deepinv.utils.tensorlist import TensorList


def dot(
    a: torch.Tensor | TensorList, b: torch.Tensor | TensorList, dim: int
) -> torch.Tensor:
    r"""
    Computes the batched dot product between two tensors or two TensorLists along specified dimensions.

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


def _sym_ortho(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stable implementation of Givens rotation.

    Adapted from https://github.com/scipy/scipy/blob/v1.15.1/scipy/sparse/linalg/_isolve/lsqr.py

    The routine '_sym_ortho' was added for numerical stability. This is
    recommended by S.-C. Choi in "Iterative Methods for Singular Linear Equations and Least-Squares
    Problems".  It removes the unpleasant potential of
    ``1/eps`` in some important places.

    """
    a, b = torch.broadcast_tensors(a, b)

    zero_b = b == 0
    zero_a = a == 0
    big_b = b.abs() > a.abs()

    safe_a = torch.where(zero_a, torch.ones_like(a), a)
    safe_b = torch.where(zero_b, torch.ones_like(b), b)

    tau_bb = a / safe_b
    s_bb = torch.sign(b) / torch.sqrt(1 + tau_bb * tau_bb)
    c_bb = s_bb * tau_bb

    safe_s_bb = torch.where(s_bb == 0, torch.ones_like(s_bb), s_bb)
    r_bb = b / safe_s_bb

    tau_ab = b / safe_a
    c_ab = torch.sign(a) / torch.sqrt(1 + tau_ab * tau_ab)
    s_ab = c_ab * tau_ab

    safe_c_ab = torch.where(c_ab == 0, torch.ones_like(c_ab), c_ab)
    r_ab = a / safe_c_ab

    zeros = torch.zeros_like(a)

    c = torch.where(
        zero_b,
        torch.sign(a),
        torch.where(zero_a, zeros, torch.where(big_b, c_bb, c_ab)),
    )

    s = torch.where(
        zero_b,
        zeros,
        torch.where(zero_a, torch.sign(b), torch.where(big_b, s_bb, s_ab)),
    )

    r = torch.where(
        zero_b, a.abs(), torch.where(zero_a, b.abs(), torch.where(big_b, r_bb, r_ab))
    )

    return c, s, r
