from __future__ import annotations
from typing import TYPE_CHECKING
from deepinv.utils.tensorlist import TensorList

if TYPE_CHECKING:
    from torch import Tensor


def dot(a: Tensor | TensorList, b: Tensor | TensorList, dim: int) -> Tensor:
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
