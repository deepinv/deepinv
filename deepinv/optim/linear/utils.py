from __future__ import annotations
from deepinv.utils import TensorList
from deepinv.utils.compat import zip_strict


def dot(a, b, dim):
    if isinstance(a, TensorList):
        aux = 0
        for ai, bi in zip_strict(a.x, b.x):
            aux += (ai.conj() * bi).sum(
                dim=dim, keepdim=True
            )  # performs batched dot product
        dot = aux
    else:
        dot = (a.conj() * b).sum(dim=dim, keepdim=True)  # performs batched dot product
    return dot
