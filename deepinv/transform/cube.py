from __future__ import annotations

from itertools import permutations, product

import torch

from deepinv.transform.base import Transform, TransformParam

# 48 cube symmetries: permutation of the spatial axes followed by reflections
_SYMMETRIES = [
    (perm, flips)
    for perm in permutations(range(3))
    for flips in product((False, True), repeat=3)
]


def _permutation_sign(perm):
    sign = 1
    perm = list(perm)
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                sign = -sign
    return sign


def _apply(x: torch.Tensor, index: int, fourier: bool = False) -> torch.Tensor:
    perm, flips = _SYMMETRIES[index]
    n = x.dim()
    x = x.permute(*range(n - 3), *(n - 3 + p for p in perm))
    dims = [n - 3 + i for i in range(3) if flips[i]]
    if dims:
        x = torch.flip(x, dims=dims)
        if fourier:
            # on the unshifted FFT grid, k -> -k maps index i to -i (mod n)
            x = torch.roll(x, shifts=[1] * len(dims), dims=dims)
    return x


def _inverse_table() -> list[int]:
    ref = torch.arange(27).reshape(3, 3, 3)
    table = []
    for i in range(len(_SYMMETRIES)):
        xi = _apply(ref, i)
        table.append(
            next(j for j in range(len(_SYMMETRIES)) if torch.equal(_apply(xi, j), ref))
        )
    return table


_INVERSES = _inverse_table()


class CubeSymmetry(Transform):
    r"""
    Symmetries of the cube for 3D volumes.

    Applies one of the 48 symmetries of the cube (rotations by multiples of :math:`90^\circ` and reflections) to volumes of shape `(B, C, D, H, W)`.
    These transforms are exact (no interpolation) and form a group, e.g. for equivariant imaging as in :footcite:t:`kishore2026icecream`. Volumes should be cubic, as permuting axes changes the shape.

    The transform parameter is the symmetry index `cube_symmetry` in :math:`\{0, \dots, 47\}`. See :class:`deepinv.transform.Transform` for details.

    :param list[int] symmetries: indices of the symmetries to sample from. Default: all symmetries, or all rotations if `rotations_only=True`.
    :param bool rotations_only: if `True`, samples only the 24 rotations.
    :param int n_trans: number of transformed versions per input volume.
    :param torch.Generator rng: random number generator. Default: :class:`torch.Generator`.

    |sep|

    :Example:

        >>> import torch
        >>> from deepinv.transform import CubeSymmetry
        >>> x = torch.randn(1, 1, 4, 4, 4)
        >>> transform = CubeSymmetry()
        >>> params = transform.get_params(x)
        >>> torch.equal(transform.inverse(transform(x, **params), **params), x)
        True
    """

    def __init__(
        self,
        *args,
        symmetries: list[int] | None = None,
        rotations_only: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("flatten_video_input", False)
        super().__init__(*args, **kwargs)
        if symmetries is None:
            symmetries = [
                i
                for i in range(len(_SYMMETRIES))
                if not rotations_only or self.determinant(i) == 1
            ]
        if len(symmetries) == 0:
            raise ValueError("At least one symmetry must be given.")
        self.symmetries = [int(s) for s in symmetries]

    @staticmethod
    def determinant(index: int) -> int:
        r"""
        Determinant of a symmetry: :math:`+1` for rotations, :math:`-1` for reflections.

        :param int index: symmetry index.
        :return: (int) determinant.
        """
        perm, flips = _SYMMETRIES[index]
        return _permutation_sign(perm) * (-1) ** sum(flips)

    @staticmethod
    def apply(x: torch.Tensor, index: int, fourier: bool = False) -> torch.Tensor:
        r"""
        Applies a symmetry to the last three dimensions of a tensor.

        :param torch.Tensor x: tensor of shape `(..., D, H, W)`.
        :param int index: symmetry index.
        :param bool fourier: if `True`, `x` is sampled on the unshifted :func:`torch.fft.fftn` grid (e.g. a Fourier mask) and transformed as the spectrum of a transformed volume.
        :return: (:class:`torch.Tensor`) transformed tensor.
        """
        return _apply(x, int(index), fourier=fourier)

    def _get_params(self, x: torch.Tensor) -> dict:
        idx = torch.randint(
            len(self.symmetries),
            (self.n_trans,),
            generator=self.rng,
            device=self.rng.device,
        )
        symmetries = torch.tensor(self.symmetries, device=self.rng.device)[idx]
        return {"cube_symmetry": symmetries}

    def invert_params(self, params: dict) -> dict:
        inverses = torch.tensor(_INVERSES)
        return {
            k: inverses[torch.as_tensor(v).long().cpu()].to(torch.as_tensor(v).device)
            for k, v in params.items()
        }

    def _transform(
        self,
        x: torch.Tensor,
        cube_symmetry: torch.Tensor | list | TransformParam = tuple(),
        **kwargs,
    ) -> torch.Tensor:
        """
        Transforms volumes given symmetry indices.

        :param torch.Tensor x: volumes of shape `(B, C, D, H, W)`.
        :param torch.Tensor, list cube_symmetry: symmetry indices, one per `n_trans`.
        :return: (:class:`torch.Tensor`) transformed volumes.
        """
        if x.dim() != 5:
            raise ValueError(
                f"CubeSymmetry expects volumes of shape (B, C, D, H, W), got {tuple(x.shape)}."
            )
        return torch.cat([_apply(x, int(s)) for s in cube_symmetry])
