from typing import Union, Iterable
import torch
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
import numpy as np
from deepinv.transform.base import Transform, TransformParam
import itertools


class Reflect(Transform):
    r"""
    Reflect (flip) in random multiple axes.

    Generates ``n_trans`` reflected images, each time subselecting axes from dim (without replacement).
    Hence to transform through all group elements, set ``n_trans`` to ``2**len(dim)`` e.g ``Reflect(dim=[-2, -1], n_trans=4)``

    See :class:`deepinv.transform.Transform` for further details and examples.

    :param int, list[int] dim: axis or axes on which to randomly select axes to reflect.
    :param int n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(
        self,
        *args,
        dim: Union[int, list[int]] = [-2, -1],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def _get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate sets of reflection axes without replacement.

        :param torch.Tensor x: input image
        :return dict: keyword args with dims = tensor of which axes to flip, one row per n_trans, padded with nans.
        """
        subsets = list(
            itertools.chain.from_iterable(
                itertools.combinations(self.dim, r) for r in range(len(self.dim) + 1)
            )
        )
        idx = torch.randperm(len(subsets), generator=self.rng, device=self.rng.device)[
            : self.n_trans
        ]
        out = torch.full(
            (self.n_trans, len(self.dim)), fill_value=float("nan"), device=x.device
        )

        for i, id in enumerate(idx):
            out[i, : len(subsets[id])] = torch.tensor(subsets[id], dtype=torch.int)

        return {"dims": TransformParam(out, neg=lambda x: x)}

    def _transform(
        self,
        x: torch.Tensor,
        dims: Union[torch.Tensor, Iterable] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Reflect image in axes given in dim.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list dims: tensor with n_trans rows of axes to subselect for each reflected image. NaN axes are ignored.
        :return: torch.Tensor: transformed images.
        """
        dims = [dim[~torch.isnan(dim)].int().tolist() for dim in dims]

        return torch.cat(
            [torch.flip(x, dims=dim) if len(dim) > 0 else x for dim in dims]
        )
