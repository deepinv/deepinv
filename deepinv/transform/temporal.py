from typing import Union, Iterable
import torch
from deepinv.transform.base import Transform, TransformParam


class ShiftTime(Transform):
    r"""
    Shift a video in time with reflective padding.

    Generates ``n_trans`` randomly transformed versions.

    See :class:`deepinv.transform.Transform` for further details and examples.

    :param int n_trans: number of transformed versions generated per input image.
    :param str padding: ``"reflect"`` performs reflective padding, ``"wrap"`` performs wrap padding (i.e. roll)
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, padding="reflect", **kwargs):
        super().__init__(*args, **kwargs)
        self.flatten_video_input = False
        self.padding = padding
        assert self.padding in ("reflect", "wrap"), "padding must be reflect or wrap."

    def roll_reflect_1d(self, x: torch.Tensor, by: int = 0, dim: int = 0):
        """Roll in one dimension with reflect padding.

        :param torch.Tensor x: input image
        :param int by: amount to roll by, defaults to 0
        :param int dim: dimension to roll, defaults to 0
        """

        def arange(*args):
            return torch.arange(*args, device=x.device)

        T = x.shape[dim]
        by %= T * 2 - 2
        if by > T - 1:
            by -= T * 2 - 2

        x_flip = torch.flip(x, dims=[dim])
        x_pad = torch.cat(
            [
                x_flip.index_select(dim, arange(T - 1)),
                x,
                x_flip.index_select(dim, arange(1, T)),
            ],
            dim=dim,
        )
        return torch.roll(x_pad, [by], [dim]).index_select(
            dim, arange(T - 1, T * 2 - 1)
        )

    def _get_params(self, x: torch.Tensor) -> dict:
        amounts = torch.randperm(x.shape[-3] * 2, generator=self.rng) - x.shape[-3]
        amounts = amounts[: self.n_trans]
        return {"amounts": amounts}

    def _transform(
        self,
        x: torch.Tensor,
        amounts: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        roll = torch.roll if self.padding == "wrap" else self.roll_reflect_1d
        return torch.cat(
            [
                roll(x, by.item() if isinstance(by, torch.Tensor) else by, -3)
                for by in amounts
            ],
            dim=0,
        )
