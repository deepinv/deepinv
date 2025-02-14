from typing import Union, Iterable
from itertools import zip_longest
import torch
from deepinv.transform.base import Transform, TransformParam


class Shift(Transform):
    r"""
    Fast integer 2D translations.

    Generates ``n_trans`` randomly shifted versions of 2D images with circular padding.

    See :class:`deepinv.transform.Transform` for further details and examples.

    :param float shift_max: maximum shift as fraction of total height/width.
    :param int n_trans: number of transformed versions generated per input image.
    :param torch.Generator rng: random number generator, if None, use torch.Generator(), defaults to None
    """

    def __init__(self, *args, shift_max=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_max = shift_max

    def _get_params(self, x: torch.Tensor) -> dict:
        """Randomly generate shift parameters.

        :param torch.Tensor x: input image
        :return dict: keyword args of shift parameters
        """
        H, W = x.shape[-2:]
        assert self.n_trans <= H - 1 and self.n_trans <= W - 1

        H_max, W_max = int(self.shift_max * H), int(self.shift_max * W)

        x_shift = (
            torch.arange(-H_max, H_max, device=self.rng.device)[
                torch.randperm(2 * H_max, generator=self.rng, device=self.rng.device)
            ][: self.n_trans]
            if H_max > 0
            else torch.zeros(self.n_trans, device=x.device)
        )
        y_shift = (
            torch.arange(-W_max, W_max, device=self.rng.device)[
                torch.randperm(2 * W_max, generator=self.rng, device=self.rng.device)
            ][: self.n_trans]
            if W_max > 0
            else torch.zeros(self.n_trans, device=x.device)
        )

        return {"x_shift": x_shift, "y_shift": y_shift}

    def _transform(
        self,
        x: torch.Tensor,
        x_shift: Union[torch.Tensor, Iterable, TransformParam] = [],
        y_shift: Union[torch.Tensor, Iterable, TransformParam] = [],
        **kwargs,
    ) -> torch.Tensor:
        """Shift image given shift parameters.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list x_shift: iterable of shifts in x direction, one per ``n_trans``.
        :param torch.Tensor, list y_shift: iterable of shifts in y direction, one per ``n_trans``.
        :return: torch.Tensor: transformed image.
        """
        return torch.cat(
            [
                torch.roll(x, [sx, sy], [-2, -1])
                for sx, sy in zip_longest(x_shift, y_shift, fillvalue=0)
            ],
            dim=0,
        )
