from __future__ import annotations
from typing import Iterable
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
            torch.arange(-W_max, W_max, device=self.rng.device)[
                torch.randperm(2 * W_max, generator=self.rng, device=self.rng.device)
            ][: self.n_trans]
            if W_max > 0
            else torch.zeros(self.n_trans, device=x.device, dtype=torch.long)
        )
        y_shift = (
            torch.arange(-H_max, H_max, device=self.rng.device)[
                torch.randperm(2 * H_max, generator=self.rng, device=self.rng.device)
            ][: self.n_trans]
            if H_max > 0
            else torch.zeros(self.n_trans, device=x.device, dtype=torch.long)
        )

        return {"x_shift": x_shift, "y_shift": y_shift}

    def _transform(
        self,
        x: torch.Tensor,
        x_shift: torch.Tensor | Iterable | TransformParam = tuple(),
        y_shift: torch.Tensor | Iterable | TransformParam = tuple(),
        **kwargs,
    ) -> torch.Tensor:
        """Shift image given shift parameters.

        :param torch.Tensor x: input image of shape (B,C,H,W)
        :param torch.Tensor, list x_shift: iterable of shifts in x direction, one per ``n_trans``.
        :param torch.Tensor, list y_shift: iterable of shifts in y direction, one per ``n_trans``.
        :return: torch.Tensor: transformed image.
        """
        # Convert input and params to tensors
        x_shift = torch.as_tensor(x_shift, device=x.device, dtype=torch.long)
        y_shift = torch.as_tensor(y_shift, device=x.device, dtype=torch.long)

        # Pad x_shift and y_shift in case they're not the same length
        N_y = y_shift.shape[0]
        N_x = x_shift.shape[0]
        N = max(N_y, N_x)
        _y_shift = torch.zeros(N, device=x.device, dtype=y_shift.dtype)
        _x_shift = torch.zeros(N, device=x.device, dtype=x_shift.dtype)
        _y_shift[:N_y] = y_shift
        _x_shift[:N_x] = x_shift
        y_shift = _y_shift
        x_shift = _x_shift

        # Prepare input and params for batch-wise transform
        B = x.shape[0]
        x = x.repeat(N, *((x.ndim - 1) * [1]))
        y_shift = y_shift.repeat_interleave(B, dim=0)
        x_shift = x_shift.repeat_interleave(B, dim=0)

        # Build the indices for torch.gather
        shape = x.shape
        B, H, W = shape[0], shape[-2], shape[-1]
        index_y = torch.arange(H, device=x.device)
        index_x = torch.arange(W, device=x.device)
        index_y = index_y.view(1, -1).repeat(B, 1)
        index_x = index_x.view(1, -1).repeat(B, 1)
        y_shift = y_shift.view(-1, 1).repeat(1, H)
        x_shift = x_shift.view(-1, 1).repeat(1, W)
        index_y = (index_y - y_shift) % H
        index_x = (index_x - x_shift) % W
        index_y = index_y.view(B, *((x.ndim - 3) * [1]), H, 1).expand(shape)
        index_x = index_x.view(B, *((x.ndim - 3) * [1]), 1, W).expand(shape)

        # Apply the shifts
        return x.gather(dim=-2, index=index_y).gather(dim=-1, index=index_x)
