from typing import Callable
from torch import zeros_like
from torch.nn import Module
import torch


class TimeMixin:
    r"""
    Base class for temporal capabilities for physics and models.

    Implements various methods to add or remove the time dimension.

    Also provides template methods for temporal physics to implement.
    """

    @staticmethod
    def flatten(x: torch.Tensor) -> torch.Tensor:
        """Flatten time dim into batch dim.

        Lets non-dynamic algorithms process dynamic data by treating time frames as batches.

        :param x: input tensor of shape (B, C, T, H, W)
        :return: output tensor of shape (B*T, C, H, W)
        """
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def unflatten(x: torch.Tensor, batch_size=1) -> torch.Tensor:
        """Creates new time dim from batch dim. Opposite of ``flatten``.

        :param x: input tensor of shape (B*T, C, H, W)
        :param int batch_size: batch size, defaults to 1
        :return: output tensor of shape (B, C, T, H, W)
        """
        BT, C, H, W = x.shape
        return x.reshape(batch_size, BT // batch_size, C, H, W).permute(0, 2, 1, 3, 4)

    @staticmethod
    def flatten_C(x: torch.Tensor) -> torch.Tensor:
        """Flatten time dim into channel dim.

        Use when channel dim doesn't matter and you don't want to deal with annoying batch dimension problems (e.g. for transforms).

        :param x: input tensor of shape (B, C, T, H, W)
        :return: output tensor of shape (B, C*T, H, W)
        """
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

    @staticmethod
    def wrap_flatten_C(f: Callable[[torch.Tensor], torch.Tensor]) -> Callable:
        """Flatten time dim into channel dim, apply function, then unwrap.

        The first argument is assumed to be the tensor to be flattened.

        :param f: function to be wrapped
        :return: wrapped function
        """

        def wrapped(x: torch.Tensor, *args, **kwargs):
            """
            :param torch.Tensor x: input tensor of shape (B, C, T, H, W)
            :return: torch.Tensor output tensor of shape (B, C, T, H, W)
            """
            return f(TimeMixin.flatten_C(x), *args, **kwargs).reshape(
                -1, x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            )

        return wrapped

    @staticmethod
    def average(
        x: torch.Tensor, mask: torch.Tensor = None, dim: int = 2
    ) -> torch.Tensor:
        """Flatten time dim of x by averaging across frames.

        If mask is non-overlapping in time dim, then this will simply be the sum across frames.

        :param x: input tensor of shape `(B,C,T,H,W)` (e.g. time-varying k-space)
        :param mask: mask showing where ``x`` is non-zero. If not provided, then calculated from ``x``.
        :param dim: time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
        :return: flattened tensor with time dim removed of shape `(B,C,H,W)`
        """
        _x = x.sum(dim)
        out = zeros_like(_x)
        m = mask if mask is not None else (x != 0)
        m = m.sum(dim)
        out[m != 0] = _x[m != 0] / m[m != 0]
        return out

    @staticmethod
    def repeat(x: torch.Tensor, target: torch.Tensor, dim: int = 2) -> torch.Tensor:
        """Repeat static image across new time dim T times. Opposite of ``average``.

        :param x: input tensor of shape `(B,C,H,W)`
        :param target: any tensor of desired shape `(B,C,T,H,W)`
        :param dim: time dimension, defaults to 2 (i.e. shape `B,C,T,H,W`)
        :return: tensor with new time dim of shape `(B,C,T,H,W)`
        """
        return x.unsqueeze(dim=dim).expand_as(target)

    def to_static(self) -> Module:
        raise NotImplementedError()
