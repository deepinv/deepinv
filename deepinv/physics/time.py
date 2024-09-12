from typing import Callable
from torch import zeros_like, Tensor
from torch.nn import Module


class TimeMixin:
    r"""
    Base class for temporal capabilities for physics and models.

    Implements various methods to add or remove the time dimension.

    Also provides template methods for temporal physics to implement.
    """

    @staticmethod
    def flatten(x: Tensor) -> Tensor:
        """Flatten time dim into batch dim.

        Lets non-dynamic algorithms process dynamic data by treating time frames as batches.

        :param Tensor x: input tensor of shape (B, C, T, H, W)
        :return Tensor: output tensor of shape (B*T, C, H, W)
        """
        B, C, T, H, W = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    @staticmethod
    def unflatten(x: Tensor, batch_size=1) -> Tensor:
        """Creates new time dim from batch dim. Opposite of ``flatten``.

        :param Tensor x: input tensor of shape (B*T, C, H, W)
        :param int batch_size: batch size, defaults to 1
        :return Tensor: output tensor of shape (B, C, T, H, W)
        """
        BT, C, H, W = x.shape
        return x.reshape(batch_size, BT // batch_size, C, H, W).permute(0, 2, 1, 3, 4)

    @staticmethod
    def flatten_C(x: Tensor) -> Tensor:
        """Flatten time dim into channel dim.

        Use when channel dim doesn't matter and you don't want to deal with annoying batch dimension problems (e.g. for transforms).

        :param Tensor x: input tensor of shape (B, C, T, H, W)
        :return Tensor: output tensor of shape (B, C*T, H, W)
        """
        return x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])

    @staticmethod
    def wrap_flatten_C(f: Callable[[Tensor], Tensor]) -> Tensor:
        """Flatten time dim into channel dim, apply function, then unwrap.

        The first argument is assumed to be the tensor to be flattened.

        :param Callable f: function to be wrapped
        :return Callable: wrapped function
        """

        def wrapped(x: Tensor, *args, **kwargs):
            """
            :param Tensor x: input tensor of shape (B, C, T, H, W)
            :return Tensor: output tensor of shape (B, C, T, H, W)
            """
            return f(TimeMixin.flatten_C(x), *args, **kwargs).reshape(
                -1, x.shape[1], x.shape[2], x.shape[3], x.shape[4]
            )

        return wrapped

    @staticmethod
    def average(x: Tensor, mask: Tensor = None, dim: int = 2) -> Tensor:
        """Flatten time dim of x by averaging across frames.
        If mask is non-overlapping in time dim, then this will simply be the sum across frames.

        :param Tensor x: input tensor of shape (B,C,T,H,W) (e.g. time-varying k-space)
        :param Tensor mask: mask showing where ``x`` is non-zero. If not provided, then calculated from ``x``.
        :param int dim: time dimension, defaults to 2 (i.e. shape B,C,T,H,W)
        :return Tensor: flattened tensor with time dim removed of shape (B,C,H,W)
        """
        _x = x.sum(dim)
        out = zeros_like(_x)
        m = mask if mask is not None else (x != 0)
        m = m.sum(dim)
        out[m != 0] = _x[m != 0] / m[m != 0]
        return out

    @staticmethod
    def repeat(x: Tensor, target: Tensor, dim: int = 2) -> Tensor:
        """Repeat static image across new time dim T times. Opposite of ``average``.

        :param Tensor x: input tensor of shape (B,C,H,W)
        :param Tensor target: any tensor of desired shape (B,C,T,H,W)
        :param int dim: time dimension, defaults to 2 (i.e. shape B,C,T,H,W)
        :return Tensor: tensor with new time dim of shape B,C,T,H,W
        """
        return x.unsqueeze(dim=dim).expand_as(target)

    def to_static(self) -> Module:
        raise NotImplementedError()
