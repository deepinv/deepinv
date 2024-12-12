from typing import Union, Callable
import torch
from torch import zeros_like, Tensor, Generator
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


class RandomMixin:
    """Base class for torch random number generator functionality for reproducibility.

    Inherit from this class to automatically correctly initialise a torch Generator.
    Note that this works using the ``__new__`` method and so does not need to be explicitly called from the subclass ``__init__``.
    """
    def __new__(cls, *args, **kwargs):
        """
        Automatically perform random initialisation before any constructors called
        """
        instance = super().__new__(cls)
        instance._init_random(*args, **kwargs)
        return instance

    def _init_random(self, device: Union[torch.device, str], rng: Generator = None):
        if rng is None:
            self.rng = Generator(device=device)
        else:
            # Make sure that the random generator is on the same device as the physics generator
            assert rng.device == torch.device(
                device
            ), f"The random generator is not on the same device as the Physics Generator. Got random generator on {rng.device} and the Physics Generator named {self.__class__.__name__} on {self.device}."
            self.rng = rng
        
        self.initial_random_state = self.rng.get_state()
        self._init_random_called = True
    
    def rng_manual_seed(self, seed: int = None):
        r"""
        Sets the seed for the random number generator.

        This is useful for making sure that repeat calls on-the-fly to a method have the same result.

        :param int seed: the seed to set for the random number generator.
         If not provided, the current state of the random number generator is used.
         Note: The `torch.manual_seed` is triggered when a the random number generator is not initialized.
        """
        if seed is not None:
            self.rng = self.rng.manual_seed(seed)

    def reset_rng(self):
        r"""
        Reset the random number generator to its initial state.
        """
        self.rng.set_state(self.initial_random_state)


class MRIMixin:
    r"""
    Mixin base class for MRI functionality.

    Base class that provides helper functions for FFT and mask checking.
    """

    def check_mask(
        self, mask: Tensor = None, three_d: bool = False, device: str = "cpu", **kwargs
    ) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,...,H,W where C=2.

        :param torch.nn.Parameter, torch.Tensor mask: MRI subsampling mask.
        :param bool three_d: If ``False`` the mask should be min 4 dimensions (B, C, H, W) for 2D data, otherwise if ``True`` the mask should have 5 dimensions (B, C, D, H, W) for 3D data.
        :param torch.device, str device: mask intended device.
        """
        if mask is not None:
            mask = mask.to(device)

            while len(mask.shape) < (
                4 if not three_d else 5
            ):  # to B,C,H,W or B,C,D,H,W
                mask = mask.unsqueeze(0)

            if mask.shape[1] == 1:  # make complex if real
                mask = torch.cat([mask, mask], dim=1)

        return mask

    @staticmethod
    def to_torch_complex(x: Tensor):
        """[B,2,...,H,W] real -> [B,...,H,W] complex"""
        return torch.view_as_complex(x.moveaxis(1, -1).contiguous())

    @staticmethod
    def from_torch_complex(x: Tensor):
        """[B,...,H,W] complex -> [B,2,...,H,W] real"""
        return torch.view_as_real(x).moveaxis(-1, 1)

    @staticmethod
    def ifft(x: Tensor, dim=(-2, -1), norm="ortho"):
        """Centered, orthogonal ifft

        :param torch.Tensor x: input kspace of complex dtype of shape [B,...] where ... is all dims to be transformed
        :param tuple dim: fft transform dims, defaults to (-2, -1)
        :param str norm: fft norm, see docs for :meth:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.ifftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)

    @staticmethod
    def fft(x: Tensor, dim=(-2, -1), norm="ortho"):
        """Centered, orthogonal fft

        :param torch.Tensor x: input image of complex dtype of shape [B,...] where ... is all dims to be transformed
        :param tuple dim: fft transform dims, defaults to (-2, -1)
        :param str norm: fft norm, see docs for :meth:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.fftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)