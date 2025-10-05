from typing import Callable
import numpy as np
import torch
from torch import Tensor, zeros_like
from torch.nn import Module
from torchvision.transforms import CenterCrop, Resize


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

        :param torch.Tensor mask: MRI subsampling mask.
        :param bool three_d: If ``False`` the mask should be min 4 dimensions (B, C, H, W) for 2D data, otherwise if ``True`` the mask should have 5 dimensions (B, C, D, H, W) for 3D data.
        :param torch.device, str device: mask intended device.
        """
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

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
        :param str norm: fft norm, see docs for :func:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.ifftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)

    @staticmethod
    def fft(x: Tensor, dim=(-2, -1), norm="ortho"):
        """Centered, orthogonal fft

        :param torch.Tensor x: input image of complex dtype of shape [B,...] where ... is all dims to be transformed
        :param tuple dim: fft transform dims, defaults to (-2, -1)
        :param str norm: fft norm, see docs for :func:`torch.fft.fftn`, defaults to "ortho"
        """
        x = torch.fft.ifftshift(x, dim=dim)
        x = torch.fft.fftn(x, dim=dim, norm=norm)
        return torch.fft.fftshift(x, dim=dim)

    def im_to_kspace(self, x: Tensor, three_d: bool = False) -> Tensor:
        """Convenience method that wraps fft.

        :param torch.Tensor x: input image of shape (B,2,...) of real dtype
        :param bool three_d: whether MRI data is 3D or not, defaults to False
        :return: Tensor: output measurements of shape (B,2,...) of real dtype
        """
        return self.from_torch_complex(
            self.fft(
                self.to_torch_complex(x), dim=(-3, -2, -1) if three_d else (-2, -1)
            )
        )

    def kspace_to_im(self, y: Tensor, three_d: bool = False) -> Tensor:
        """Convenience method that wraps inverse fft.

        :param torch.Tensor y: input measurements of shape (B,2,...) of real dtype
        :param bool three_d: whether MRI data is 3D or not, defaults to False
        :return: Tensor: output image of shape (B,2,...) of real dtype
        """
        return self.from_torch_complex(
            self.ifft(
                self.to_torch_complex(y), dim=(-3, -2, -1) if three_d else (-2, -1)
            )
        )

    def crop(
        self,
        x: Tensor,
        crop: bool = True,
        shape: tuple[int, int] = None,
        rescale: bool = False,
    ) -> Tensor:
        """Center crop 2D image according to ``img_size``.

        This matches the RSS reconstructions of the original raw data in :class:`deepinv.datasets.FastMRISliceDataset`.

        If ``img_size`` has odd height, then adjust by one pixel to match FastMRI data.

        :param torch.Tensor x: input tensor of shape (...,H,W)
        :param bool crop: whether to perform crop, defaults to `True`. If `True`, `rescale` must be `False`.
        :param tuple[int, int] shape: optional shape (..., H,W) to crop to. If `None`, crops to `img_size` attribute.
        :param bool rescale: whether to rescale instead of cropping. If `True`, `crop` must be `False`.
            Note to be careful here as resizing will change aspect ratio.
        """
        crop_size = shape[-2:] if shape is not None else self.img_size[-2:]
        odd_h = crop_size[0] % 2 == 1

        if odd_h:
            crop_size = (crop_size[0] + 1, crop_size[1])

        if rescale and crop:
            raise ValueError("Only one of rescale or crop can be used.")
        elif rescale:
            cropped = Resize(crop_size)(x.reshape(-1, *x.shape[-2:])).reshape(
                *x.shape[:-2], *crop_size
            )
        elif crop:
            cropped = CenterCrop(crop_size)(x)
        else:
            return x

        if odd_h:
            cropped = cropped[..., :-1, :]

        return cropped

    @staticmethod
    def rss(
        x: Tensor, multicoil: bool = True, mag: bool = True, three_d: bool = False
    ) -> Tensor:
        r"""Perform root-sum-square reconstruction on multicoil data, defined as

        .. math::

                \operatorname{RSS}(x) = \sqrt{\sum_{n=1}^N |x_n|^2}

        where :math:`x_n` are the coil images of :math:`x`, :math:`|\cdot|` denotes the magnitude
        and :math:`N` is the number of coils. Note that the sum is performed voxel-wise.

        :param torch.Tensor x: input image of shape (B,2,...) where 2 represents
            real and imaginary channels
        :param bool multicoil: if ``True``, assume ``x`` is of shape (B,2,N,...),
            and reduce over coil dimension N too.
        :param bool mag: if `False`, do not reduce over the complex dimension. Rarely used.
        :param bool three_d: used only for validating input shape, set to `True` if input is 3D data.
        """
        assert (
            x.shape[1] == 2 and not x.is_complex()
        ), "x should be of shape (B,2,...) and not of complex dtype."

        mc_dim = 1 if multicoil else 0
        th_dim = 1 if three_d else 0
        assert (
            len(x.shape) == 4 + mc_dim + th_dim
        ), "x should be of shape (B,2,...) for singlecoil data or (B,2,N,...) for multicoil data."

        ss = x.pow(2)

        if mag:
            ss = ss.sum(dim=1, keepdim=True)

        if multicoil:
            ss = ss.sum(dim=2)

        return ss.sqrt()
