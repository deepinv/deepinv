from __future__ import annotations

from typing import Callable
import numpy as np
import torch
from torch import Tensor, zeros_like
from torch.nn import Module
from torchvision.transforms import CenterCrop, Resize
from deepinv.utils.decorators import _deprecated_argument
from ._internal import _as_pair, _add_tuple
from ._tiling import (
    _compute_compatible_img_size,
    _compute_needed_pad,
    _compute_num_patches,
    _image_to_patches_impl,
    _patches_to_image_impl,
)


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

    @staticmethod
    @_deprecated_argument("device")
    def check_mask(mask: Tensor = None, three_d: bool = False) -> None:
        r"""
        Updates MRI mask and verifies mask shape to be B,C,...,H,W where C=2.

        :param torch.Tensor mask: MRI subsampling mask.
        :param bool three_d: If ``False`` the mask should be min 4 dimensions (B, C, H, W) for 2D data, otherwise if ``True`` the mask should have 5 dimensions (B, C, D, H, W) for 3D data.
        """
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)

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


class TiledMixin2d:
    r"""
    Mixin base class for 2D tiled patch extraction and reconstruction.
    Provides methods to extract overlapping patches from images and reconstruct images from patches.

    It also handles padding if necessary to ensure all patches have the same size.
    The patch extraction and reconstruction are implemented using PyTorch's unfold and fold operations for efficiency.

    :param int | tuple[int, int] patch_size: Size of each patch (height, width) or single `int` for square patches.
    :param int | tuple[int, int] stride: Stride between adjacent patches (height, width). If a single `int` is provided, it is used for both dimensions. Defaults to half the patch size.
    :param bool pad_if_needed: If `True`, the image will be padded if necessary to ensure all patches have the same size. Defaults to `True`.

    |sep|

    The following example demonstrates how to use the `TiledMixin2d` to extract patches from an image and reconstruct the image from those patches.

    :Examples:

        >>> import torch
        >>> from deepinv.utils.mixins import TiledMixin2d
        >>> # Create an image of shape (B, C, H, W)
        >>> B, C, H, W = 1, 1, 5, 5
        >>> image = torch.arange(B * C * H * W, dtype=torch.float32).reshape(B, C, H, W)
        >>> print(image)
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])

        >>> # Initialize the TiledMixin2d with patch size and stride
        >>> patch_size = (3, 3)
        >>> stride = (2, 2)
        >>> tiled_mixin = TiledMixin2d(patch_size=patch_size, stride=stride)

        >>> # Extract patches from the image
        >>> patches = tiled_mixin.image_to_patches(image)
        >>> print("Extracted Patches Shape:", patches.shape)
        Extracted Patches Shape: torch.Size([1, 1, 2, 2, 3, 3])
        >>> print(patches[..., 0, 0, :, :]) # Print the first patch for verification
        tensor([[[[ 0.,  1.,  2.],
                  [ 5.,  6.,  7.],
                  [10., 11., 12.]]]])

        >>> # Reconstruct the image from the patches
        >>> reconstructed_image = tiled_mixin.patches_to_image(patches, img_size=(H, W))
        >>> print("Reconstructed Image Shape:", reconstructed_image.shape)
        Reconstructed Image Shape: torch.Size([1, 1, 5, 5])
        >>> print(reconstructed_image)
        tensor([[[[ 0.,  1.,  4.,  3.,  4.],
                  [ 5.,  6., 14.,  8.,  9.],
                  [20., 22., 48., 26., 28.],
                  [15., 16., 34., 18., 19.],
                  [20., 21., 44., 23., 24.]]]])
        >>> # Note that by default, the reconstructed image is not necessarily equal to the original image due to overlapping regions being summed. Setting `reduce_overlap="mean"` in `patches_to_image` will average the overlapping regions instead of summing, which can give a closer reconstruction to the original image.
        >>> reconstructed_image_mean = tiled_mixin.patches_to_image(patches, img_size=(H, W), reduce_overlap="mean")
        >>> print(reconstructed_image_mean)
        tensor([[[[ 0.,  1.,  2.,  3.,  4.],
                  [ 5.,  6.,  7.,  8.,  9.],
                  [10., 11., 12., 13., 14.],
                  [15., 16., 17., 18., 19.],
                  [20., 21., 22., 23., 24.]]]])
    """

    def __init__(
        self,
        patch_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = None,
        pad_if_needed: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_size = _as_pair(patch_size)
        self.stride = (
            _as_pair(stride)
            if stride is not None
            else tuple(p // 2 for p in self.patch_size)
        )

        if self.stride[0] > self.patch_size[0] or self.stride[1] > self.patch_size[1]:
            raise ValueError(
                f"Stride {self.stride} must be smaller or equal than patch_size {self.patch_size}."
            )

        self.pad_if_needed = pad_if_needed

    def image_to_patches(
        self, image: Tensor, pad: int | tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> Tensor:
        r"""
        Split an image into overlapping patches.

        The image will be padded if necessary to ensure all patches have the same size.

        :param torch.Tensor image: Input image tensor of shape `(B, C, H, W)`.
        :param int | tuple[int, int, int, int] pad: Optional, if provided, the patch size will be increased by this padding on each side. Can be a single int for symmetric padding or a tuple of 4 ints for (left, right, top, bottom) padding. Defaults to `(0, 0, 0, 0)` for no additional padding.
        :return: Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
        """
        if isinstance(pad, int):
            pad = (pad, pad, pad, pad)
        elif not isinstance(pad, (list, tuple)) and len(pad) != 4:
            raise ValueError(
                f"Invalid pad argument: {pad}. Must be int or tuple of 4 ints."
            )

        patch_size = _add_tuple(self.patch_size, (pad[2] + pad[3], pad[0] + pad[1]))
        return _image_to_patches_impl(
            image=image,
            patch_size=patch_size,
            stride=self.stride,
            pad_if_needed=self.pad_if_needed,
        )

    def patches_to_image(
        self,
        patches: Tensor,
        img_size: tuple[int, int] | None = None,
        reduce_overlap: str = "sum",
    ) -> Tensor:
        r"""
        Reconstruct an image from overlapping patches.

        This is the inverse operation of `image_to_patches`. Note that overlapping
        regions are summed. So the reconstructed image is not necessarily equal to the original image.

        :param torch.Tensor patches: Patches tensor of shape `(B, C, n_rows, n_cols, patch_h, patch_w)`.
        :param img_size: Target output size (height, width). If provided, output is cropped to this size from the top-left corner.
        :param reduce_overlap: How to handle overlapping regions. Options are `"sum"` or `"mean"`.

        :return: Reconstructed image tensor of shape `(B, C, H, W)`.
        """
        return _patches_to_image_impl(
            patches=patches,
            stride=self.stride,
            img_size=img_size,
            reduce_overlap=reduce_overlap,
        )

    def get_needed_pad(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """
        Get required padding.

        :param img_size: Original image size (height, width).
        :return: Tuple of (compatible_size, padding).
        """
        return _compute_needed_pad(img_size, self.patch_size, self.stride)

    def get_compatible_img_size(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """
        Get compatible image size for patch extraction.

        :param img_size: Original image size (height, width).
        :return: Compatible image size (height, width).
        """
        return _compute_compatible_img_size(img_size, self.patch_size, self.stride)

    def get_num_patches(self, img_size: tuple[int, int]) -> tuple[int, int]:
        """
        Get number of patches along height and width.
            - If `pad_if_needed` is `True`, this will return the number of patches that can be extracted after padding the image to a compatible size.
            - If `pad_if_needed` is `False`, this will return the number of patches that can be extracted without padding, which may not cover the whole image.

        :param img_size: Image size (height, width).
        :return: Number of patches (n_h, n_w).
        """
        return _compute_num_patches(
            img_size=img_size,
            patch_size=self.patch_size,
            stride=self.stride,
            pad_if_needed=self.pad_if_needed,
        )
