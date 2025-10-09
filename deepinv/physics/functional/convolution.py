import torch
import torch.nn.functional as F
from torch import Tensor
import torch.fft as fft
import warnings
from itertools import chain
from deepinv.utils.decorators import _deprecated_func_replaced_by

_warned_messages = set()


def _warn_once_padding(pad_value: list[int], padding: str, category=UserWarning):
    """Emit a warning only once per unique message."""
    unit_kernel = "x".join(["1"] * len(pad_value))
    message = (
        f"You're using padding = '{padding}' with a {unit_kernel} kernel. This is equivalent to no padding. "
        f"Consider using padding = 'valid' instead."
    )
    if message not in _warned_messages:
        warnings.warn(message, category)
        _warned_messages.add(message)


def _not_implemented_padding_messages(padding):
    return f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."


def conv2d(
    x: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function performing the 2d convolution of images ``x`` and ``filter``.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose2d`

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` where ``b`` can be either ``1`` or ``B``
        and ``c`` can be either ``1`` or ``C``.
        Filter center is at ``(hh, ww)`` where ``hh = h//2`` if h is odd and
        ``hh = h//2 - 1`` if h is even. Same for ``ww``.
    :param bool correlation: choose True if you want a cross-correlation (default `False`)

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param padding: (options = ``valid``, ``circular``, ``replicate``, ``reflect``, ``constant``) If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.
        ``constant`` corresponds to zero padding or ``same`` in :func:`torch.nn.functional.conv2d`
    :return: :class:`torch.Tensor`: the blurry output.

    .. note::

        Contrarily to Pytorch :func:`torch.nn.functional.conv2d`, which performs a cross-correlation, this function performs a convolution.

        This function gives the same result as :func:`deepinv.physics.functional.conv2d_fft`. However, for small kernels, this function is faster.
        For large kernels, :func:`deepinv.physics.functional.conv2d_fft` is usually faster but requires more memory.

    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    filter = _flip_filter_if_needed(filter, correlation, dims=(-2, -1))

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    # Ensure fast memory layout before heavy ops.
    x = x.contiguous()

    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1)

    if padding != "valid":
        ph = h // 2
        ih = (h - 1) % 2
        pw = w // 2
        iw = (w - 1) % 2
        if ph == 0 and pw == 0:
            _warn_once_padding([ph, pw], padding)

        pad = (pw, pw - iw, ph, ph - ih)  # because functional.pad is w,h instead of h,w

        x = F.pad(x, pad, mode=padding, value=0)
        B, C, H, W = x.size()

    # Move batch dim of the input into channels
    x = x.reshape(1, -1, H, W)
    # Expand the channel dim of the filter and move it into batch dimension
    filter = filter.reshape(B * C, -1, h, w)
    # Perform the convolution, using the groups parameter
    output = F.conv2d(x, filter, padding="valid", groups=B * C)
    # Make it in the good shape
    output = output.view(B, C, output.size(-2), -1)

    return output.contiguous()


def conv_transpose2d(
    y: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of x and filter. The transposed of this operation is :func:`deepinv.physics.functional.conv2d`

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` ) where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool correlation: choose True if you want a cross-correlation (default `False`)

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the output is larger than the image (padding)
        otherwise the output has the same size as the image.

    :return: :class:`torch.Tensor` : the output

    .. note::

        This functions gives the same result as :func:`deepinv.physics.functional.conv_transpose2d_fft`. However, for small kernels, this function is faster.
        For large kernels, :func:`deepinv.physics.functional.conv_transpose2d_fft` is usually faster but requires more memory.

    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    filter = _flip_filter_if_needed(filter, correlation, dims=(-2, -1))

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    ph = h // 2
    pw = w // 2
    ih = (h - 1) % 2
    iw = (w - 1) % 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1)

    # Move batch dim of the input into channels
    y = y.contiguous()
    y = y.reshape(1, -1, H, W)
    # Expand the channel dim of the filter and move it into batch dimension
    filter = filter.reshape(B * C, -1, h, w)
    # Perform the convolution, using the groups parameter
    x = F.conv_transpose2d(y, filter, groups=B * C)
    # Make it in the good shape
    x = x.view(B, C, x.size(-2), -1)

    if padding == "valid":
        out = x
    elif padding == "circular":
        out = x[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

        # sides and corners
        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                if sy == 0 and sx == 0:
                    continue
                ty, ysrc = _tgt_src_for_axis_circular(sy, ph, ih)
                tx, xsrc = _tgt_src_for_axis_circular(sx, pw, iw)
                out[:, :, ty, tx].add_(x[:, :, ysrc, xsrc])

    elif padding == "reflect":
        out = x[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                if sy == 0 and sx == 0:
                    continue
                ty, ysrc = _tgt_src_for_axis_reflect(sy, ph, ih)
                tx, xsrc = _tgt_src_for_axis_reflect(sx, pw, iw)
                flip_dims = tuple(dim for s, dim in zip((sy, sx), (2, 3)) if s != 0)
                chunk = x[:, :, ysrc, xsrc]
                if flip_dims:
                    chunk = chunk.flip(dims=flip_dims)
                out[:, :, ty, tx].add_(chunk)

    elif padding == "replicate":
        out = x[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                if sy == 0 and sx == 0:
                    continue
                ty, ysrc, yred = _tgt_src_for_axis_replicate(sy, ph, ih)
                tx, xsrc, xred = _tgt_src_for_axis_replicate(sx, pw, iw)
                reduce_dims = tuple(
                    dim for red, dim in zip((yred, xred), (2, 3)) if red
                )
                chunk = x[:, :, ysrc, xsrc]
                if reduce_dims:
                    chunk = chunk.sum(dim=reduce_dims)
                out[:, :, ty, tx].add_(chunk)

    elif padding == "constant":
        out = x[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]
    else:
        raise ValueError(_not_implemented_padding_messages(padding))

    return out.contiguous()


def conv2d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 2d convolution of images ``x`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose2d_fft`

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as
    `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    .. note::

        The convolution here is a convolution, not a correlation as in :func:`torch.nn.functional.conv2d`.
        This function gives the same result as :func:`deepinv.physics.functional.conv2d` and is faster for large kernels.

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation.
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding),
        otherwise the output has the same size as the image. Default is ``'valid'``.
    :return: :class:`torch.Tensor`: the output of the convolution of the shape size as `x`.
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"

    ph, pw = h // 2, w // 2
    ih, iw = (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    def fft2(t, s=None):
        return fft.rfft2(t, s=s) if real_fft else fft.fft2(t, s=s)

    def ifft2(t, s):
        return fft.irfft2(t, s=s).real if real_fft else fft.ifft2(t, s=s).real

    x = x.contiguous()
    if padding == "circular":
        # Circular convolution with kernel center aligned via filter centering
        img_size = (H, W)
        fx = fft2(x, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-2, -1))
        out = ifft2(fx * ff, s=img_size)

    elif padding == "valid":
        # Full linear convolution then crop to valid window
        sH, sW = H + h - 1, W + w - 1
        img_size = (sH, sW)
        fx = fft2(x, s=img_size)
        ff = fft2(filter, s=img_size)
        full = ifft2(fx * ff, s=img_size)
        out = full[:, :, h - 1 : H, w - 1 : W]

    elif padding in ("constant", "reflect", "replicate"):
        # Linear convolution on a padded grid via circular FFT-conv on that grid.
        pad = (pw, pw - iw, ph, ph - ih)  # (W_left, W_right, H_top, H_bottom)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        img_size = x_pad.shape[-2:]
        fx = fft2(x_pad, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-2, -1))
        y_pad = ifft2(fx * ff, s=img_size)

        # Extract central region back to original size
        out = y_pad[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

    else:
        raise ValueError(_not_implemented_padding_messages(padding))

    return out.contiguous()


def conv_transpose2d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of ``x`` and ``filter`` using FFT.
    The adjoint of this operation is :func:`deepinv.physics.functional.conv2d_fft`.

    :param torch.Tensor y: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` ) where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation.
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``. If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image. Default is ``'valid'``.

    :return: :class:`torch.Tensor`: the output of the convolution, which has the same shape as :math:`y`.

    .. note::
        This functions gives the same result as :func:`deepinv.physics.functional.conv_transpose2d`.
        However, for large kernels, this function is faster but requires more memory.
        For small kernels, consider using :func:`deepinv.physics.functional.conv_transpose2d`.
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    if c != C:
        assert c == 1

    if b != B:
        assert b == 1

    ph, pw = h // 2, w // 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    ih, iw = (h - 1) % 2, (w - 1) % 2

    def fft2(t, s=None):
        return fft.rfft2(t, s=s) if real_fft else fft.fft2(t, s=s)

    def ifft2(t, s):
        return fft.irfft2(t, s=s).real if real_fft else fft.ifft2(t, s=s).real

    if padding == "circular":
        # Circular adjoint: multiply by conj of centered filter FFT, no roll.
        img_size = (H, W)
        fy = fft2(y, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-2, -1))
        out = ifft2(fy * torch.conj(ff), s=img_size)

    elif padding == "valid":
        # Adjoint of full-conv + center crop
        sH, sW = H + h - 1, W + w - 1
        img_size = (sH, sW)
        y_full = F.pad(y, (w - 1, w - 1, h - 1, h - 1), mode="constant", value=0)
        fy = fft2(y_full, s=img_size)
        ff = fft2(filter, s=img_size)
        out = ifft2(fy * torch.conj(ff), s=img_size)

    elif padding in ("constant", "reflect", "replicate"):
        # Forward: pad (P) -> conv (C) -> crop (S)
        # Adjoint:  S* -> C* -> P*
        Hp = H + ph + (ph - ih)
        Wp = W + pw + (pw - iw)
        img_size = (Hp, Wp)

        # S*: embed y into center of padded grid
        y_big = F.pad(y, (pw, pw - iw, ph, ph - ih), mode="constant", value=0)
        # C*: circular transpose conv on padded grid using centered filter
        fy = fft2(y_big, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-2, -1))
        z_big = ifft2(fy * torch.conj(ff), s=img_size)

        # P*: adjoint of padding -> fold to original H x W
        if padding == "constant":
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ]
        elif padding == "reflect":
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ].clone()
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    if sy == 0 and sx == 0:
                        continue
                    ty, ysrc = _tgt_src_for_axis_reflect(sy, ph, ih)
                    tx, xsrc = _tgt_src_for_axis_reflect(sx, pw, iw)
                    flip_dims = tuple(dim for s, dim in zip((sy, sx), (2, 3)) if s != 0)
                    chunk = z_big[:, :, ysrc, xsrc]
                    if flip_dims:
                        chunk = chunk.flip(dims=flip_dims)
                    out[:, :, ty, tx].add_(chunk)
        else:  # replicate
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ].clone()
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    if sy == 0 and sx == 0:
                        continue
                    ty, ysrc, yred = _tgt_src_for_axis_replicate(sy, ph, ih)
                    tx, xsrc, xred = _tgt_src_for_axis_replicate(sx, pw, iw)
                    reduce_dims = tuple(
                        dim for red, dim in zip((yred, xred), (2, 3)) if red
                    )
                    chunk = z_big[:, :, ysrc, xsrc]
                    if reduce_dims:
                        chunk = chunk.sum(dim=reduce_dims)
                    out[:, :, ty, tx].add_(chunk)
    else:
        raise ValueError(_not_implemented_padding_messages(padding))

    return out.contiguous()


def conv3d(
    x: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function to perform 3D convolution of images `x` and `filter`.

    The transposed of this operation is :func:`deepinv.physics.functional.conv_transpose3d`.

    :param torch.Tensor x: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param str padding: can be ``'valid'`` (default), ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``. If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.

    :return: :class:`torch.Tensor`: the output of the convolution, which has the shape ``(B, C, D-d+1, W-w+1, H-h+1)`` if ``padding = 'valid'`` and the same shape as ``x`` otherwise.
    """
    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    B, C, D, H, W = x.shape
    b, c, d, h, w = filter.shape

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1, -1)
    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1, -1)

    # Flip the kernel for true convolution
    filter = _flip_filter_if_needed(filter, correlation, dims=(-3, -2, -1))
    x = x.contiguous()
    # Determine padding
    if padding.lower() != "valid":
        # Calculate padding to keep output same size as input
        pd = d // 2
        ph = h // 2
        pw = w // 2
        ih = (h - 1) % 2
        iw = (w - 1) % 2
        id = (d - 1) % 2
        pad = (
            pw,
            pw - iw,
            ph,
            ph - ih,
            pd,
            pd - id,
        )  # F.pad expects (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x = F.pad(x, pad, mode=padding, value=0)

        if pd == 0 and pw == 0 and ph == 0:
            _warn_once_padding([ph, pw, pd], padding)

        B, C, D, H, W = x.shape

    # Grouped convolution trick for per-batch filters and channels
    x = x.reshape(1, B * C, D, H, W)
    filter = filter.reshape(B * C, -1, d, h, w)
    out = F.conv3d(x, filter, padding="valid", groups=B * C)
    # Make it in the good shape
    out = out.reshape(B, C, *out.shape[-3:])
    return out.contiguous()


def conv_transpose3d(
    y: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function to perform 3D transpose convolution.
    The transposed of this operation is :func:`deepinv.physics.functional.conv3d`.

    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param str padding: can be ``'valid'`` (default), ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``. If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image.
    :param bool correlation: choose True if you want a cross-correlation (default `False`).

    :return: :class:`torch.Tensor`: the output of the convolution, which has the shape ``(B, C, D+d-1, W+w-1, H+h-1)`` if ``padding = 'valid'`` and the same shape as ``y`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    B, C, D, H, W = y.shape
    b, c, d, h, w = filter.shape

    pd = d // 2
    ph = h // 2
    pw = w // 2
    id = (d - 1) % 2
    ih = (h - 1) % 2
    iw = (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)
    # Flip the kernel for true convolution
    filter = _flip_filter_if_needed(filter, correlation, dims=(-3, -2, -1))

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1, -1)
    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1, -1)

    # Use grouped convolution trick for per-batch filters and channels
    y = y.contiguous()
    y = y.reshape(1, B * C, D, H, W)
    filter = filter.reshape(B * C, 1, d, h, w)

    x = F.conv_transpose3d(y, filter, groups=B * C)
    x = x.reshape(B, C, *x.shape[-3:])

    if padding == "valid":
        out = x
    elif padding == "circular":
        # Start from the central crop
        out = x[
            :,
            :,
            _center_crop_slice_1d(pd, id),
            _center_crop_slice_1d(ph, ih),
            _center_crop_slice_1d(pw, iw),
        ]
        # Triple loop over shifts for (z, y, x); skip the (0,0,0) case
        for sz in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    if sz == 0 and sy == 0 and sx == 0:
                        continue
                    tz, sz_src = _tgt_src_for_axis_circular(sz, pd, id)
                    ty, sy_src = _tgt_src_for_axis_circular(sy, ph, ih)
                    tx, sx_src = _tgt_src_for_axis_circular(sx, pw, iw)
                    out[:, :, tz, ty, tx].add_(x[:, :, sz_src, sy_src, sx_src])
    elif padding == "constant":
        out = x[
            :,
            :,
            _center_crop_slice_1d(pd, id),
            _center_crop_slice_1d(ph, ih),
            _center_crop_slice_1d(pw, iw),
        ]
    elif padding == "reflect":
        # Center crop
        out = x[
            :,
            :,
            _center_crop_slice_1d(pd, id),
            _center_crop_slice_1d(ph, ih),
            _center_crop_slice_1d(pw, iw),
        ]
        for sz in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    if sz == 0 and sy == 0 and sx == 0:
                        continue
                    tz, zsrc = _tgt_src_for_axis_reflect(sz, pd, id)
                    ty, ysrc = _tgt_src_for_axis_reflect(sy, ph, ih)
                    tx, xsrc = _tgt_src_for_axis_reflect(sx, pw, iw)
                    flip_dims = tuple(
                        dim for s, dim in zip((sz, sy, sx), (2, 3, 4)) if s != 0
                    )
                    chunk = x[:, :, zsrc, ysrc, xsrc]
                    if flip_dims:
                        chunk = chunk.flip(dims=flip_dims)
                    out[:, :, tz, ty, tx].add_(chunk)
    elif padding == "replicate":
        out = x[
            :,
            :,
            _center_crop_slice_1d(pd, id),
            _center_crop_slice_1d(ph, ih),
            _center_crop_slice_1d(pw, iw),
        ]
        for sz in (-1, 0, 1):
            for sy in (-1, 0, 1):
                for sx in (-1, 0, 1):
                    if sz == 0 and sy == 0 and sx == 0:
                        continue
                    tz, zsrc, zred = _tgt_src_for_axis_replicate(sz, pd, id)
                    ty, ysrc, yred = _tgt_src_for_axis_replicate(sy, ph, ih)
                    tx, xsrc, xred = _tgt_src_for_axis_replicate(sx, pw, iw)
                    reduce_dims = tuple(
                        dim for red, dim in zip((zred, yred, xred), (2, 3, 4)) if red
                    )
                    chunk = x[:, :, zsrc, ysrc, xsrc]
                    if reduce_dims:
                        chunk = chunk.sum(dim=reduce_dims)
                    out[:, :, tz, ty, tx].add_(chunk)
    else:
        raise ValueError(_not_implemented_padding_messages(padding))
    return out.contiguous()


def conv3d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d convolution of ``x`` and `filter` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose3d_fft`.

    If ``b = 1`` or ``c = 1``, this function applies the same filter for each channel and each image.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image. Default is ``'valid'``.

    .. note::

        The filter center is located at ``(d//2, h//2, w//2)``.

        This function and :func:`deepinv.physics.functional.conv3d` are equivalent. However, this function is more efficient for large filters but requires more memory.

    :return: :class:`torch.Tensor`: the output of the convolution, which has the same shape as :math:`x` if ``padding = 'circular'``, ``(B, C, D-d+1, W-w+1, H-h+1)`` otherwise.
    """

    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    B, C, D, H, W = x.size()
    b, c, d, h, w = filter.size()

    filter = filter.contiguous()
    x = x.contiguous()
    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1, -1)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)

    def fft3(t, s=None):
        return (
            fft.rfftn(t, s=s, dim=(-3, -2, -1))
            if real_fft
            else fft.fftn(t, s=s, dim=(-3, -2, -1))
        )

    def ifft3(t, s=None):
        return (
            fft.irfftn(t, s=s, dim=(-3, -2, -1)).real
            if real_fft
            else fft.ifftn(t, s=s, dim=(-3, -2, -1)).real
        )

    if padding == "circular":
        # Circular convolution with kernel center aligned via filter centering
        img_size = (D, H, W)
        fx = fft3(x, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-3, -2, -1))
        out = ifft3(fx * ff, s=img_size)

    elif padding == "valid":
        # Full linear convolution then crop to valid window
        sD, sH, sW = D + d - 1, H + h - 1, W + w - 1
        img_size = (sD, sH, sW)
        fx = fft3(x, s=img_size)
        ff = fft3(filter, s=img_size)
        full = ifft3(fx * ff, s=img_size)
        out = full[:, :, d - 1 : D, h - 1 : H, w - 1 : W]

    elif padding in ("constant", "reflect", "replicate"):
        # Linear convolution on a padded grid via circular FFT-conv on that grid.
        pad = (
            pw,
            pw - iw,
            ph,
            ph - ih,
            pd,
            pd - id,
        )  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        img_size = x_pad.shape[-3:]

        fx = fft3(x_pad, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-3, -2, -1))
        y_pad = ifft3(fx * ff, s=img_size)

        # Extract central region back to original size
        out = y_pad[
            :,
            :,
            _center_crop_slice_1d(pd, id),
            _center_crop_slice_1d(ph, ih),
            _center_crop_slice_1d(pw, iw),
        ]

    else:
        raise ValueError(_not_implemented_padding_messages(padding))

    return out.contiguous()


def conv_transpose3d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d transposed convolution of ``y`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv3d_fft`.

    If ``b = 1`` or ``c = 1``, then this function applies the same filter for each channel.
    Otherwise, each channel of each image is convolved with the corresponding kernel.


    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image. Default is ``'valid'``.

    .. note::

        This function and :func:`deepinv.physics.functional.conv_transpose3d` are equivalent. However, this function is more efficient for large filters.

    :return: :class:`torch.Tensor`: the output of the transposed convolution, which has the same shape as ``y`` if ``padding = 'circular'``, ``(B, C, D+d-1, W+w-1, H+h-1)`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    # Get dimensions of the input and the filter
    B, C, D, H, W = y.size()
    b, c, d, h, w = filter.size()

    filter = filter.contiguous()
    y = y.contiguous()
    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        filter = filter.expand(B, -1, -1, -1, -1)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)

    def fft3(t, s=None):
        return (
            fft.rfftn(t, s=s, dim=(-3, -2, -1))
            if real_fft
            else fft.fftn(t, s=s, dim=(-3, -2, -1))
        )

    def ifft3(t, s):
        return (
            fft.irfftn(t, s=s, dim=(-3, -2, -1)).real
            if real_fft
            else fft.ifftn(t, s=s, dim=(-3, -2, -1)).real
        )

    if padding == "circular":
        img_size = (D, H, W)
        fy = fft3(y, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-3, -2, -1))
        out = ifft3(fy * torch.conj(ff), s=img_size)

    elif padding == "valid":
        sD, sH, sW = D + d - 1, H + h - 1, W + w - 1
        img_size = (sD, sH, sW)
        y_full = F.pad(
            y, (w - 1, w - 1, h - 1, h - 1, d - 1, d - 1), mode="constant", value=0
        )
        fy = fft3(y_full, s=img_size)
        ff = fft3(filter, s=img_size)
        out = ifft3(fy * torch.conj(ff), s=img_size)

    elif padding in ("constant", "reflect", "replicate"):
        # Forward: pad (P) -> conv (C) -> crop (S)
        # Adjoint:  S* -> R* -> C* -> P*
        Dp = D + pd + (pd - id)
        Hp = H + ph + (ph - ih)
        Wp = W + pw + (pw - iw)
        img_size = (Dp, Hp, Wp)

        # S*: embed y into center of padded grid
        y_big = F.pad(
            y, (pw, pw - iw, ph, ph - ih, pd, pd - id), mode="constant", value=0
        )

        # C*: circular transpose conv on padded grid
        fy = fft3(y_big, s=img_size)
        ff = filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-3, -2, -1))
        z_big = ifft3(fy * torch.conj(ff), s=img_size)

        # P*: adjoint of padding -> fold to original D x H x W
        if padding == "constant":
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(pd, id),
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ]
        elif padding == "reflect":
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(pd, id),
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ].clone()
            for sz in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sx in (-1, 0, 1):
                        if sz == 0 and sy == 0 and sx == 0:
                            continue
                        tz, zsrc = _tgt_src_for_axis_reflect(sz, pd, id)
                        ty, ysrc = _tgt_src_for_axis_reflect(sy, ph, ih)
                        tx, xsrc = _tgt_src_for_axis_reflect(sx, pw, iw)
                        flip_dims = tuple(
                            dim for s, dim in zip((sz, sy, sx), (2, 3, 4)) if s != 0
                        )
                        chunk = z_big[:, :, zsrc, ysrc, xsrc]
                        if flip_dims:
                            chunk = chunk.flip(dims=flip_dims)
                        out[:, :, tz, ty, tx].add_(chunk)
        else:  # replicate
            out = z_big[
                :,
                :,
                _center_crop_slice_1d(pd, id),
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ].clone()
            for sz in (-1, 0, 1):
                for sy in (-1, 0, 1):
                    for sx in (-1, 0, 1):
                        if sz == 0 and sy == 0 and sx == 0:
                            continue
                        tz, zsrc, zred = _tgt_src_for_axis_replicate(sz, pd, id)
                        ty, ysrc, yred = _tgt_src_for_axis_replicate(sy, ph, ih)
                        tx, xsrc, xred = _tgt_src_for_axis_replicate(sx, pw, iw)
                        reduce_dims = tuple(
                            dim
                            for red, dim in zip((zred, yred, xred), (2, 3, 4))
                            if red
                        )
                        chunk = z_big[:, :, zsrc, ysrc, xsrc]
                        if reduce_dims:
                            chunk = chunk.sum(dim=reduce_dims)
                        out[:, :, tz, ty, tx].add_(chunk)
    else:
        raise ValueError(_not_implemented_padding_messages(padding))

    return out.contiguous()


# Some helper functions for computing the slice indices for padding modes in convolution operations
# Map the shift {-1,0,+1} to (target slice on out, source slice on x) per axis
# --------------------------------------------------------------------------------------------------
def _tgt_src_for_axis_circular(s, p, i):
    # target slice on out,  source slice on x
    if s == 0:
        return slice(None), slice(p, -p + i)
    elif s == -1:
        return slice(0, p - i), slice(-p + i, None)
    else:
        return slice(-p, None), slice(None, p)


def _tgt_src_for_axis_reflect(s, p, i):
    # target slice on out, source slice on x (reflect requires flipping along axes with s != 0)
    if s == 0:
        return slice(None), slice(p, -p + i)
    elif s == -1:
        return slice(1, 1 + p), slice(0, p)
    else:
        return slice(-p + i - 1, -1), slice(-p + i, None)


def _tgt_src_for_axis_replicate(s, p, i):
    # target index/slice on out, source slice on x, and whether to reduce (sum) over this axis
    if s == 0:
        return slice(None), slice(p, -p + i), False
    elif s == -1:
        return 0, slice(0, p), True
    else:
        return -1, slice(-p + i, None), True


def _center_crop_slice_1d(p: int, i: int) -> slice:
    # When p == i == 0, return the whole axis; otherwise replicate p : -p + i
    # This prevent issues with 0:-0 slices
    return slice(None) if (p == 0 and i == 0) else slice(p, -p + i)


def filter_fft(
    filter: Tensor,
    img_size: tuple[int, ...],
    real_fft: bool = True,
    dims: tuple[int, ...] = (-1, -2),
) -> Tensor:
    r"""
    A helper function to compute the centered FFT of filter zero-padded to img_size.
    """
    dims = sorted(dims)
    f_shape = filter.shape
    f_size = tuple(f_shape[d] for d in dims)
    i_size = tuple(img_size[d] for d in dims)
    pad = tuple(i - f for f, i in zip(reversed(f_size), reversed(i_size)))
    pad = tuple(
        chain.from_iterable((0, v) for v in pad)
    )  # (0, W_right, 0, H_bottom, 0, D_back, ...)

    filter = F.pad(filter, pad, mode="constant", value=0)

    shifts = tuple(-int((f - 1) / 2) for f in f_size)
    filter = torch.roll(filter, shifts=shifts, dims=dims)
    return fft.rfftn(filter, dim=dims) if real_fft else fft.fftn(filter, dim=dims)


# Keep it for backward compatibility
@_deprecated_func_replaced_by(
    replacement="deepinv.physics.functional.filter_fft", since="0.3.4"
)
def filter_fft_2d(
    filter: Tensor, img_size: tuple[int, int], real_fft: bool = True
) -> Tensor:
    r"""
    A helper function to compute the centered FFT of a 2D filter zero-padded to img_size.
    """
    return filter_fft(filter, img_size=img_size, real_fft=real_fft, dims=(-2, -1))


def _flip_filter_if_needed(
    filter: Tensor, correlation: bool, dims: tuple[int, ...]
) -> Tensor:
    # Flip the kernel for true convolution
    if not correlation:
        filter = filter.flip(dims=dims)
    return filter.contiguous()
