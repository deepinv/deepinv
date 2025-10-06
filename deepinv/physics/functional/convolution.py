import torch
import torch.nn.functional as F
from torch import Tensor
import torch.fft as fft


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
    :param bool correlation: choose True if you want a cross-correlation (default False)

    .. note:

        Contrarily to Pytorch :func:`torch.functional.conv2d`, which performs a cross-correlation, this function performs a convolution.

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param padding: (options = ``valid``, ``circular``, ``replicate``, ``reflect``, ``constant``) If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.
        ``constant`` corresponds to zero padding or ``same`` in :func:`torch.nn.functional.conv2d`
    :return: (:class:`torch.Tensor`) : the output

    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    if not correlation:
        filter = torch.flip(filter, [-2, -1])

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1)

    if padding != "valid":
        ph = h // 2
        ih = (h - 1) % 2
        pw = w // 2
        iw = (w - 1) % 2
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

    return output


def conv_transpose2d(
    y: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of x and filter. The transposed of this operation is :func:`deepinv.physics.functional.conv2d`

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` ) where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool correlation: choose True if you want a cross-correlation (default False)

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the output is larger than the image (padding)
        otherwise the output has the same size as the image.

    :return: (:class:`torch.Tensor`) : the output
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    if not correlation:
        filter = torch.flip(filter, [-2, -1])

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    ph = h // 2
    pw = w // 2
    ih = (h - 1) % 2
    iw = (w - 1) % 2

    if padding != "valid":
        if ph == 0 or pw == 0:
            raise ValueError(
                "Both dimensions of the filter must be strictly greater than 2 if padding != 'valid'"
            )

    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1)

    # Move batch dim of the input into channels
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
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]

        # sides and corners
        for sy in (-1, 0, 1):
            for sx in (-1, 0, 1):
                if sy == 0 and sx == 0:
                    continue
                ty, sy = _tgt_src_for_axis_circular(sy, ph, ih)
                tx, sx = _tgt_src_for_axis_circular(sx, pw, iw)
                out[:, :, ty, tx].add_(x[:, :, sy, sx])

    elif padding == "reflect":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]

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
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]

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
        out = x[:, :, ph : -(ph - ih), pw : -(pw - iw)]

    else:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )

    return out


def conv2d_fft(x: Tensor, filter: Tensor, real_fft: bool = True) -> torch.Tensor:
    r"""
    A helper function performing the 2d convolution of images ``x`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose2d_fft`

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as
    `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    For convolution using FFT consider only ``'circular'`` padding (i.e., circular convolution).

    .. note:

        The convolution here is a convolution, not a correlation as in conv2d.

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :return: torch.Tensor : the output of the convolution of the shape size as :math:`x`
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()
    img_size = x.shape[1:]

    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"

    filter_f = filter_fft_2d(filter, img_size, real_fft)
    x_f = fft.rfft2(x) if real_fft else fft.fft2(x)

    return fft.irfft2(x_f * filter_f).real


def conv_transpose2d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True
) -> torch.Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of ``x`` and ``filter`` using FFT. The adjoint of this operation is :func:`deepinv.physics.functional.conv2d_fft`.

    :param torch.Tensor y: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` ) where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    For convolution using FFT consider only ``'circular'`` padding (i.e., circular convolution).

    :return: torch.Tensor : the output of the convolution, which has the same shape as :math:`y`
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()
    img_size = (C, H, W)

    if c != C:
        assert c == 1

    if b != B:
        assert b == 1

    filter_f = filter_fft_2d(filter, img_size, real_fft)
    y_f = fft.rfft2(y) if real_fft else fft.fft2(y)

    return fft.irfft2(y_f * torch.conj(filter_f), s=(H, W)).real


def filter_fft_2d(filter, img_size, real_fft=True):
    ph = int((filter.shape[-2] - 1) / 2)
    pw = int((filter.shape[-1] - 1) / 2)

    filt2 = torch.zeros(
        tuple(filter.shape[:2]) + tuple(img_size[-2:]), device=filter.device
    )

    filt2[..., : filter.shape[-2], : filter.shape[-1]] = filter
    filt2 = torch.roll(filt2, shifts=(-ph, -pw), dims=(-2, -1))

    return fft.rfft2(filt2) if real_fft else fft.fft2(filt2)


def conv3d(
    x: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function to perform 3D convolution of images :math:``x`` and ``filter``.

    The transposed of this operation is :func:`deepinv.physics.functional.conv_transpose3d`.

    :param torch.Tensor x: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param str padding: can be ``'valid'`` (default), ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``. If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.

    :return: (:class:`torch.Tensor`) : the output of the convolution, which has the shape ``(B, C, D-d+1, W-w+1, H-h+1)`` if ``padding = 'valid'`` and the same shape as ``x`` otherwise.
    """
    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    B, C, D, H, W = x.shape
    b, c, kd, kh, kw = filter.shape

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert c == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1, -1)
    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1, -1)

    # Flip the kernel for true convolution
    if not correlation:
        filter = filter.flip(dims=[2, 3, 4])

    # Determine padding
    if padding.lower() != "valid":
        # Calculate padding to keep output same size as input
        pad_d = kd // 2
        pad_h = kh // 2
        pad_w = kw // 2
        pad = (
            pad_w,
            pad_w,
            pad_h,
            pad_h,
            pad_d,
            pad_d,
        )  # F.pad expects (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x = F.pad(x, pad, mode=padding)

    # Grouped convolution trick for per-batch filters and channels
    x = x.reshape(1, B * C, x.shape[2], x.shape[3], x.shape[4])
    filter_reshaped = filter.reshape(B * C, 1, kd, kh, kw)

    out = F.conv3d(x, filter_reshaped, groups=B * C)

    # Reshape back to (B, C, D_out, H_out, W_out)
    D_out, H_out, W_out = out.shape[2], out.shape[3], out.shape[4]
    out = out.reshape(B, C, D_out, H_out, W_out)

    return out


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

    :return: (:class:`torch.Tensor`) : the output of the convolution, which has the shape ``(B, C, D+d-1, W+w-1, H+h-1)`` if ``padding = 'valid'`` and the same shape as ``y`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    B, C, D, H, W = y.shape
    b, c, d, h, w = filter.shape

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert c == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1, -1)
    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1, -1)

    # Flip the kernel for true convolution
    if not correlation:
        filter = filter.flip(dims=[2, 3, 4])

    # Use grouped convolution trick for per-batch filters and channels
    y = y.reshape(1, B * C, D, H, W)
    filter = filter.reshape(B * C, 1, d, h, w)

    x = F.conv_transpose3d(y, filter, groups=B * C)
    x = x.reshape(B, C, x.shape[2], x.shape[3], x.shape[4])

    if padding == "valid":
        out = x
    else:
        pd = d // 2
        ph = h // 2
        pw = w // 2
        id = (d - 1) % 2
        ih = (h - 1) % 2
        iw = (w - 1) % 2

        if pd == 0 or ph == 0 or pw == 0:
            raise ValueError(
                "All three dimensions of the filter must be strictly greater than 2 if padding != 'valid'"
            )
        if padding == "circular":
            # Start from the central crop
            out = x[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw]

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
            out = x[:, :, pd : -(pd - id), ph : -(ph - ih), pw : -(pw - iw)]
        elif padding == "reflect":
            # Center crop
            out = x[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw]
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

            out = x[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw]
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
                        chunk = x[:, :, zsrc, ysrc, xsrc]
                        if reduce_dims:
                            chunk = chunk.sum(dim=reduce_dims)
                        out[:, :, tz, ty, tx].add_(chunk)
        else:
            raise ValueError(
                f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
            )
    return out


# Map the shift {-1,0,+1} to (target slice on out, source slice on x) per axis
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


def conv3d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d convolution of ``x`` and `filter` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose3d_fft`.

    If ``b = 1`` or ``c = 1``, this function applies the same filter for each channel and each image.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    Padding conditions include ``'circular'`` and ``'valid'``.

    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose True (default) to accelerate computation
    :param str padding: can be ``'valid'`` (default) or ``'circular'``

    .. note::
        The filter center is located at ``(d//2, h//2, w//2)``.

    :return: torch.Tensor : the output of the convolution, which has the same shape as :math:``x`` if ``padding = 'circular'``, ``(B, C, D-d+1, W-w+1, H-h+1)`` if ``padding = 'valid'``
    """

    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    B, C, D, H, W = x.size()
    img_size = x.shape[-3:]
    b, c, d, h, w = filter.size()

    if c != C:
        assert c == 1
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert b == 1
        filter = filter.expand(B, -1, -1, -1, -1)

    if real_fft:
        f_f = fft.rfftn(filter, s=img_size, dim=(-3, -2, -1))
        x_f = fft.rfftn(x, dim=(-3, -2, -1))
        res = fft.irfftn(x_f * f_f, s=img_size, dim=(-3, -2, -1))
    else:
        f_f = fft.fftn(filter, s=img_size, dim=(-3, -2, -1))
        x_f = fft.fftn(x, dim=(-3, -2, -1))
        res = fft.ifftn(x_f * f_f, s=img_size, dim=(-3, -2, -1))

    if padding == "valid":
        return res[:, :, d - 1 :, h - 1 :, w - 1 :]
    elif padding == "circular":
        shifts = (-(d // 2), -(h // 2), -(w // 2))
        return torch.roll(res, shifts=shifts, dims=(-3, -2, -1))
    else:
        raise ValueError("padding = '" + padding + "' not implemented")


def conv_transpose3d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d transposed convolution of ``y`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv3d_fft`.

    If ``b = 1`` or ``c = 1``, then this function applies the same filter for each channel.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    Padding conditions include ``'circular'`` and ``'valid'``.

    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose True (default) to accelerate computation
    :param str padding: can be ``'valid'`` (default) or ``'circular'``

    :return: torch.Tensor : the output of the convolution, which has the same shape as :math:`y`
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    # Get dimensions of the input and the filter
    B, C, D, H, W = y.size()
    b, c, d, h, w = filter.size()
    if padding == "valid":
        img_size = (D + d - 1, H + h - 1, W + w - 1)
    elif padding == "circular":
        img_size = (D, H, W)
        shifts = (d // 2, h // 2, w // 2)
        y = torch.roll(y, shifts=shifts, dims=(-3, -2, -1))
    else:
        raise ValueError("padding = '" + padding + "' not implemented")

    if c != C:
        assert c == 1
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert b == 1
        filter = filter.expand(B, -1, -1, -1, -1)

    if real_fft:
        f_f = fft.rfftn(filter, s=img_size, dim=(-3, -2, -1))
        y_f = fft.rfftn(y, s=img_size, dim=(-3, -2, -1))
        res = fft.irfftn(y_f * torch.conj(f_f), s=img_size, dim=(-3, -2, -1))
    else:
        f_f = fft.fftn(filter, s=img_size, dim=(-3, -2, -1))
        y_f = fft.fftn(y, s=img_size, dim=(-3, -2, -1))
        res = fft.ifftn(y_f * torch.conj(f_f), s=img_size, dim=(-3, -2, -1))

    if padding == "valid":
        return torch.roll(res, shifts=(d - 1, h - 1, w - 1), dims=(-3, -2, -1))
    else:
        return res
