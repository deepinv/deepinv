import torch
import torch.nn.functional as F
from torch import Tensor
import torch.fft as fft
import warnings

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
        
        if ph == 0 and pw == 0:
            warnings.warn(f"Use are using padding = '{padding}' with a 1x1 kernel. This is equivalent to no padding. "
                          f"Consider using padding = 'valid' instead.", UserWarning)
            
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

    if padding != "valid" and (ph == 0 and pw == 0):
        warnings.warn(f"Use are using padding = '{padding}' with a 1x1 kernel. This is equivalent to no padding. "
                      f"Consider using padding = 'valid' instead.", UserWarning)

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
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )

    return out


def conv2d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "circular"
) -> torch.Tensor:
    r"""
    A helper function performing the 2d convolution of images ``x`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose2d_fft`

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as
    `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    .. note::

        The convolution here is a convolution, not a correlation as in :func:`torch.nn.functional.conv2d`.

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation.
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding),
        otherwise the output has the same size as the image. Default is ``'circular'``.
    :return: torch.Tensor : the output of the convolution of the shape size as :math:`x`
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    if c != C:
        assert (
            c == 1
        ), "Number of channels of the kernel is not matched for broadcasting"
    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"

    ph, pw = h // 2, w // 2
    ih, iw = (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (ph == 0 and pw == 0):
        warnings.warn(f"Use are using padding = '{padding}' with a 1x1 kernel. This is equivalent to no padding. "
                      f"Consider using padding = 'valid' instead.", UserWarning)
    def fft2(t, s=None):
        return fft.rfft2(t, s=s) if real_fft else fft.fft2(t, s=s)

    def ifft2(t, s):
        return fft.irfft2(t, s=s).real if real_fft else fft.ifft2(t, s=s).real

    if padding == "circular":
        # Circular convolution with kernel center aligned via filter centering
        img_size = (H, W)
        filt_shifted = _center_filter_2d(filter, img_size)
        fx = fft2(x, s=img_size)
        ff = fft2(filt_shifted, s=img_size)
        return ifft2(fx * ff, s=img_size)

    elif padding == "valid":
        # Full linear convolution then crop to valid window
        sH, sW = H + h - 1, W + w - 1
        img_size = (sH, sW)
        fx = fft2(x, s=img_size)
        ff = fft2(filter, s=img_size)
        full = ifft2(fx * ff, s=img_size)
        return full[:, :, h - 1 : H, w - 1 : W]

    elif padding in ("constant", "reflect", "replicate"):
        # Linear convolution on a padded grid via circular FFT-conv on that grid.
        pad = (pw, pw - iw, ph, ph - ih)  # (W_left, W_right, H_top, H_bottom)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        img_size = x_pad.shape[-2:]

        # Center filter on (0,0) in the padded grid
        filt_shifted = _center_filter_2d(filter, img_size)

        fx = fft2(x_pad, s=img_size)
        ff = fft2(filt_shifted, s=img_size)
        y_pad = ifft2(fx * ff, s=img_size)

        # Extract central region back to original size
        return y_pad[:, :, _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

    else:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )

def conv_transpose2d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "circular"
) -> torch.Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of ``x`` and ``filter`` using FFT.
    The adjoint of this operation is :func:`deepinv.physics.functional.conv2d_fft`.

    :param torch.Tensor y: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` ) where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.

    If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation.
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``. If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image. Default is ``'circular'``.

    :return: torch.Tensor : the output of the convolution, which has the same shape as :math:`y`
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
    
    if padding != "valid":
        if ph == 0 and pw == 0:
            warnings.warn(f"Use are using padding = '{padding}' with a 1x1 kernel. This is equivalent to no padding. "
                          f"Consider using padding = 'valid' instead.", UserWarning)
            
    ih, iw = (h - 1) % 2, (w - 1) % 2

    def fft2(t, s=None):
        return fft.rfft2(t, s=s) if real_fft else fft.fft2(t, s=s)

    def ifft2(t, s):
        return fft.irfft2(t, s=s).real if real_fft else fft.ifft2(t, s=s).real

    if padding == "circular":
        # Circular adjoint: multiply by conj of centered filter FFT, no roll.
        img_size = (H, W)
        filt_shifted = _center_filter_2d(filter, img_size)
        fy = fft2(y, s=img_size)
        ff = fft2(filt_shifted, s=img_size)
        return ifft2(fy * torch.conj(ff), s=img_size)

    elif padding == "valid":
        # Adjoint of full-conv + center crop
        sH, sW = H + h - 1, W + w - 1
        img_size = (sH, sW)
        y_full = F.pad(y, (w - 1, w - 1, h - 1, h - 1), mode="constant", value=0)
        fy = fft2(y_full, s=img_size)
        ff = fft2(filter, s=img_size)
        return ifft2(fy * torch.conj(ff), s=img_size)

    elif padding in ("constant", "reflect", "replicate"):
        # Forward: pad (P) -> conv (C) -> crop (S)
        # Adjoint:  S* -> C* -> P*
        Hp = H + ph + (ph - ih)
        Wp = W + pw + (pw - iw)
        img_size = (Hp, Wp)

        # S*: embed y into center of padded grid
        y_big = F.pad(y, (pw, pw - iw, ph, ph - ih), mode="constant", value=0)
        # C*: circular transpose conv on padded grid using centered filter
        filt_shifted = _center_filter_2d(filter, (Hp, Wp))
        fy = fft2(y_big, s=img_size)
        ff = fft2(filt_shifted, s=img_size)
        z_big = ifft2(fy * torch.conj(ff), s=img_size)

        # P*: adjoint of padding -> fold to original H x W
        if padding == "constant":
            out = z_big[
                :, :,
                _center_crop_slice_1d(ph, ih),
                _center_crop_slice_1d(pw, iw),
            ]
        elif padding == "reflect":
            out = z_big[
                :, :,
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
                :, :,
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
        return out

    else:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )

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
    b, c, d, h, w = filter.shape

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
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
            warnings.warn(f"Use are using padding = '{padding}' with a 1x1x1 kernel. This is equivalent to no padding. "
                          f"Consider using padding = 'valid' instead.", UserWarning)
            
        B, C, D, H, W = x.shape

    # Grouped convolution trick for per-batch filters and channels
    x = x.reshape(1, B * C, D, H, W)
    filter = filter.reshape(B * C, -1, d, h, w)
    out = F.conv3d(x, filter, padding='valid', groups=B * C)
    # Make it in the good shape
    out = out.reshape(B, C, *out.shape[-3:])
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
    
    pd = d // 2
    ph = h // 2
    pw = w // 2
    id = (d - 1) % 2
    ih = (h - 1) % 2
    iw = (w - 1) % 2
    
    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        warnings.warn(f"Use are using padding = '{padding}' with a 1x1x1 kernel. This is equivalent to no padding. "
                      f"Consider using padding = 'valid' instead.", UserWarning)

    # Adjust filter shape if batch or channel is 1
    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
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
    x = x.reshape(B, C, *x.shape[-3:])

    if padding == "valid":
        out = x
    elif padding == "circular":
        # Start from the central crop
        out = x[:, :, _center_crop_slice_1d(pd, id), _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]
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
        out = x[:,:, _center_crop_slice_1d(pd, id), _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]
    elif padding == "reflect":
        # Center crop
        out = x[:, :, _center_crop_slice_1d(pd, id), _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]
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
        out = x[:, :, _center_crop_slice_1d(pd, id), _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]
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


def conv3d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "circular"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d convolution of ``x`` and `filter` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose3d_fft`.

    If ``b = 1`` or ``c = 1``, this function applies the same filter for each channel and each image.
    Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose True (default) to accelerate computation
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image. Default is ``'circular'``.

    .. note::

        The filter center is located at ``(d//2, h//2, w//2)``.

    .. tip::

        This function and :func:`deepinv.physics.functional.conv3d` are equivalent. However, this function is more efficient for large filters.

    :return: torch.Tensor : the output of the convolution, which has the same shape as :math:``x`` if ``padding = 'circular'``, ``(B, C, D-d+1, W-w+1, H-h+1)`` otherwise.
    """

    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    B, C, D, H, W = x.size()
    b, c, d, h, w = filter.size()

    if c != C:
        assert c == 1, "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1, -1)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        warnings.warn(
            f"Use are using padding = '{padding}' with a 1x1x1 kernel. This is equivalent to no padding. "
            f"Consider using padding = 'valid' instead.",
            UserWarning,
        )

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

    if padding == "valid":
        # Linear full convolution then crop to the valid window
        sD, sH, sW = D + d - 1, H + h - 1, W + w - 1
        fx = fft3(x, s=(sD, sH, sW))
        ff = fft3(filter, s=(sD, sH, sW))
        full = ifft3(fx * ff, s=(sD, sH, sW))
        return full[:, :, d - 1 : D, h - 1 : H, w - 1 : W]

    elif padding in ("constant", "reflect", "replicate", "circular"):
        # Match conv3d: symmetric pre-padding, then valid conv on the padded grid
        pad = (pw, pw - iw, ph, ph - ih, pd, pd - id)  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        img_size = x_pad.shape[-3:]

        # Full linear conv on padded grid via FFT, then take valid part
        fx = fft3(x_pad, s=img_size)
        ff = fft3(filter, s=img_size)
        y_pad = ifft3(fx * ff, s=img_size)
        
        # Extract central region back to original size
        return y_pad[:, :, _center_crop_slice_1d(pd, id), _center_crop_slice_1d(ph, ih), _center_crop_slice_1d(pw, iw)]

    else:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )


def conv_transpose3d_fft(
    y: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "circular"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d transposed convolution of ``y`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv3d_fft`.

    If ``b = 1`` or ``c = 1``, then this function applies the same filter for each channel.
    Otherwise, each channel of each image is convolved with the corresponding kernel.


    :param torch.Tensor y: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param bool real_fft: for real filters and images choose True (default) to accelerate computation
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'``.
        If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image. Default is ``'circular'``.

    .. tip::

        This function and :func:`deepinv.physics.functional.conv_transposed3d` are equivalent. However, this function is more efficient for large filters.

    :return: torch.Tensor : the output of the transposed convolution, which has the same shape as :math:``y`` if ``padding = 'circular'``, ``(B, C, D+d-1, W+w-1, H+h-1)`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"

    # Get dimensions of the input and the filter
    B, C, D, H, W = y.size()
    b, c, d, h, w = filter.size()

    if c != C:
        assert c == 1, "Number of channels of the kernel is not matched for broadcasting"
        filter = filter.expand(-1, C, -1, -1, -1)

    if b != B:
        assert b == 1, "Batch size of the kernel is not matched for broadcasting"
        filter = filter.expand(B, -1, -1, -1, -1)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2
    
    if padding != "valid":
        if pd == 0 and pw == 0 and ph == 0:
            warnings.warn(f"Use are using padding = '{padding}' with a 1x1x1 kernel. This is equivalent to no padding. "
                          f"Consider using padding = 'valid' instead.", UserWarning)

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
        filt_shifted = _center_filter_3d(filter, img_size)
        fx = fft3(x, s=img_size)
        ff = fft3(filt_shifted, s=img_size)
        return ifft3(fx * ff, s=img_size)

    elif padding == "valid":
        # Adjoint of linear conv + valid crop: embed y into center of full grid
        sD, sH, sW = D + d - 1, H + h - 1, W + w - 1
        img_size = (sD, sH, sW)
        y_full = torch.zeros((B, C, sD, sH, sW), device=y.device, dtype=y.dtype)
        y_full[:, :, d - 1 : d - 1 + D, h - 1 : h - 1 + H, w - 1 : w - 1 + W] = y
        fy = fftn3(y_full, s=img_size)
        ff = fftn3(filter, s=img_size)
        return ifftn3(fy * torch.conj(ff), s=img_size)

    elif padding in ("constant", "reflect", "replicate"):
        if pd == 0 or ph == 0 or pw == 0:
            raise ValueError(
                "All three dimensions of the filter must be strictly greater than 1 for this padding mode."
            )

        # Forward: pad (P) -> conv (C) -> roll (R) -> crop (S)
        # Adjoint:  S* -> R* -> C* -> P*
        Dp = D + pd + (pd - id)
        Hp = H + ph + (ph - ih)
        Wp = W + pw + (pw - iw)
        img_size = (Dp, Hp, Wp)

        # S*: embed y into center of padded grid
        y_big = torch.zeros((B, C, Dp, Hp, Wp), device=y.device, dtype=y.dtype)
        y_big[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw] = y
        # R*: roll by +center
        y_big = torch.roll(y_big, shifts=(pd, ph, pw), dims=(-3, -2, -1))

        # C*: circular transpose conv on padded grid
        fy = fftn3(y_big, s=img_size)
        ff = fftn3(filter, s=img_size)
        z_big = ifftn3(fy * torch.conj(ff), s=img_size)

        # P*: adjoint of padding -> fold to original D x H x W
        if padding == "constant":
            out = z_big[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw]
        elif padding == "reflect":
            out = z_big[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw].clone()
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
            out = z_big[:, :, pd : -pd + id, ph : -ph + ih, pw : -pw + iw].clone()
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
        return out

    else:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect' or 'constant'."
        )


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


def _center_filter_2d(filter, img_size):
    r"""
    A helper function to zero-padded to img_size and center the frequencies.
    """
    b, c, fh, fw = filter.shape
    ih, iw = img_size[-2:] 
    ph = int((fh - 1) / 2)
    pw = int((fw - 1) / 2)
    filter = F.pad(filter, (0, iw - fw, 0, ih - fh))
    filter = torch.roll(filter, shifts=(-ph, -pw), dims=(-2, -1))
    return filter

def _center_filter_3d(filter, img_size):
    r"""
    A helper function to zero-pad to img_size and center the filter spatially.
    """
    b, c, fd, fh, fw = filter.shape
    id, ih, iw = img_size[-3:]
    pd = int((fd - 1) / 2)
    ph = int((fh - 1) / 2)
    pw = int((fw - 1) / 2)
    filter = F.pad(filter, (0, iw - fw, 0, ih - fh, 0, id - fd))
    filter = torch.roll(filter, shifts=(-pd, -ph, -pw), dims=(-3, -2, -1))
    return filter