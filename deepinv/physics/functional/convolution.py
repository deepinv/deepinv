import torch
import torch.nn.functional as F
from torch import Tensor
import torch.fft as fft
import warnings
from itertools import chain, product
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
        warnings.warn(message, category, stacklevel=2)
        _warned_messages.add(message)


def _raise_value_error_padding_messages(padding):
    if padding.lower() not in [
        "valid",
        "circular",
        "replicate",
        "reflect",
        "zeros",
        "constant",
    ]:
        raise ValueError(
            f"padding = '{padding}' not implemented. Please use one of 'valid', 'circular', 'replicate', 'reflect', 'constant' or 'zeros'."
        )
    else:
        if padding.lower() == "zeros":
            padding = "constant"
    return padding


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

    :param padding: (options = ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'`` or ``'zeros'``). If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image. Note that ``'constant'`` and ``'zeros'`` are equivalent. Default is ``'valid'``.
    :return: :class:`torch.Tensor`: the blurry output.

    .. note::

        Contrary to PyTorch's :func:`torch.nn.functional.conv2d`, which performs a cross-correlation, this function performs a convolution by default unless ``correlation=True``.

        This function gives the same result as :func:`deepinv.physics.functional.conv2d_fft`. However, for small kernels, this function is faster.
        For large kernels, :func:`deepinv.physics.functional.conv2d_fft` is usually faster but requires more memory.

    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"
    padding = _raise_value_error_padding_messages(padding)

    filter = _flip_filter_if_needed(filter, correlation, dims=(-2, -1))

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    # Ensure fast memory layout before heavy ops.
    x = x.contiguous()

    # Prepare filter for grouped conv trick (supports broadcasting b==1 or c==1)
    filter = _prepare_filter_for_grouped(filter, B, C)

    if padding != "valid":
        ph = h // 2
        ih = (h - 1) % 2
        pw = w // 2
        iw = (w - 1) % 2
        if ph == 0 and pw == 0:
            _warn_once_padding([ph, pw], padding)

        pad = (pw - iw, pw, ph - ih, ph)  # because functional.pad is w,h instead of h,w
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

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'`` or ``'zeros'``.
        If ``padding='valid'`` the output is larger than the image (padding) the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent. Default is ``'valid'``.

    :return: :class:`torch.Tensor` : the output

    .. note::

        This functions gives the same result as :func:`deepinv.physics.functional.conv_transpose2d_fft`. However, for small kernels, this function is faster.
        For large kernels, :func:`deepinv.physics.functional.conv_transpose2d_fft` is usually faster but requires more memory.

    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"
    padding = _raise_value_error_padding_messages(padding)
    filter = _flip_filter_if_needed(filter, correlation, dims=(-2, -1))

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    ph, pw, ih, iw = h // 2, w // 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    # Prepare filter for grouped conv trick (supports broadcasting b==1 or c==1)
    filter = _prepare_filter_for_grouped(filter, B, C)

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
        return x
    return _apply_transpose_padding(x, padding=padding, p=(ph, pw), i=(ih, iw))


def conv2d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 2d convolution of images ``x`` and ``filter`` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose2d_fft`

    .. note::

        The convolution here is a convolution, not a correlation as in :func:`torch.nn.functional.conv2d`.
        This function gives the same result as :func:`deepinv.physics.functional.conv2d` and is faster for large kernels.

    :param torch.Tensor x: Image of size ``(B, C, W, H)``.
    :param torch.Tensor filter: Filter of size ``(b, c, w, h)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
        If ``b = 1`` or ``c = 1``, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
        Otherwise, each channel of each image is convolved with the corresponding kernel.
    :param bool real_fft: for real filters and images choose `True` (default) to accelerate computation.
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``,  ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding),
        otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent. Default is ``'valid'``.
    :return: :class:`torch.Tensor`: the output of the convolution of the shape size as `x`.
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"
    padding = _raise_value_error_padding_messages(padding)

    # Get dimensions of the input and the filter
    B, C, H, W = x.size()
    b, c, h, w = filter.size()

    filter = _prepare_filter_for_grouped(filter, B, C, raise_error_only=True)

    ph, pw, ih, iw = h // 2, w // 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    x = x.contiguous()
    if padding == "circular":
        # Circular convolution with kernel center aligned via filter centering
        out = _circular_conv_fft(
            x, filter, s=(H, W), real_fft=real_fft, dims=(-2, -1), shift_filter=True
        )

    elif padding == "valid":
        # Full linear convolution then crop to valid window
        full = _circular_conv_fft(
            x,
            filter,
            s=(H + h - 1, W + w - 1),
            real_fft=real_fft,
            dims=(-2, -1),
            shift_filter=False,
        )
        out = full[:, :, h - 1 : H, w - 1 : W]

    else:
        # Linear convolution on a padded grid via circular FFT-conv on that grid.
        pad = (pw, pw, ph, ph)  # (W_left, W_right, H_top, H_bottom)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        out = _circular_conv_fft(
            x_pad,
            filter,
            s=x_pad.shape[-2:],
            real_fft=real_fft,
            dims=(-2, -1),
            shift_filter=True,
        )
        # Extract central region back to original size
        out = out[:, :, _center_crop_slice_1d(ph, 0), _center_crop_slice_1d(pw, 0)]

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
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``,  ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent.
        Default is ``'valid'``.

    :return: :class:`torch.Tensor`: the output of the convolution, which has the same shape as :math:`y`.

    .. note::
        This functions gives the same result as :func:`deepinv.physics.functional.conv_transpose2d`.
        However, for large kernels, this function is faster but requires more memory.
        For small kernels, consider using :func:`deepinv.physics.functional.conv_transpose2d`.
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"
    padding = _raise_value_error_padding_messages(padding)

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    filter = _prepare_filter_for_grouped(filter, B, C, raise_error_only=True)

    ph, pw = h // 2, w // 2

    if padding != "valid" and (ph == 0 and pw == 0):
        _warn_once_padding([ph, pw], padding)

    ih, iw = (h - 1) % 2, (w - 1) % 2

    y = y.contiguous()
    filter = filter.contiguous()

    if padding == "circular":
        # Circular adjoint: multiply by conj of centered filter FFT, no roll.
        out = _circular_conv_fft(
            y,
            filter,
            s=(H, W),
            real_fft=real_fft,
            dims=(-2, -1),
            shift_filter=True,
            transpose=True,
        )
    elif padding == "valid":
        # Adjoint of full-conv + center crop
        y_full = F.pad(y, (w - 1, w - 1, h - 1, h - 1), mode="constant", value=0)
        out = _circular_conv_fft(
            y_full,
            filter,
            s=(H + h - 1, W + w - 1),
            real_fft=real_fft,
            dims=(-2, -1),
            transpose=True,
        )

    else:
        # Forward: pad (P) -> conv (C) -> crop (S)
        # Adjoint:  S* -> C* -> P*
        # S*: embed y into center of padded grid
        y_big = F.pad(y, (pw, pw, ph, ph), mode="constant", value=0)
        # C*: circular transpose conv on padded grid using centered filter
        z_big = _circular_conv_fft(
            y_big,
            filter,
            s=(H + 2 * ph, W + 2 * pw),
            real_fft=real_fft,
            dims=(-2, -1),
            shift_filter=True,
            transpose=True,
        )
        # P*: adjoint of padding -> fold to original H x W
        z_big = z_big[..., ih:, iw:]
        out = _apply_transpose_padding(z_big, padding=padding, p=(ph, pw), i=(ih, iw))

    return out.contiguous()


def conv3d(
    x: Tensor, filter: Tensor, padding: str = "valid", correlation=False
) -> torch.Tensor:
    r"""
    A helper function to perform 3D convolution of images `x` and `filter`.

    The transposed of this operation is :func:`deepinv.physics.functional.conv_transpose3d`.

    :param torch.Tensor x: Image of size ``(B, C, D, H, W)``.
    :param torch.Tensor filter: Filter of size ``(b, c, d, h, w)`` where ``b`` can be either ``1`` or ``B`` and ``c`` can be either ``1`` or ``C``.
    :param str padding: can be ``'valid'`` (default), ``'circular'``, ``'replicate'``, ``'reflect'``,  ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent. Default is ``'valid'``.
    :return: :class:`torch.Tensor`: the output of the convolution, which has the shape ``(B, C, D-d+1, W-w+1, H-h+1)`` if ``padding = 'valid'`` and the same shape as ``x`` otherwise.

    .. note::

        Contrary to Pytorch's :func:`torch.nn.functional.conv3d`, which performs a cross-correlation, this function performs a convolution by default unless ``correlation=True``.

    """
    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    padding = _raise_value_error_padding_messages(padding)

    B, C, D, H, W = x.shape
    b, c, d, h, w = filter.shape

    # Prepare filter for grouped conv trick (supports broadcasting b==1 or c==1)
    filter = _prepare_filter_for_grouped(filter, B, C)

    # Flip the kernel for true convolution
    filter = _flip_filter_if_needed(filter, correlation, dims=(-3, -2, -1))
    x = x.contiguous()
    # Determine padding
    if padding.lower() != "valid":
        # Calculate padding to keep output same size as input
        pd, ph, pw = d // 2, h // 2, w // 2
        id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2
        pad = (
            pw - iw,
            pw,
            ph - ih,
            ph,
            pd - id,
            pd,
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
    :param str padding: can be ``'valid'`` (default), ``'circular'``, ``'replicate'``, ``'reflect'``, ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent. Default is ``'valid'``.
    :param bool correlation: choose `True` if you want the transpose of the cross-correlation (default `False`).

    :return: :class:`torch.Tensor`: the output of the convolution, which has the shape ``(B, C, D+d-1, W+w-1, H+h-1)`` if ``padding = 'valid'`` and the same shape as ``y`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    padding = _raise_value_error_padding_messages(padding)
    B, C, D, H, W = y.shape
    b, c, d, h, w = filter.shape

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)
    # Flip the kernel for true convolution
    filter = _flip_filter_if_needed(filter, correlation, dims=(-3, -2, -1))

    # Prepare filter for grouped conv trick (supports broadcasting b==1 or c==1)
    filter = _prepare_filter_for_grouped(filter, B, C)

    # Use grouped convolution trick for per-batch filters and channels
    y = y.contiguous()
    y = y.reshape(1, B * C, D, H, W)
    filter = filter.reshape(B * C, 1, d, h, w)

    x = F.conv_transpose3d(y, filter, groups=B * C)
    x = x.reshape(B, C, *x.shape[-3:])

    out = _apply_transpose_padding(x, padding=padding, p=(pd, ph, pw), i=(id, ih, iw))
    return out


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
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``,  ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is smaller than the image (no padding), otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent.
        Default is ``'valid'``.

    .. note::

        The filter center is located at ``(d//2, h//2, w//2)``.

        This function and :func:`deepinv.physics.functional.conv3d` are equivalent. However, this function is more efficient for large filters but requires more memory.

    :return: :class:`torch.Tensor`: the output of the convolution, which has the same shape as :math:`x` if ``padding = 'circular'``, ``(B, C, D-d+1, W-w+1, H-h+1)`` otherwise.
    """

    assert x.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    padding = _raise_value_error_padding_messages(padding)

    B, C, D, H, W = x.size()
    b, c, d, h, w = filter.size()

    x = x.contiguous()
    filter = _prepare_filter_for_grouped(filter, B, C)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)

    if padding == "circular":
        # Circular convolution with kernel center aligned via filter centering
        img_size = (D, H, W)
        out = _circular_conv_fft(
            x,
            filter,
            s=img_size,
            real_fft=real_fft,
            dims=(-3, -2, -1),
            shift_filter=True,
        )

    elif padding == "valid":
        # Full linear convolution then crop to valid window
        out = _circular_conv_fft(
            x,
            filter,
            s=(D + d - 1, H + h - 1, W + w - 1),
            real_fft=real_fft,
            dims=(-3, -2, -1),
        )
        out = out[:, :, d - 1 : D, h - 1 : H, w - 1 : W]

    else:
        # Linear convolution on a padded grid via circular FFT-conv on that grid.
        pad = (
            pw,
            pw,
            ph,
            ph,
            pd,
            pd,
        )  # (W_left, W_right, H_top, H_bottom, D_front, D_back)
        x_pad = F.pad(x, pad, mode=padding, value=0)
        out = _circular_conv_fft(
            x_pad,
            filter,
            s=x_pad.shape[-3:],
            real_fft=real_fft,
            dims=(-3, -2, -1),
            shift_filter=True,
        )
        # Extract central region back to original size
        out = out[
            :,
            :,
            _center_crop_slice_1d(pd, 0),
            _center_crop_slice_1d(ph, 0),
            _center_crop_slice_1d(pw, 0),
        ]

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
    :param str padding: can be ``'valid'``, ``'circular'``, ``'replicate'``, ``'reflect'``,  ``'constant'`` or ``'zeros'``.
        If ``padding = 'valid'`` the output is larger than the image (padding), otherwise the output has the same size as the image.
        Note that ``'constant'`` and ``'zeros'`` are equivalent.
        Default is ``'valid'``.

    .. note::

        This function and :func:`deepinv.physics.functional.conv_transpose3d` are equivalent. However, this function is more efficient for large filters.

    :return: :class:`torch.Tensor`: the output of the transposed convolution, which has the same shape as ``y`` if ``padding = 'circular'``, ``(B, C, D+d-1, W+w-1, H+h-1)`` otherwise.
    """

    assert y.dim() == filter.dim() == 5, "Input and filter must be 5D tensors"
    padding = _raise_value_error_padding_messages(padding)
    # Get dimensions of the input and the filter
    B, C, D, H, W = y.size()
    b, c, d, h, w = filter.size()

    y = y.contiguous()
    filter = _prepare_filter_for_grouped(filter, B, C)

    pd, ph, pw = d // 2, h // 2, w // 2
    id, ih, iw = (d - 1) % 2, (h - 1) % 2, (w - 1) % 2

    if padding != "valid" and (pd == 0 and pw == 0 and ph == 0):
        _warn_once_padding([ph, pw, pd], padding)

    if padding == "circular":
        out = _circular_conv_fft(
            y,
            filter,
            s=(D, H, W),
            real_fft=real_fft,
            dims=(-3, -2, -1),
            shift_filter=True,
            transpose=True,
        )

    elif padding == "valid":
        y_full = F.pad(
            y, (w - 1, w - 1, h - 1, h - 1, d - 1, d - 1), mode="constant", value=0
        )

        out = _circular_conv_fft(
            y_full,
            filter,
            s=(D + d - 1, H + h - 1, W + w - 1),
            real_fft=real_fft,
            dims=(-3, -2, -1),
            transpose=True,
        )

    else:
        # Forward: pad (P) -> conv (C) -> crop (S)
        # Adjoint:  S* -> R* -> C* -> P*
        Dp = D + 2 * pd
        Hp = H + 2 * ph
        Wp = W + 2 * pw
        img_size = (Dp, Hp, Wp)

        # S*: embed y into center of padded grid
        y_big = F.pad(y, (pw, pw, ph, ph, pd, pd), mode="constant", value=0)

        # C*: circular transpose conv on padded grid
        z_big = _circular_conv_fft(
            y_big,
            filter,
            s=img_size,
            real_fft=real_fft,
            dims=(-3, -2, -1),
            shift_filter=True,
            transpose=True,
        )

        # P*: adjoint of padding -> fold to original D x H x W
        z_big = z_big[..., id:, ih:, iw:]
        out = _apply_transpose_padding(
            z_big, padding=padding, p=(pd, ph, pw), i=(id, ih, iw)
        )

    return out.contiguous()


# Some helper functions for computing the slice indices for padding modes in convolution operations
# Map the shift {-1,0,+1} to (target, source[, flip][, reduce]) per axis based on padding mode
# --------------------------------------------------------------------------------------------------
def _tgt_src_for_axis(mode: str, s: int, p: int, i: int):
    """
    Unified axis mapping for transpose padding accumulation.

    Args:
        mode: one of 'circular', 'reflect', 'replicate'
        s: shift in {-1, 0, +1}
        p: half-size along axis (e.g., h//2)
        i: parity (f-1) % 2 for filter along axis

    Returns:
        For 'circular': (target_slice, source_slice, False, False)
        For 'reflect' : (target_slice, source_slice, flip(bool), False)
        For 'replicate': (target_slice_or_index, source_slice, False, reduce(bool))
    """
    if mode == "circular":
        if s == 0:
            return slice(None), slice(p - i, -p if p > 0 else None), False, False
        elif s == -1:
            return slice(0, p), slice(-p, None), False, False
        else:
            return slice(-(p - i), None), slice(None, p - i), False, False

    if mode == "reflect":
        if s == 0:
            return slice(None), slice(p - i, -p if p > 0 else None), False, False
        elif s == -1:
            return slice(1, 1 + (p - i)), slice(0, p - i), True, False
        else:
            return slice(-p - 1, -1), slice(-p, None), True, False

    if mode == "replicate":
        if s == 0:
            return slice(None), slice(p - i, -p if p > 0 else None), False, False
        elif s == -1:
            return 0, slice(0, p - i), False, True
        else:
            return -1, slice(-p, None), False, True
    else:
        _raise_value_error_padding_messages(mode)


def _center_crop_slice_1d(p: int, i: int) -> slice:
    # When p == i == 0, return the whole axis; otherwise replicate p : -p + i
    # This prevent issues with 0:-0 slices
    return slice(None) if (p == 0 and i == 0) else slice(p - i, -p if p > 0 else None)


def _apply_transpose_padding(
    x: Tensor, padding: str, p: tuple[int, ...], i: tuple[int, ...]
) -> Tensor:
    """
    Fold/crop the result of a transpose convolution to handle padding modes for 2D or 3D.

    Args:
        x: result of conv_transposeNd with shape (B, C, ...spatial...)
        padding: one of 'circular', 'replicate', 'reflect', 'zeros'
        p: half sizes per spatial dim (e.g., (ph, pw) for 2D or (pd, ph, pw) for 3D)
        i: parity flags per spatial dim: (f-1) % 2 for each filter size

    Returns:
        Tensor shaped like the input image (center-cropped and with edge folding applied as needed).
    """

    if padding == "valid":
        # No padding to apply
        return x.contiguous()

    n_spatial = len(p)
    assert n_spatial in (2, 3), "Only 2D or 3D supported"

    # Build center crop
    center_slices = tuple(_center_crop_slice_1d(pk, ik) for pk, ik in zip(p, i))
    index = (slice(None), slice(None), *center_slices)
    out = x[index]

    if padding == "constant":
        return out.contiguous()

    # For modes that add side/corner contributions, clone to avoid aliasing issues
    out = out.clone()

    # Iterate over all combinations of shifts per spatial axis
    for shifts in product((-1, 0, 1), repeat=n_spatial):
        if all(s == 0 for s in shifts):
            continue

        tgt_indices = []
        src_slices = []
        flip_dims = []  # dims to flip for reflect
        reduce_dims = []  # dims to reduce (sum) for replicate

        for axis, s in enumerate(shifts):
            pk, ik = p[axis], i[axis]
            t, src, flip, red = _tgt_src_for_axis(padding, s, pk, ik)
            tgt_indices.append(t)
            src_slices.append(src)
            if flip:
                flip_dims.append(2 + axis)
            if red:
                reduce_dims.append(2 + axis)

        src_index = (slice(None), slice(None), *src_slices)
        tgt_index = (slice(None), slice(None), *tgt_indices)

        chunk = x[src_index]
        if padding == "reflect" and flip_dims:
            chunk = chunk.flip(dims=tuple(flip_dims))
        if padding == "replicate" and reduce_dims:
            # Sum over specified spatial dims (supports tuple of dims)
            chunk = chunk.sum(dim=tuple(reduce_dims))

        out[tgt_index].add_(chunk)

    return out.contiguous()


def _prepare_filter_for_grouped(
    filter: Tensor, B: int, C: int, *, raise_error_only: bool = False
) -> Tensor:
    """
    Prepare filter tensor for grouped convolution trick used in direct conv/transpose conv.
    Works for both 2D (b, c, h, w) and 3D (b, c, d, h, w) filters.
        Broadcasting behavior:
            - If c != C, require c == 1 and expand along channel to C.
            - If b != B, require b == 1 and expand along batch to B.
    """
    b, c = filter.shape[:2]
    spatial = filter.shape[2:]

    if c != C:
        assert (
            c == 1
        ), f"Number of channels of the kernel is not matched for broadcasting, got c={c} and C={C}"
        if not raise_error_only:
            filter = filter.expand(-1, C, *spatial)
    if b != B:
        assert (
            b == 1
        ), f"Batch size of the kernel is not matched for broadcasting, got b={b} and B={B}"
        if not raise_error_only:
            filter = filter.expand(B, -1, *spatial)

    return filter.contiguous()


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

    shifts = tuple(-int(f / 2) for f in f_size)
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


def _circular_conv_fft(
    x: Tensor,
    filter: Tensor,
    s: tuple[int, ...],
    real_fft: bool,
    dims: tuple[int, ...],
    shift_filter: bool = False,
    transpose: bool = False,
) -> Tensor:
    r"""
    A helper function performing the circular convolution of ``x`` and ``filter`` using FFT.
    Can be used for 1D, 2D or 3D convolution depending on the dims argument.
    """
    dims = sorted(dims)
    fx = fft.rfftn(x, s=s, dim=dims) if real_fft else fft.fftn(x, s=s, dim=dims)
    if shift_filter:
        ff = filter_fft(filter, img_size=s, real_fft=real_fft, dims=dims)
    else:
        ff = (
            fft.rfftn(filter, s=s, dim=dims)
            if real_fft
            else fft.fftn(filter, s=s, dim=dims)
        )
    prod = fx * ff if not transpose else fx * torch.conj(ff)
    return (
        fft.irfftn(prod, s=s, dim=dims).real
        if real_fft
        else fft.ifftn(prod, s=s, dim=dims).real
    )
