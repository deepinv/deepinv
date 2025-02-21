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

    :param padding: (options = ``valid``, ``circular``, ``replicate``, ``reflect``, ``constant``) If ``padding = 'valid'`` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.
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
        assert c == 1
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1
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
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

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
        assert c == 1
        filter = filter.expand(-1, C, -1, -1)

    if b != B:
        assert b == 1
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
        # sides
        out[:, :, : ph - ih, :] += x[:, :, -ph + ih :, pw : -pw + iw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw : -pw + iw]
        out[:, :, :, : pw - iw] += x[:, :, ph : -ph + ih, -pw + iw :]
        out[:, :, :, -pw:] += x[:, :, ph : -ph + ih, :pw]
        # corners
        out[:, :, : ph - ih, : pw - iw] += x[:, :, -ph + ih :, -pw + iw :]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, : ph - ih, -pw:] += x[:, :, -ph + ih :, :pw]
        out[:, :, -ph:, : pw - iw] += x[:, :, :ph, -pw + iw :]

    elif padding == "reflect":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw : -pw + iw].flip(dims=(2,))
        out[:, :, -ph + ih - 1 : -1, :] += x[:, :, -ph + ih :, pw : -pw + iw].flip(
            dims=(2,)
        )
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph : -ph + ih, :pw].flip(dims=(3,))
        out[:, :, :, -pw + iw - 1 : -1] += x[:, :, ph : -ph + ih, -pw + iw :].flip(
            dims=(3,)
        )
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph + ih - 1 : -1, -pw + iw - 1 : -1] += x[
            :, :, -ph + ih :, -pw + iw :
        ].flip(dims=(2, 3))
        out[:, :, -ph + ih - 1 : -1, 1 : 1 + pw] += x[:, :, -ph + ih :, :pw].flip(
            dims=(2, 3)
        )
        out[:, :, 1 : 1 + ph, -pw + iw - 1 : -1] += x[:, :, :ph, -pw + iw :].flip(
            dims=(2, 3)
        )

    elif padding == "replicate":
        out = x[:, :, ph : -ph + ih, pw : -pw + iw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw : -pw + iw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph + ih :, pw : -pw + iw].sum(2)
        out[:, :, :, 0] += x[:, :, ph : -ph + ih, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph : -ph + ih, -pw + iw :].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph + ih :, -pw + iw :].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph + ih :, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw + iw :].sum(3).sum(2)

    elif padding == "constant":
        out = x[:, :, ph : -(ph - ih), pw : -(pw - iw)]

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

    return fft.irfft2(y_f * torch.conj(filter_f)).real


def filter_fft_2d(filter, img_size, real_fft=True):
    ph = int((filter.shape[2] - 1) / 2)
    pw = int((filter.shape[3] - 1) / 2)

    filt2 = torch.zeros(
        tuple(filter.shape[:2]) + tuple(img_size[-2:]), device=filter.device
    )

    filt2[..., : filter.shape[2], : filter.shape[3]] = filter
    filt2 = torch.roll(filt2, shifts=(-ph, -pw), dims=(2, 3))

    return fft.rfft2(filt2) if real_fft else fft.fft2(filt2)


def conv3d(x: Tensor, filter: Tensor, padding: str = "valid"):
    r"""
    A helper function to perform 3D convolution of images :math:``x`` and ``filter``.

    The transposed of this operation is :func:`deepinv.physics.functional.conv_transpose3d`
    """
    pass


def conv_transpose3d(y: Tensor, filter: Tensor, padding: str = "valid"):
    r"""
    A helper function to perform 3D transpose convolution.
    """
    pass


def conv3d_fft(
    x: Tensor, filter: Tensor, real_fft: bool = True, padding: str = "valid"
) -> torch.Tensor:
    r"""
    A helper function performing the 3d convolution of ``x`` and `filter` using FFT.

    The adjoint of this operation is :func:`deepinv.physics.functional.conv_transpose3d_fft`.

    If ``b = 1`` or ``c = 1``, this function applies the same filter for each channel.
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
