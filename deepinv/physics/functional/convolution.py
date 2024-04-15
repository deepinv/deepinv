import torch
import torch.nn.functional as F
from torch import Tensor
import torch.fft as fft


def conv2d(x: Tensor, filter: Tensor, padding: str = "valid") -> Tensor:
    r"""
    A helper function performing the 2d convolution of images `x` and `filter`. The adjoint of this operation is :meth:`deepinv.physics.functional.conv_transposed2d`

    :param torch.Tensor x: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param padding: (options = `valid`, `circular`, `replicate`, `reflect`) If `padding = 'valid'` the blurred output is smaller than the image (no padding), otherwise the blurred output has the same size as the image.

    :return: (torch.Tensor) : the output
    """
    assert x.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

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
        ph = (h - 1) // 2
        pw = (w - 1) // 2
        pad = (pw, pw, ph, ph)

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


def conv_transpose2d(y: Tensor, filter: Tensor, padding: str = "valid") -> Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of x and filter. The transposed of this operation is :meth:`deepinv.physics.functional.conv2d`

    :param torch.Tensor x: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` ) where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    :param str padding: options are ``'valid'``, ``'circular'``, ``'replicate'`` and ``'reflect'``.
        If ``padding='valid'`` the blurred output is smaller than the image (no padding)
        otherwise the blurred output has the same size as the image.

    :return: (torch.Tensor) : the output
    """

    assert y.dim() == filter.dim() == 4, "Input and filter must be 4D tensors"

    # Get dimensions of the input and the filter
    B, C, H, W = y.size()
    b, c, h, w = filter.size()

    ph = (h - 1) // 2
    pw = (w - 1) // 2

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
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, :ph, :] += x[:, :, -ph:, pw:-pw]
        out[:, :, -ph:, :] += x[:, :, :ph, pw:-pw]
        out[:, :, :, :pw] += x[:, :, ph:-ph, -pw:]
        out[:, :, :, -pw:] += x[:, :, ph:-ph, :pw]
        # corners
        out[:, :, :ph, :pw] += x[:, :, -ph:, -pw:]
        out[:, :, -ph:, -pw:] += x[:, :, :ph, :pw]
        out[:, :, :ph, -pw:] += x[:, :, -ph:, :pw]
        out[:, :, -ph:, :pw] += x[:, :, :ph, -pw:]

    elif padding == "reflect":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 1 : 1 + ph, :] += x[:, :, :ph, pw:-pw].flip(dims=(2,))
        out[:, :, -ph - 1 : -1, :] += x[:, :, -ph:, pw:-pw].flip(dims=(2,))
        out[:, :, :, 1 : 1 + pw] += x[:, :, ph:-ph, :pw].flip(dims=(3,))
        out[:, :, :, -pw - 1 : -1] += x[:, :, ph:-ph, -pw:].flip(dims=(3,))
        # corners
        out[:, :, 1 : 1 + ph, 1 : 1 + pw] += x[:, :, :ph, :pw].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, -pw - 1 : -1] += x[:, :, -ph:, -pw:].flip(dims=(2, 3))
        out[:, :, -ph - 1 : -1, 1 : 1 + pw] += x[:, :, -ph:, :pw].flip(dims=(2, 3))
        out[:, :, 1 : 1 + ph, -pw - 1 : -1] += x[:, :, :ph, -pw:].flip(dims=(2, 3))

    elif padding == "replicate":
        out = x[:, :, ph:-ph, pw:-pw]
        # sides
        out[:, :, 0, :] += x[:, :, :ph, pw:-pw].sum(2)
        out[:, :, -1, :] += x[:, :, -ph:, pw:-pw].sum(2)
        out[:, :, :, 0] += x[:, :, ph:-ph, :pw].sum(3)
        out[:, :, :, -1] += x[:, :, ph:-ph, -pw:].sum(3)
        # corners
        out[:, :, 0, 0] += x[:, :, :ph, :pw].sum(3).sum(2)
        out[:, :, -1, -1] += x[:, :, -ph:, -pw:].sum(3).sum(2)
        out[:, :, -1, 0] += x[:, :, -ph:, :pw].sum(3).sum(2)
        out[:, :, 0, -1] += x[:, :, :ph, -pw:].sum(3).sum(2)

    return out


def conv2d_fft(x: Tensor, filter: Tensor, real_fft: bool = True) -> Tensor:
    r"""
    A helper function performing the 2d convolution of images `x` and `filter` using FFT. The adjoint of this operation is :meth:`deepinv.physics.functional.conv_transposed2d_fft()`

    :param torch.Tensor x: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    For convolution using FFT consider only `circular` padding (i.e., circular convolution).

    :return torch.Tensor : the output of the convolution of the shape size as :math:`x`
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


def conv_transpose2d_fft(y: Tensor, filter: Tensor, real_fft: bool = True) -> Tensor:
    r"""
    A helper function performing the 2d transposed convolution 2d of `x` and `filter` using FFT. The adjoint of this operation is :meth:`deepinv.physics.functional.conv2d_fft()`.

    :param torch.Tensor y: Image of size `(B, C, W, H)`.
    :param torch.Tensor filter: Filter of size `(b, c, w, h)` ) where `b` can be either `1` or `B` and `c` can be either `1` or `C`.

    If `b = 1` or `c = 1`, then this function supports broadcasting as the same as `numpy <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_. Otherwise, each channel of each image is convolved with the corresponding kernel.

    For convolution using FFT consider only `circular` padding (i.e., circular convolution).

    :return torch.Tensor : the output of the convolution, which has the same shape as :math:`y`
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

    filt2 = torch.zeros(filter.shape[:2] + img_size[-2:], device=filter.device)

    filt2[..., : filter.shape[2], : filter.shape[3]] = filter
    filt2 = torch.roll(filt2, shifts=(-ph, -pw), dims=(2, 3))

    return fft.rfft2(filt2) if real_fft else fft.fft2(filt2)


def conv3d(x: Tensor, filter: Tensor, padding: str = "valid"):
    r"""
    A helper function to perform 3D convolution of images :math:`x` and `filter`.  The transposed of this operation is :meth:`deepinv.physics.functional.conv_transposed3d()`
    """
    pass


def conv_transpose3d(y: Tensor, filter: Tensor, padding: str = "valid"):
    r"""
    A helper function to perform 3D transpose convolution.
    """
    pass


# if __name__ == "__main__":
#     from skimage.data import astronaut
#     from skimage.transform import resize
#     import deepinv as dinv
#     from deepinv.physics.blur import gaussian_blur

#     B = 4
#     C = 3
#     H = 1024
#     W = 1024

#     img = resize(astronaut(), (H, W))

#     device = "cuda"
#     dtype = torch.float32

#     x = torch.from_numpy(img).permute(2, 0, 1)[None].to(device=device, dtype=dtype)
#     x = x.expand(B, -1, -1, -1)

#     filter = gaussian_blur(3.0).expand(B, C, -1, -1).to(device=device, dtype=dtype)
#     padding = "circular"
#     Ax = conv2d(x, filter.flip(-1).flip(-2), padding)
#     dinv.utils.plot(Ax[0])

#     y = torch.randn_like(Ax)
#     z = conv_transpose2d(y, filter.flip(-1).flip(-2), padding)
#     print((Ax * y).sum(dim=(1, 2, 3)) - (x * z).sum(dim=(1, 2, 3)))

#     Ax_fft = conv2d_fft(x, filter)
#     dinv.utils.plot(Ax_fft[0])

#     y_fft = torch.randn_like(Ax_fft)
#     z_fft = conv_transpose2d_fft(y_fft, filter)
#     print((Ax_fft * y_fft).sum(dim=(1, 2, 3)) - (x * z_fft).sum(dim=(1, 2, 3)))
#     print((Ax - Ax_fft).abs().sum())

#     # Benchmark
#     # from torch.utils.benchmark import Timer

#     # for kernel_size in range(33, H // 2 - 1, 10):
#     #     filter = torch.randn(
#     #         (B, C, kernel_size * 2 + 1, kernel_size * 2 + 1), device=device, dtype=dtype
#     #     )
#     #     print("Kernel size: ", kernel_size * 2 + 1)
#     #     conv_timer = Timer(
#     #         stmt="conv2d(x, filter, padding)",
#     #         globals=globals(),
#     #         num_threads=1,
#     #     )
#     #     print("Conv: ", conv_timer.blocked_autorange(min_run_time=10).median)
#     #     fft_timer = Timer(
#     #         stmt="conv2d_fft(x, filter)",
#     #         globals=globals(),
#     #         num_threads=1,
#     #     )
#     #     print("FFT: ", conv_timer.blocked_autorange(min_run_time=10).median)
