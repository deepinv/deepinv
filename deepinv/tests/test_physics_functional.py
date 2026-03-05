import pytest
import torch
import deepinv.physics.functional as dF
from functools import partial
import deepinv as dinv

# Some global constants
ALL_CONV_PADDING = ("valid", "circular", "zeros", "replicate", "reflect")
device = "cpu"


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("nchan_im,nchan_filt", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("padding", ALL_CONV_PADDING)
@pytest.mark.parametrize("real_fft", [True, False])
@pytest.mark.parametrize("use_fft", [False, True])
@pytest.mark.parametrize("correlation", [True, False])
@pytest.mark.parametrize("im_size_spatial", [(5, 5), (6, 6), (5, 6), (6, 5)])
@pytest.mark.parametrize("filt_size_spatial", [(3, 3), (4, 4), (3, 4), (4, 3)])
def test_conv2d_adjointness(
    device,
    B,
    nchan_im,
    nchan_filt,
    padding,
    real_fft,
    use_fft,
    correlation,
    im_size_spatial,
    filt_size_spatial,
):
    torch.manual_seed(0)

    sim = [nchan_im, *im_size_spatial]
    sfil = [nchan_filt, *filt_size_spatial]

    if use_fft:
        conv2d_fn = partial(dF.conv2d_fft, real_fft=real_fft)
        conv_transpose2d_fn = partial(dF.conv_transpose2d_fft, real_fft=real_fft)
    else:
        conv2d_fn = partial(dF.conv2d, correlation=correlation)
        conv_transpose2d_fn = partial(dF.conv_transpose2d, correlation=correlation)

    for bf in set((1, B)):
        x = torch.rand((B, *sim), device=device)
        h = torch.rand((bf, *sfil), device=device)
        h = h / h.sum(
            dim=(-1, -2), keepdim=True
        )  # normalize filter to avoid numerical issues

        Ax = conv2d_fn(x, h, padding=padding)
        y = torch.rand_like(Ax)
        Aty = conv_transpose2d_fn(y, h, padding=padding)

        lhs = torch.sum(Ax * y)
        rhs = torch.sum(Aty * x)
        assert torch.abs(lhs - rhs) < 1e-4 * max(
            torch.abs(lhs), torch.abs(rhs)
        )  # relative tolerance


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("nchan_im,nchan_filt", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("padding", ALL_CONV_PADDING)
@pytest.mark.parametrize("transposed", [True, False])
@pytest.mark.parametrize("im_size_spatial", [(5, 5), (6, 6), (5, 6), (6, 5)])
@pytest.mark.parametrize("filt_size_spatial", [(3, 3), (4, 4), (3, 4), (4, 3)])
def test_conv2d_spatial_and_fft_equivalence(
    device,
    B,
    nchan_im,
    nchan_filt,
    padding,
    transposed,
    im_size_spatial,
    filt_size_spatial,
):
    torch.manual_seed(0)

    sim = [nchan_im, *im_size_spatial]
    sfil = [nchan_filt, *filt_size_spatial]

    if transposed:
        spatial_fn = dF.conv_transpose2d
        fft_fn = partial(dF.conv_transpose2d_fft, real_fft=True)  # Only test real_fft
    else:
        spatial_fn = dF.conv2d
        fft_fn = partial(dF.conv2d_fft, real_fft=True)
    for bf in (1, B):
        x = torch.rand((B, *sim), device=device)
        h = torch.rand((bf, *sfil), device=device)
        h = h / h.sum(
            dim=(-1, -2), keepdim=True
        )  # normalize filter to avoid numerical issues

        spatial_output = spatial_fn(x, h, padding=padding)
        fft_output = fft_fn(x, h, padding=padding)

        assert spatial_output.shape == fft_output.shape
        assert torch.allclose(spatial_output, fft_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("nchan_im,nchan_filt", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("padding", ALL_CONV_PADDING)  # safe set
@pytest.mark.parametrize("real_fft", [True, False])
@pytest.mark.parametrize("use_fft", [True, False])
@pytest.mark.parametrize(
    "im_size_spatial", [(5, 5, 5), (6, 6, 6), (5, 5, 6), (5, 6, 5)]
)
@pytest.mark.parametrize(
    "filt_size_spatial", [(3, 3, 3), (4, 4, 4), (4, 3, 4), (3, 4, 3)]
)
def test_conv3d_adjointness(
    device,
    B,
    nchan_im,
    nchan_filt,
    padding,
    real_fft,
    use_fft,
    im_size_spatial,
    filt_size_spatial,
):
    torch.manual_seed(0)

    sim = [nchan_im, *im_size_spatial]
    sfil = [nchan_filt, *filt_size_spatial]

    if use_fft:
        conv3d_fn = partial(dF.conv3d_fft, real_fft=real_fft)
        conv_transpose3d_fn = partial(dF.conv_transpose3d_fft, real_fft=real_fft)
    else:
        conv3d_fn = dF.conv3d
        conv_transpose3d_fn = dF.conv_transpose3d

    for bf in set((1, B)):
        x = torch.rand((B, *sim), device=device, dtype=torch.float64)
        h = torch.rand((bf, *sfil), device=device, dtype=torch.float64)
        h = h / h.sum(
            dim=(-1, -2, -3), keepdim=True
        )  # normalize filter to avoid numerical issues

        Ax = conv3d_fn(x, h, padding=padding)
        y = torch.rand_like(Ax)
        Aty = conv_transpose3d_fn(y, h, padding=padding)

        lhs = torch.sum(Ax * y)
        rhs = torch.sum(Aty * x)
        assert torch.abs(lhs - rhs) < 1e-3 * max(
            torch.abs(lhs), torch.abs(rhs)
        )  # relative tolerance


@pytest.mark.parametrize("nchan_im,nchan_filt", [(1, 1), (3, 1)])
@pytest.mark.parametrize("padding", ("circular",))  # safe set
@pytest.mark.parametrize(
    "im_size_spatial", [(5, 5, 5), (6, 6, 6), (5, 5, 6), (5, 6, 5)]
)
@pytest.mark.parametrize(
    "filt_size_spatial", [(3, 3, 3), (4, 4, 4), (4, 3, 4), (3, 4, 3)]
)
def test_conv3d_norm(
    device, nchan_im, nchan_filt, padding, im_size_spatial, filt_size_spatial
):
    torch.manual_seed(0)
    max_iter = 1000
    tol = 1e-6
    # Note : does not work for nchan_im, nchan_filt = (3, 3)

    sim = [nchan_im, *im_size_spatial]
    sfil = [nchan_filt, *filt_size_spatial]

    x = torch.randn(sim)[None].to(device)
    x /= torch.linalg.vector_norm(x)
    h = torch.rand(sfil)[None].to(device)
    h /= h.sum()

    zold = torch.zeros_like(x)
    for it in range(max_iter):
        y = dF.conv3d_fft(x, h, padding=padding)
        y = dF.conv_transpose3d_fft(y, h, padding=padding)
        z = (
            torch.matmul(x.conj().reshape(-1), y.reshape(-1))
            / torch.linalg.vector_norm(x) ** 2
        )

        rel_var = torch.linalg.vector_norm(z - zold)
        if rel_var < tol:
            break
        zold = z
        x = y / torch.linalg.vector_norm(y)

    assert torch.abs(zold.item() - torch.ones(1)) < 1e-2


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("nchan_im,nchan_filt", [(1, 1), (3, 1), (3, 3)])
@pytest.mark.parametrize("padding", ALL_CONV_PADDING)
@pytest.mark.parametrize("transposed", [False, True])  # test conv3d or conv_transpose3d
@pytest.mark.parametrize(
    "im_size_spatial", [(5, 5, 5), (6, 6, 6), (5, 5, 6), (5, 6, 5)]
)
@pytest.mark.parametrize(
    "filt_size_spatial", [(3, 3, 3), (4, 4, 4), (4, 3, 4), (3, 4, 3)]
)
def test_conv3d_spatial_and_fft_equivalence(
    device,
    B,
    nchan_im,
    nchan_filt,
    padding,
    transposed,
    im_size_spatial,
    filt_size_spatial,
):
    torch.manual_seed(0)

    sim = [nchan_im, *im_size_spatial]
    sfil = [nchan_filt, *filt_size_spatial]

    if transposed:
        spatial_fn = dF.conv_transpose3d
        fft_fn = partial(dF.conv_transpose3d_fft, real_fft=True)  # Only test real_fft
    else:
        spatial_fn = dF.conv3d
        fft_fn = partial(dF.conv3d_fft, real_fft=True)

    for bf in set([1, B]):
        x = torch.rand((B, *sim), device=device)
        h = torch.rand((bf, *sfil), device=device)
        h = h / h.sum(
            dim=(-1, -2, -3), keepdim=True
        )  # normalize filter to avoid numerical issues

        spatial_output = spatial_fn(x, h, padding=padding)
        fft_output = fft_fn(x, h, padding=padding)

        assert spatial_output.shape == fft_output.shape
        assert torch.allclose(spatial_output, fft_output, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("kernel", ["cubic", "gaussian"])
@pytest.mark.parametrize("scale", [2, 0.5])
@pytest.mark.parametrize("antialiasing", [True, False])
def test_imresize(kernel, scale, antialiasing):
    sigma = 2
    img_size = (1, 64, 64)
    x = torch.randn(1, *img_size)
    y = dinv.physics.functional.imresize_matlab(
        x,
        scale=scale,
        kernel=kernel,
        sigma=sigma,
        padding_type="reflect",
        antialiasing=antialiasing,
    )
    assert y.shape == (
        1,
        img_size[0],
        int(img_size[1] * scale),
        int(img_size[2] * scale),
    )


def test_imresize_div2k():
    x = dinv.utils.load_example("div2k_valid_hr_0877.png") * 255.0
    y = dinv.utils.load_example("div2k_valid_lr_bicubic_0877x4.png") * 255.0
    y2 = dinv.physics.functional.imresize_matlab(x, scale=1 / 4).round()
    assert dinv.metric.PSNR()(y2 / 255.0, y / 255.0) > 59


def test_dct_idct(device):
    shape = (1, 1, 8, 8)
    x = torch.ones(shape).to(device)
    y = dinv.physics.functional.dct_2d(x)
    xrec = dinv.physics.functional.idct_2d(y)
    assert torch.linalg.vector_norm(x - xrec) < 1e-5

    y = dinv.physics.functional.dct_2d(x, norm="ortho")
    xrec = dinv.physics.functional.idct_2d(y, norm="ortho")
    assert torch.linalg.vector_norm(x - xrec) < 1e-5
