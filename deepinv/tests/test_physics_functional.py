import pytest
import torch
import deepinv.physics.functional as dF
from functools import partial
import deepinv as dinv
from deepinv.utils import TensorList

# Some global constants
ALL_CONV_PADDING = ("valid", "circular", "zeros", "replicate", "reflect")


@pytest.mark.parametrize(
    "physics_class",
    [
        dinv.physics.Tomography,
        dinv.physics.TomographyWithAstra,
        dinv.physics.PET,
    ],
)
def test_tomography_subsets(physics_class, device):
    if physics_class is dinv.physics.PET:
        pytest.importorskip("parallelproj")
        physics = dinv.physics.PET(img_size=(8, 8), normalize=False, device=device)
        physics.background.fill_(3.0)
        physics.normalize = True
        physics.operator_norm.fill_(2.0)

        subset_physics = dinv.physics.split_physics(
            physics, num_subsets=2, device=device
        )
        x = torch.ones((1, 1, *physics.img_size), device=device)
        y = physics.A(x, add_background=True)
        y_subsets = dinv.physics.split_measurements(y, physics, num_subsets=2)

        assert all(torch.all(subset.background == 3.0) for subset in subset_physics)
        for subset_y, subset in zip(y_subsets, subset_physics, strict=True):
            assert torch.allclose(
                subset.A(x, add_background=True),
                subset_y,
            )
        return

    if physics_class is dinv.physics.TomographyWithAstra:
        pytest.importorskip("astra")
        if device.type != "cuda":
            pytest.skip("TomographyWithAstra requires CUDA")

    img_width = 16
    num_angles = 8
    num_subsets = 4

    if physics_class is dinv.physics.Tomography:
        physics = physics_class(
            img_width=img_width,
            angles=num_angles,
            device=device,
            circle=True,
            normalize=False,
            parallel_computation=False,
        )
        normalized_physics = physics_class(
            img_width=img_width,
            angles=num_angles,
            device=device,
            circle=True,
            normalize=True,
            parallel_computation=False,
        )
    else:
        physics = physics_class(
            img_size=(img_width, img_width),
            angles=num_angles,
            n_detector_pixels=img_width,
            device=device,
            normalize=False,
        )
        normalized_physics = physics_class(
            img_size=(img_width, img_width),
            angles=num_angles,
            n_detector_pixels=img_width,
            device=device,
            normalize=True,
        )

    x = torch.rand(
        (1, 1, img_width, img_width),
        generator=torch.Generator(device).manual_seed(0),
        device=device,
    )
    y = physics.A(x)

    angles = torch.arange(num_angles, device=device)
    geometry_vectors = torch.arange(num_angles * 12, device=device).reshape(
        num_angles, 12
    )

    angle_indices = dinv.physics.get_subset_tensor(angles, num_subsets)
    vector_indices = dinv.physics.get_subset_tensor(geometry_vectors, num_subsets)
    assert len(angle_indices) == num_subsets
    assert len(vector_indices) == num_subsets
    expected_indices = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]], device=device)
    assert torch.equal(torch.stack(angle_indices), expected_indices)
    assert torch.equal(torch.stack(vector_indices), expected_indices)

    y_subsets = dinv.physics.split_measurements(y, physics, num_subsets)
    subset_physics = dinv.physics.split_physics(physics, num_subsets, device=device)

    assert isinstance(y_subsets, TensorList)
    assert isinstance(subset_physics, dinv.physics.StackedLinearPhysics)
    assert len(y_subsets) == num_subsets
    assert len(subset_physics) == num_subsets

    stacked_y = subset_physics.A(x)
    for i, idx in enumerate(angle_indices):
        angle_dim = -1 if physics_class is dinv.physics.Tomography else -2
        expected_y = y.index_select(angle_dim, idx)
        expected_angles = physics.angles.index_select(0, idx.to(physics.angles.device))

        assert torch.allclose(y_subsets[i], expected_y)
        assert torch.allclose(stacked_y[i], expected_y, atol=1e-5)
        assert torch.allclose(subset_physics[i].angles, expected_angles)

    assert torch.allclose(
        subset_physics.A_adjoint(y_subsets), physics.A_adjoint(y), atol=1e-5
    )
    assert torch.allclose(
        subset_physics.compute_sqnorm(x, max_iter=20, verbose=False),
        physics.compute_sqnorm(x, max_iter=20, verbose=False),
        rtol=1e-5,
    )

    normalized_subset_physics = dinv.physics.split_physics(
        normalized_physics, num_subsets, device=device
    )
    normalized_y = normalized_physics.A(x)
    normalized_y_subsets = dinv.physics.split_measurements(
        normalized_y, normalized_physics, num_subsets
    )
    normalized_stacked_y = normalized_subset_physics.A(x)

    for i, idx in enumerate(angle_indices):
        expected_y = normalized_y.index_select(angle_dim, idx)
        assert torch.allclose(normalized_y_subsets[i], expected_y)
        assert torch.allclose(normalized_stacked_y[i], expected_y, atol=1e-5)

    assert all(
        subset.normalize
        and torch.equal(subset.operator_norm, normalized_physics.operator_norm)
        for subset in normalized_subset_physics
    )
    assert torch.allclose(
        normalized_subset_physics.A_adjoint(normalized_y_subsets),
        normalized_physics.A_adjoint(normalized_y),
        atol=1e-5,
    )


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


def test_imresize_div2k(load_example_image):
    x = load_example_image("div2k_valid_hr_0877.png") * 255.0
    y = load_example_image("div2k_valid_lr_bicubic_0877x4.png") * 255.0
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


# NOTE: This test is a non-regression test that checks that the new implementation
# of gaussian_blur in deepinv.physics.functional.gaussian_blur produces similar
# results to a first implementation using torchvision.transforms.functional.rotate
# on an axis-aligned Gaussian kernel.
@pytest.mark.parametrize("sigma", [(0.5, 1.0), (2.0, 2.0), (3.0, 4.0)])
@pytest.mark.parametrize("angle", [15.0, 45.0, 90.0])
def test_gaussian_blur_non_regression(device, sigma, angle):
    from torchvision.transforms.functional import rotate
    import torchvision

    def gaussian_blur(
        sigma: float | tuple[float, ...] = (1, 1),
        angle: float = 0,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:

        if isinstance(sigma, (int, float)):
            sigma = (sigma, sigma)

        s = max(sigma)
        c = int(s / 0.3 + 1)
        k_size = 2 * c + 1

        delta = torch.arange(k_size, device=device)

        x, y = torch.meshgrid(delta, delta, indexing="ij")
        x = x - c
        y = y - c
        filt = (x / sigma[0]).pow(2)
        filt += (y / sigma[1]).pow(2)
        filt = torch.exp(-filt / 2.0)

        filt = (
            rotate(
                filt.unsqueeze(0).unsqueeze(0),
                angle,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            )
            .squeeze(0)
            .squeeze(0)
        )

        filt = filt / filt.flatten().sum()

        return filt.unsqueeze(0).unsqueeze(0)

    ref_filter = gaussian_blur(sigma=sigma, angle=angle, device=device)
    new_filter = dF.gaussian_blur(sigma=sigma, angle=angle, device=device)
    assert torch.allclose(
        ref_filter, new_filter, rtol=1e-1, atol=2e-2
    ), f"Filters differ for sigma={sigma} and angle={angle}. Got old implementation:\n{ref_filter}\nNew implementation:\n{new_filter}"

    # Compute Normalized Cross-Correlation (NCC)
    numerator = torch.sum(
        (ref_filter - ref_filter.mean()) * (new_filter - new_filter.mean())
    )
    denominator = torch.sqrt(
        torch.sum((ref_filter - ref_filter.mean()) ** 2)
        * torch.sum((new_filter - new_filter.mean()) ** 2)
    )

    normalized_cross_correlation = (numerator / denominator).item()

    assert normalized_cross_correlation == pytest.approx(
        1.0, abs=5e-3
    ), f"NCC is {normalized_cross_correlation:.6f}, expected approximately 1.0"


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (2, 3, 10, 6)])
@pytest.mark.parametrize("padding", [(1, 1), (3, 2)])
def test_liu_jia_pad(shape, padding):
    torch.manual_seed(0)
    B, C, H, W = shape
    pad_h, pad_w = padding
    x = torch.rand(*shape)

    y = dF.liu_jia_pad(x, padding=padding)

    assert y.shape == (B, C, H + 2 * pad_h, W + 2 * pad_w)

    center = y[..., pad_h : pad_h + H, pad_w : pad_w + W]
    assert torch.equal(center, x)
