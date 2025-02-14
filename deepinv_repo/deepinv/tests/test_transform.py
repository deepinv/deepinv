import pytest
import deepinv as dinv
import torch

ADD_TIME_DIM = [True, False]

"""
We test many combinations of transforms via transform arithmetic.
For all transforms, we test correct shapes and also that the forward then inverse = the identity.
All new transforms must be tested individually.
Basic arithmetic (* for composition, + for stack, and | for random OR) are tested, as well as more complicated arithmetic.
Transforms prepended with `VARIANT` means changing the operator precedence.
Caveat: certain orderings of complicated arithmetic cannot be achieved with the testing code so will have to be manually coded in if desired in future.
"""
TRANSFORMS = [
    "shift",
    "rotate",
    "scale",
    "reflect",
    "shift+scale",
    "shift*scale",
    "scale+rotate",
    "scale*rotate",
    "scale3*rotate3",
    "scale|shift",
    "rotate|scale",
    "VARIANTshift+scale*rotate",  # (shift+scale) * rotate
    "VARIANTshift*scale|rotate",  # shift * (scale|rotate)
    "shift+scale*rotate",  # shift + (scale*rotate)
    "shift+scale|rotate",  # shift + (scale|rotate)
    "shift*scale|rotate",  # (shift*scale) | rotate # NOTE no way here to do (shift+scale) | rotate
    "homography",
    "euclidean",
    "similarity",
    "affine",
    "pantiltrotate",
    "diffeomorphism",
]


def choose_transform(transform_name, device, rng):

    if "VARIANT" in transform_name:
        transform_name = transform_name[7:]
        if "*" in transform_name:
            names = transform_name.split("*")
            return choose_transform(
                names[0], device=device, rng=rng
            ) * choose_transform(names[1], device=device, rng=rng)

    if "+" in transform_name:
        names = transform_name.split("+")
        return choose_transform(names[0], device=device, rng=rng) + choose_transform(
            names[1], device=device, rng=rng
        )

    if "|" in transform_name:
        names = transform_name.split("|")
        return choose_transform(names[0], device=device, rng=rng) | choose_transform(
            names[1], device=device, rng=rng
        )

    if "VARIANT" not in transform_name:
        if "*" in transform_name:
            names = transform_name.split("*")
            return choose_transform(
                names[0], device=device, rng=rng
            ) * choose_transform(names[1], device=device, rng=rng)

    if transform_name == "diffeomorphism":
        pytest.importorskip(
            "libcpab",
            reason="This test requires libcpab. Install with `pip install libcpab`.",
        )

    if transform_name in (
        "homography",
        "euclidean",
        "similarity",
        "affine",
        "pantiltrotate",
    ):
        pytest.importorskip(
            "kornia",
            reason="This test requires kornia. It should be "
            "installed with `pip install kornia`",
        )
        proj_kwargs = {
            "theta_max": 5,
            "theta_z_max": 20,
            "zoom_factor_min": 0.85,
            "shift_max": 0.2,
            "skew_max": 5,
            "x_stretch_factor_min": 0.85,
            "y_stretch_factor_min": 0.85,
            "padding": "zeros",
            "interpolation": "bicubic",
            "device": device,
            "rng": rng,
        }

    if transform_name == "shift":
        return dinv.transform.Shift(rng=rng)
    elif transform_name == "rotate":
        return dinv.transform.Rotate(rng=rng)
    elif transform_name == "rotate3":
        return dinv.transform.Rotate(n_trans=3, rng=rng)
    elif transform_name == "reflect":
        return dinv.transform.Reflect(dim=[-2, -1], rng=rng)
    elif transform_name == "scale":
        # Limit to 0.75 only to avoid severe edge/interp effects
        return dinv.transform.Scale(factors=[0.75], rng=rng)
    elif transform_name == "scale3":
        return dinv.transform.Scale(factors=[0.75], n_trans=3, rng=rng)
    elif transform_name == "homography":
        # Limit to avoid severe edge/interp effects. All the subgroups will zero their appropriate params.
        return dinv.transform.projective.Homography(**proj_kwargs)
    elif transform_name == "euclidean":
        return dinv.transform.projective.Euclidean(**proj_kwargs)
    elif transform_name == "similarity":
        return dinv.transform.projective.Similarity(**proj_kwargs)
    elif transform_name == "affine":
        return dinv.transform.projective.Affine(**proj_kwargs)
    elif transform_name == "pantiltrotate":
        return dinv.transform.projective.PanTiltRotate(**proj_kwargs)
    elif transform_name == "diffeomorphism":
        return dinv.transform.CPABDiffeomorphism(device=device)  # doesn't support rng
    else:
        raise ValueError("Invalid transform_name provided")


@pytest.fixture
def image(device):
    # Random image
    return torch.randn(1, 3, 64, 64).to(device)


@pytest.fixture
def pattern_offset():
    return 45, 65


@pytest.fixture
def pattern(pattern_offset, device):
    # Fixed binary image of small white square
    x = torch.zeros(1, 3, 256, 256, device=device)
    h, w = pattern_offset
    x[..., h : h + 30, w : w + 30] = 1
    return x


def check_correct_pattern(x, x_t, pattern_offset):
    """Check transformed image is same as original.
    Removes border effects on the small white square, caused by interpolation effects during transformation.
    Checks white square is in same location and not in another location.
    """
    h, w = pattern_offset
    H, W = x.shape[-2:]
    return torch.allclose(
        x[..., h + 10 : h + 20, w + 10 : w + 20],
        x_t[..., h + 10 : h + 20, w + 10 : w + 20],
        atol=1e-5,
    ) and torch.allclose(
        x[..., H - h - 20 : H - h - 10, W - w - 20 : W - w - 10],
        x_t[..., H - h - 20 : H - h - 10, W - w - 20 : W - w - 10],
    )


@pytest.mark.parametrize("transform_name", TRANSFORMS)
@pytest.mark.parametrize("add_time_dim", ADD_TIME_DIM)
def test_transforms(transform_name, image, add_time_dim: bool, device, rng):
    transform = choose_transform(transform_name, device=device, rng=rng)
    if add_time_dim:
        image = torch.stack((image, image), dim=2)
    image_t = transform(image)

    assert image.device == image_t.device == device

    # Check if any constituent part of transform is a stacking
    if "+" in transform_name:
        assert image.shape[1:] == image_t.shape[1:]
        assert image.shape[0] * 2 == image_t.shape[0]
    elif "3" in transform_name:
        assert image.shape[1:] == image_t.shape[1:]
        assert image.shape[0] * 9 == image_t.shape[0]
    else:
        assert image.shape == image_t.shape


@pytest.mark.parametrize("transform_name", TRANSFORMS)
@pytest.mark.parametrize("add_time_dim", ADD_TIME_DIM)
def test_transform_identity(
    transform_name, pattern, pattern_offset, add_time_dim: bool, device, rng
):
    if add_time_dim:
        pattern = torch.stack((pattern, pattern), dim=2)

    if device.type != "cpu" and transform_name in (
        "homography",
        "euclidean",
        "similarity",
        "affine",
        "pantiltrotate",
    ):
        # more reliable with a cpu rng here
        rng = torch.Generator().manual_seed(0)

    t = choose_transform(transform_name, device=device, rng=rng)
    assert check_correct_pattern(pattern, t.identity(pattern), pattern_offset)
    assert check_correct_pattern(
        pattern, t.symmetrize(lambda x: x)(pattern), pattern_offset
    )


def test_rotate_90():
    # Test if rotate with theta=90 results in exact pixel rotation
    x = torch.randn(1, 2, 16, 16)
    transform = dinv.transform.Rotate()
    y1 = transform.transform(x, theta=[90.0])
    y2 = torch.rot90(x, dims=[-2, -1])
    assert torch.all(y1 == y2)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_size(batch_size):
    # Test batch retains correct order when >1 n_trans
    transform = dinv.transform.Rotate(multiples=90, n_trans=3) * dinv.transform.Reflect(
        dim=[-1], n_trans=2
    )
    x = torch.randn(batch_size, 2, 16, 16)
    xt = transform.identity(x, average=True)
    assert torch.allclose(x, xt)

    # Test still works when collate_batch is False
    xt = transform.symmetrize(lambda x: x, average=True, collate_batch=False)(x)
    assert torch.allclose(x, xt)


def test_shift_time():
    # Video with moving line
    x = torch.zeros(1, 3, 8, 16, 16)  # B,C,T,H,W
    for i in range(8):
        x[:, :, i, i * 2, :] = 1

    t1 = dinv.transform.ShiftTime(n_trans=1, padding="wrap")
    t2 = dinv.transform.Reflect(dim=[-1])

    assert torch.allclose(t1.identity(x), x)
    assert torch.allclose((t1 * t2).identity(x), x)
