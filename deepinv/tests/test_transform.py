import pytest
import deepinv as dinv
import torch

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
    "BODMASshift+scale*rotate",  # (shift+scale) * rotate
    "BODMASshift*scale|rotate",  # shift * (scale|rotate)
    "shift+scale*rotate",  # shift + (scale*rotate)
    "shift+scale|rotate",  # shift + (scale|rotate)
    "shift*scale|rotate",  # (shift*scale) | rotate # NOTE no way here to do (shift+scale) | rotate
    "homography",
    "euclidean",
    "similarity",
    "affine",
    "pantiltrotate",
]


def choose_transform(transform_name):

    if "BODMAS" in transform_name:
        transform_name = transform_name[6:]
        if "*" in transform_name:
            names = transform_name.split("*")
            return choose_transform(names[0]) * choose_transform(names[1])

    if "+" in transform_name:
        names = transform_name.split("+")
        return choose_transform(names[0]) + choose_transform(names[1])

    if "|" in transform_name:
        names = transform_name.split("|")
        return choose_transform(names[0]) | choose_transform(names[1])

    if "BODMAS" not in transform_name:
        if "*" in transform_name:
            names = transform_name.split("*")
            return choose_transform(names[0]) * choose_transform(names[1])

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
        }

    if transform_name == "shift":
        return dinv.transform.Shift()
    elif transform_name == "rotate":
        return dinv.transform.Rotate()
    elif transform_name == "rotate3":
        return dinv.transform.Rotate(n_trans=3)
    elif transform_name == "reflect":
        return dinv.transform.Reflect(dim=[-2, -1])
    elif transform_name == "scale":
        # Limit to 0.75 only to avoid severe edge/interp effects
        return dinv.transform.Scale(factors=[0.75])
    elif transform_name == "scale3":
        return dinv.transform.Scale(factors=[0.75], n_trans=3)
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
    else:
        raise ValueError("Invalid transform_name provided")


@pytest.fixture
def image():
    # Random image
    return torch.randn(1, 3, 64, 64)


@pytest.fixture
def pattern_offset():
    return 45, 65


@pytest.fixture
def pattern(pattern_offset):
    # Fixed binary image of small white square
    x = torch.zeros(1, 3, 256, 256)
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
def test_transforms(transform_name, image):
    transform = choose_transform(transform_name)
    image_t = transform(image)

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
def test_transform_identity(transform_name, pattern, pattern_offset):
    t = choose_transform(transform_name)
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
    transform = dinv.transform.Rotate(multiples=90, n_trans=3) * dinv.transform.Reflect(
        dim=[-1], n_trans=2
    )
    x = torch.randn(batch_size, 2, 16, 16)
    xt = transform.identity(x, average=True)
    assert torch.allclose(x, xt)
