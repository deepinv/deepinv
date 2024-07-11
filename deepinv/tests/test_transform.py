import pytest
import deepinv as dinv
import torch

TRANSFORMS = [
    "shift",
    "rotate",
    "scale",]
#TODO add shift+scale, shift*scale, rotate*scale etc. and handle in choose()
[
    "homography",
    "euclidean",
    "similarity",
    "affine",
    "pantiltrotate",
]


def choose_transform(transform_name):
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

    if transform_name == "shift":
        return dinv.transform.Shift()
    elif transform_name == "rotate":
        return dinv.transform.Rotate()
    elif transform_name == "scale":
        return dinv.transform.Scale()
    elif transform_name == "homography":
        return dinv.transform.projective.Homography()
    elif transform_name == "euclidean":
        return dinv.transform.projective.Euclidean()
    elif transform_name == "similarity":
        return dinv.transform.projective.Similarity()
    elif transform_name == "affine":
        return dinv.transform.projective.Affine()
    elif transform_name == "pantiltrotate":
        return dinv.transform.projective.PanTiltRotate()
    else:
        raise ValueError("Invalid transform_name provided")


@pytest.fixture
def image():
    # Random image
    return torch.randn(1, 3, 64, 64)

@pytest.fixture
def pattern():
    # Fixed binary image of small white square
    x = torch.zeros(1, 3, 256, 256)
    x[..., 50:70, 70:90] = 1
    return x

def check_correct_pattern(x, x_t):
    """Check transformed image is same as original.
    Removes border effects on the small white square, caused by interpolation effects during transformation.
    """
    return torch.allclose(x[..., 55:65, 75:85], x_t[..., 55:65, 75:85])

@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transforms(transform_name, image):
    transform = choose_transform(transform_name)
    image_transformed = transform(image)
    assert image.shape == image_transformed.shape


@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transform_arithmetic(transform_name, image):
    transform = choose_transform(transform_name)

    t1 = transform + dinv.transform.Shift()
    image_t = t1(image)
    assert image_t.shape[1:] == image.shape[1:]
    assert image_t.shape[0] == image.shape[0] * 2

    t2 = transform * dinv.transform.Shift()
    assert t2(image).shape == image.shape

@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transform_identity(transform_name, pattern):
    t0 = choose_transform(transform_name)

    t1 = t0 + dinv.transform.Shift()
    t2 = t0 * dinv.transform.Shift()

    for t in (t0, t1, t2):
        print(t)
        assert check_correct_pattern(pattern, t2.identity(pattern))
        assert check_correct_pattern(pattern, t.symmetrize(lambda x: x)(pattern))