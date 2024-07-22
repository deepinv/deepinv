import pytest
import deepinv as dinv
import torch

TRANSFORMS = [
    "shift",
    "rotate",
    "scale",
    "shift+scale",
    "shift*scale",
    "scale+rotate",
    "scale*rotate",
    ""
    "BODMASshift+scale*rotate", # (shift+scale) * rotate
    "shift+scale*rotate",] # shift + (scale*rotate)
[
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

    if transform_name == "shift":
        return dinv.transform.Shift()
    elif transform_name == "rotate":
        return dinv.transform.Rotate()
    elif transform_name == "scale":
        return dinv.transform.Scale(factors=[0.75]) #limit to 0.75 only to avoid severe edge effects
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
    Checks white square is in same location and not in another location.
    """
    return torch.allclose(x[..., 55:65, 75:85], x_t[..., 55:65, 75:85], atol=1e-5) and \
        torch.allclose(x[..., 75:85, 55:65], x_t[..., 75:85, 55:65])

@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transforms(transform_name, image):
    transform = choose_transform(transform_name)
    image_t = transform(image)

    # Check if any constituent part of transform is a stacking
    if transform.__class__.__name__ == "StackTransform" or getattr(transform, "t1", transform).__class__.__name__ == "StackTransform" or getattr(transform, "t2", transform).__class__.__name__ == "StackTransform":
        assert image.shape[1:] == image_t.shape[1:]
        assert image.shape[0] * 2 == image_t.shape[0]
    else:
        assert image.shape == image_t.shape

@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transform_identity(transform_name, pattern):
    t = choose_transform(transform_name)
    print(t)
    assert check_correct_pattern(pattern, t.identity(pattern))
    assert check_correct_pattern(pattern, t.symmetrize(lambda x: x)(pattern))