import pytest
import deepinv as dinv
from torch import randn

TRANSFORMS = [
    "shift",
    "rotate",
    "scale",
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
    return randn(1, 3, 64, 64)


@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transforms(transform_name, image):
    transform = choose_transform(transform_name)
    image_transformed = transform(image)
    assert image.shape == image_transformed.shape
