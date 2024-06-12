import pytest
import deepinv as dinv
from torch import randn

TRANSFORMS = [
    "shift",
    "rotate",
    "scale",
]


def choose_transform(transform_name):
    if transform_name == "shift":
        return dinv.transform.Shift()
    elif transform_name == "rotate":
        return dinv.transform.Rotate()
    elif transform_name == "scale":
        return dinv.transform.Scale()
    else:
        raise ValueError("Invalid transform_name provided")


@pytest.fixture
def image():
    return randn(1, 3, 64, 64)


@pytest.mark.parametrize("transform_name", TRANSFORMS)
def test_transform_arithmetic(transform_name, image):
    transform = choose_transform(transform_name)

    t1 = transform + dinv.transform.Shift()
    image_t = t1(image)
    assert image_t.shape[1:] == image.shape[1:]
    assert image_t.shape[0] == image.shape[0] * 2

    t2 = transform * dinv.transform.Shift()
    assert t2(image).shape == image.shape
