import pytest
from torch import allclose
from deepinv.physics.generator import (
    GaussianMaskGenerator,
    UniformMaskGenerator,
    RandomMaskGenerator,
)

C, T, H, W = 2, 12, 256, 512

GENERATORS = ["gaussian", "random", "uniform"]
IMG_SIZES = [(H, W), (C, H, W), (C, T, H, W), (64, 64)]
ACCELERATIONS = [4, 10, 12]
CENTER_FRACTIONS = [0, 0.04, 24 / 512]


@pytest.fixture
def batch_size():
    return 2


def choose_generator(generator_name, img_size, acc, center_fraction):
    if generator_name == "gaussian":
        g = GaussianMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "random":
        g = RandomMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "uniform":
        g = UniformMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    return g


@pytest.mark.parametrize("generator_name", GENERATORS)
@pytest.mark.parametrize("img_size", IMG_SIZES)
@pytest.mark.parametrize("acc", ACCELERATIONS)
@pytest.mark.parametrize("center_fraction", CENTER_FRACTIONS)
def test_generator(generator_name, img_size, batch_size, acc, center_fraction):
    generator = choose_generator(generator_name, img_size, acc, center_fraction)
    # test across different accs and centre fracations
    H, W = img_size[-2:]
    assert W // generator.acc == (generator.n_lines + generator.n_center)

    mask = generator.step(batch_size=batch_size)["mask"]

    if len(img_size) == 2:
        assert len(mask.shape) == 4
        C = 1
    elif len(img_size) == 3:
        assert len(mask.shape) == 4
        C = img_size[0]
    elif len(img_size) == 4:
        assert len(mask.shape) == 5
        C = img_size[0]
        assert mask.shape[2] == img_size[1]

    assert mask.shape[0] == batch_size
    assert mask.shape[1] == C
    assert mask.shape[-2:] == img_size[-2:]

    for b in range(batch_size):
        for c in range(C):
            if len(img_size) == 4:
                for t in range(img_size[1]):
                    mask[b, c, t, :, :].sum() * generator.acc == H * W
            else:
                mask[b, c, :, :].sum() * generator.acc == H * W

    mask2 = generator.step(batch_size=batch_size)["mask"]

    if generator.n_lines != 0 and generator_name != "uniform":
        assert not allclose(mask, mask2)
