import pytest
import numpy as np
import torch
import deepinv as dinv
import itertools

from deepinv.physics.generator import (
    GaussianMaskGenerator,
    EquispacedMaskGenerator,
    RandomMaskGenerator,
)

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
SIZES = [(5, 5), (6, 6)]
NUM_CHANNELS = [1, 3]


# MRI Generators
C, T, H, W = 2, 12, 256, 512
MRI_GENERATORS = ["gaussian", "random", "uniform"]
MRI_IMG_SIZES = [(H, W), (C, H, W), (C, T, H, W), (64, 64)]
MRI_ACCELERATIONS = [4, 10, 12]
MRI_CENTER_FRACTIONS = [0, 0.04, 24 / 512]

# Inpainting/Splitting Generators
INPAINTING_IMG_SIZES = [
    (2, 64, 40),
    (2, 1000),
    (2, 3, 64, 40),
]  # (C,H,W), (C,M), (C,T,H,W)
INPAINTING_GENERATORS = ["bernoulli", "gaussian"]


def find_generator(name, size, num_channels, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda:0
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "MotionBlurGenerator":
        g = dinv.physics.generator.MotionBlurGenerator(
            psf_size=size, num_channels=num_channels, device=device
        )
        keys = ["filter"]
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            num_channels=num_channels,
        )
        keys = ["filter", "coeff", "pupil"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device)
        keys = ["sigma"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
def test_shape(name, size, num_channels, device):
    r"""
    Tests generators shape.
    """

    generator, size, keys = find_generator(name, size, num_channels, device)
    batch_size = 4

    params = generator.step(batch_size=batch_size)

    assert list(params.keys()) == keys

    if "filter" in params.keys():
        assert params["filter"].shape == (batch_size, num_channels, size[0], size[1])


@pytest.mark.parametrize("name", GENERATORS)
def test_generation_newparams(name, device):
    r"""
    Tests generators shape.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if name == "MotionBlurGenerator":
        param_key = "filter"
    elif name == "DiffractionBlurGenerator":
        param_key = "filter"
    elif name == "SigmaGenerator":
        param_key = "sigma"

    params0 = generator.step(batch_size=batch_size)
    params1 = generator.step(batch_size=batch_size)

    assert torch.any(params0[param_key] != params1[param_key])


@pytest.mark.parametrize("name", GENERATORS)
def test_generation(name, device):
    r"""
    Tests generators shape.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, 1, device)
    batch_size = 1
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    params = generator.step(batch_size=batch_size)

    if name == "MotionBlurGenerator":
        w = params["filter"]
        if device.type == "cpu":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.1509433985,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.3081761003,
                                0.1572327018,
                                0.3836477995,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                        ]
                    ]
                ]
            )
        elif device.type == "cuda":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0691823885,
                                0.0628930852,
                                0.0000000000,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0503144637,
                                0.4842767417,
                                0.0943396240,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.1069182381,
                                0.1320754737,
                                0.0000000000,
                            ],
                            [
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                                0.0000000000,
                            ],
                        ]
                    ]
                ],
            ).to(device)

    elif name == "DiffractionBlurGenerator":
        w = params["filter"]
        if device.type == "cpu":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0081667975,
                                0.0339039154,
                                0.0463643819,
                                0.0238370951,
                                0.0067134043,
                            ],
                            [
                                0.0235104840,
                                0.0769919083,
                                0.1068268567,
                                0.0638824701,
                                0.0154726375,
                            ],
                            [
                                0.0315882340,
                                0.0922789276,
                                0.1303794235,
                                0.0824520364,
                                0.0192645062,
                            ],
                            [
                                0.0210407600,
                                0.0526346825,
                                0.0764168128,
                                0.0469734631,
                                0.0075026853,
                            ],
                            [
                                0.0074746846,
                                0.0069856029,
                                0.0118351672,
                                0.0068998244,
                                0.0006031910,
                            ],
                        ]
                    ]
                ]
            )
        elif device.type == "cuda":
            wref = torch.tensor(
                [
                    [
                        [
                            [
                                0.0032691115,
                                0.0060402630,
                                0.0175693501,
                                0.0059448336,
                                0.0007023035,
                            ],
                            [
                                0.0095862420,
                                0.0427204743,
                                0.0728377998,
                                0.0452912413,
                                0.0118838884,
                            ],
                            [
                                0.0251656137,
                                0.0764198229,
                                0.1213025227,
                                0.0953904763,
                                0.0436460413,
                            ],
                            [
                                0.0162273813,
                                0.0507672094,
                                0.0916804373,
                                0.0891042799,
                                0.0563216321,
                            ],
                            [
                                0.0018812985,
                                0.0099059129,
                                0.0294828881,
                                0.0403944701,
                                0.0364645906,
                            ],
                        ]
                    ]
                ]
            ).to(device)

    elif name == "SigmaGenerator":
        w = params["sigma"]
        if device.type == "cpu":
            wref = torch.tensor([0.2531657219])
        elif device.type == "cuda":
            wref = torch.tensor([0.2055327892]).to(device)

    assert torch.allclose(w, wref, atol=1e-6)


######################
### MRI GENERATORS ###
######################


@pytest.fixture
def batch_size():
    return 2


def choose_mri_generator(generator_name, img_size, acc, center_fraction):
    if generator_name == "gaussian":
        g = GaussianMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "random":
        g = RandomMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    elif generator_name == "uniform":
        g = EquispacedMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction
        )
    return g


@pytest.mark.parametrize("generator_name", MRI_GENERATORS)
@pytest.mark.parametrize("img_size", MRI_IMG_SIZES)
@pytest.mark.parametrize("acc", MRI_ACCELERATIONS)
@pytest.mark.parametrize("center_fraction", MRI_CENTER_FRACTIONS)
def test_mri_generator(generator_name, img_size, batch_size, acc, center_fraction):
    generator = choose_mri_generator(generator_name, img_size, acc, center_fraction)
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
        assert not torch.allclose(mask, mask2)


#############################
### INPAINTING GENERATORS ###
#############################


def choose_inpainting_generator(name, img_size, split_ratio, pixelwise, device):
    if name == "bernoulli":
        return dinv.physics.generator.BernoulliSplittingMaskGenerator(
            tensor_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=torch.Generator().manual_seed(0),
        )
    elif name == "gaussian":
        return dinv.physics.generator.GaussianSplittingMaskGenerator(
            tensor_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=np.random.default_rng(0),
        )
    else:
        raise Exception("The generator chosen doesn't exist")


@pytest.mark.parametrize("generator_name", INPAINTING_GENERATORS)
@pytest.mark.parametrize("img_size", INPAINTING_IMG_SIZES)
@pytest.mark.parametrize("pixelwise", (False, True))
@pytest.mark.parametrize("split_ratio", (0.5,))
def test_inpainting_generators(
    generator_name, batch_size, img_size, pixelwise, split_ratio, device
):
    if generator_name == "gaussian" and len(img_size) < 3:
        pytest.skip(
            "Gaussian splitting mask not valid for images of shape smaller than (C, H, W)"
        )

    gen = choose_inpainting_generator(
        generator_name, img_size, split_ratio, pixelwise, device
    )  # Assume generator always receives "correct" img_size i.e. not one with dims missing

    def correct_ratio(ratio):
        assert torch.isclose(
            ratio,
            torch.Tensor([split_ratio]),
            rtol=1e-2,
            atol=1e-2,
        )

    def correct_pixelwise(mask):
        if pixelwise:
            assert torch.all(mask[:, 0, ...] == mask[:, 1, ...])
        else:
            assert not torch.all(mask[:, 0, ...] == mask[:, 1, ...])

    # Standard generate mask
    mask1 = gen.step(batch_size=batch_size)["mask"]
    correct_ratio(mask1.sum() / np.prod((batch_size, *img_size)))
    correct_pixelwise(mask1)

    # Standard without batch dim
    mask1 = gen.step(batch_size=None)["mask"]
    assert tuple(mask1.shape) == tuple(img_size)
    correct_ratio(mask1.sum() / np.prod(img_size))

    # Standard mask but by passing flat input_mask of ones
    input_mask = torch.ones(batch_size, *img_size)
    # should ignore batch_size
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum())
    correct_pixelwise(mask2)

    # As above but with no batch dimension in input_mask
    input_mask = torch.ones(*img_size)
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask)[
        "mask"
    ]  # should use batch_size
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # As above but with img_size missing channel dimension (bad practice)
    input_mask = torch.ones(*img_size[1:])
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # Generate splitting mask from already subsampled mask
    input_mask = torch.zeros(batch_size, *img_size)
    input_mask[..., 10:20] = 1
    mask3 = gen.step(batch_size=batch_size, input_mask=input_mask)["mask"]
    correct_ratio(mask3.sum() / input_mask.sum())
    correct_pixelwise(mask3)
