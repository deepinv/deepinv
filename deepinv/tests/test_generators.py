from deepinv.physics.generator import (
    GaussianMaskGenerator,
    EquispacedMaskGenerator,
    RandomMaskGenerator,
    PolyOrderMaskGenerator,
)
from deepinv.physics.generator.base import seed_from_string
import pytest
import numpy as np
import torch
import deepinv as dinv
import itertools
from pathlib import Path

# Avoiding nondeterministic algorithms
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
if not torch.cuda.is_available() and torch.__version__ >= "2.1.0":
    torch.use_deterministic_algorithms(True)

torch.backends.cudnn.deterministic = True

# Generators to test (make sure they appear in find_generator as well)
GENERATORS = [
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "ProductConvolutionBlurGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
SIZES = [(5, 5), (6, 6)]
NUM_CHANNELS = [1, 3]


# MRI Generators
C, T, H, W = 2, 12, 256, 512
MRI_GENERATORS = ["gaussian", "random", "uniform", "poly"]
MRI_IMG_SIZES = [(H, W), (C, H, W), (C, T, H, W), (64, 64)]
MRI_ACCELERATIONS = [4, 10, 12]
MRI_CENTER_FRACTIONS = [0, 0.04, 24 / 512]

# Inpainting/Splitting Generators
INPAINTING_IMG_SIZES = [
    (2, 64, 40),
    (2, 1000),  # This will show warning but (C, M) is valid
    (2, 3, 64, 40),
]  # (C,H,W), (C,M), (C,T,H,W)
INPAINTING_GENERATORS = ["bernoulli", "gaussian", "multiplicative"]

# All devices to test
DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")

DTYPES = [torch.float32, torch.float64]


def find_generator(name, size, num_channels, device, dtype):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda:0
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    if name == "MotionBlurGenerator":
        g = dinv.physics.generator.MotionBlurGenerator(
            psf_size=size, num_channels=num_channels, device=device, dtype=dtype
        )
        keys = ["filter"]
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size, device=device, num_channels=num_channels, dtype=dtype
        )
        keys = ["filter", "coeff", "pupil"]
    elif name == "ProductConvolutionBlurGenerator":
        g = dinv.physics.generator.ProductConvolutionBlurGenerator(
            psf_generator=dinv.physics.generator.DiffractionBlurGenerator(
                psf_size=size, device=device, num_channels=num_channels, dtype=dtype
            ),
            img_size=512,
            n_eigen_psf=10,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "multipliers", "padding"]
    elif name == "DownsamplingGenerator":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"], factors=[2, 4]
        )
        keys = ["filters", "factors"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device, dtype=dtype)
        keys = ["sigma"]
    elif name == "GainGenerator":
        g = dinv.physics.generator.GainGenerator(device=device, dtype=dtype)
        keys = ["gain"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_shape(name, size, num_channels, device, dtype):
    r"""
    Tests generators shape.
    """

    generator, size, keys = find_generator(name, size, num_channels, device, dtype)
    batch_size = 4

    params = generator.step(batch_size=batch_size)

    assert list(params.keys()) == keys

    if "filter" in params.keys():
        assert params["filter"].shape == (batch_size, num_channels, size[0], size[1])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_newparams(name, device, dtype):
    r"""
    Tests generators shape.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, 1, device, dtype)
    batch_size = 1

    if name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=0)
    params1 = generator.step(batch_size=batch_size, seed=1)

    for key in param_key:
        assert torch.any(params0[key] != params1[key])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_seed(name, device, dtype):
    r"""
    Tests generators shape.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, 1, device, dtype)
    batch_size = 1

    if name == "MotionBlurGenerator":
        param_key = ["filter"]
    elif name == "DiffractionBlurGenerator":
        param_key = ["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        param_key = ["filters", "multipliers"]
    elif name == "SigmaGenerator":
        param_key = ["sigma"]

    params0 = generator.step(batch_size=batch_size, seed=42)
    params1 = generator.step(batch_size=batch_size, seed=42)

    for key in param_key:
        assert torch.allclose(params0[key], params1[key])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_generation(name, device, dtype):
    r"""
    Tests generators shape.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, 1, device, dtype)
    batch_size = 1
    params = generator.step(batch_size=batch_size, seed=0)
    if name == "MotionBlurGenerator" or name == "DiffractionBlurGenerator":
        w = params["filter"]
    elif name == "ProductConvolutionBlurGenerator":
        w = params["filters"]
    elif name == "SigmaGenerator":
        w = params["sigma"]

    wref = (
        torch.load(
            f"deepinv/tests/assets/generators/{name.lower()}_{device}_{dtype}.pt"
        )
        .to(device)
        .to(dtype)
    )
    assert torch.allclose(w, wref, atol=1e-8)


@pytest.mark.parametrize(
    "name", sorted(set(GENERATORS).difference(set(["ProductConvolutionBlurGenerator"])))
)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_average(name, device, dtype):
    r"""
    Tests generators average.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, 1, device, dtype)
    # Set generator seed for reproducibility
    generator.rng_manual_seed(0)

    n_avg = 4

    # Store the keys of a single step call for future comparison
    params = generator.step(batch_size=1, seed=0)
    keys = set(params.keys())

    for batch_size in [1, 2, n_avg]:
        batch_size = 1
        params = generator.average(5, batch_size=batch_size)
        assert isinstance(params, dict)
        assert set(params.keys()) == keys


#################################
### DOWNSAMPLING GENERATORS #####
#################################


@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_downsampling_generator(num_channels, device, dtype):
    r"""
    Test downsampling generator.
    This test is different from the above ones because we do not generate a random kernel at each iteration, but
    we sample from a list.
    """
    # we need sufficiently large sizes to ensure well definedness of the operation
    size = (32, 32)

    physics = dinv.physics.Downsampling(
        img_size=(num_channels, size[0], size[1]),
        device=device,
        filter="bicubic",
        factor=4,
    )
    generator, _, _ = find_generator(
        "DownsamplingGenerator", size, num_channels, device, dtype
    )

    batch_size = 1  # Must be 1 as filters with different shapes can't be batched
    params = generator.step(batch_size=batch_size, seed=0)

    x = torch.randn((batch_size, num_channels, size[0], size[1])).to(device)
    y = physics(x, **params)

    assert y.shape[-1] == x.shape[-1] // params["factor"]


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
    elif generator_name == "poly":
        g = PolyOrderMaskGenerator(
            img_size, acceleration=acc, center_fraction=center_fraction, poly_order=2
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

    mask = generator.step(batch_size=batch_size, seed=0)["mask"]

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


def choose_inpainting_generator(name, img_size, split_ratio, pixelwise, device, rng):
    if name == "bernoulli":
        return dinv.physics.generator.BernoulliSplittingMaskGenerator(
            img_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=rng,
        )
    elif name == "gaussian":
        return dinv.physics.generator.GaussianSplittingMaskGenerator(
            img_size=img_size,
            split_ratio=split_ratio,
            device=device,
            pixelwise=pixelwise,
            rng=rng,
        )
    elif name == "multiplicative":
        mri_gen = dinv.physics.generator.GaussianMaskGenerator(
            img_size=img_size,
            acceleration=2,
            device=device,
            rng=rng,
        )
        return dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
            img_size=img_size,
            split_generator=mri_gen,
            device=device,
        )
    else:
        raise Exception("The generator chosen doesn't exist")


@pytest.mark.parametrize("generator_name", INPAINTING_GENERATORS)
@pytest.mark.parametrize("img_size", INPAINTING_IMG_SIZES)
@pytest.mark.parametrize("pixelwise", (False, True))
@pytest.mark.parametrize("split_ratio", (0.5,))
@pytest.mark.parametrize("device", DEVICES)
def test_inpainting_generators(
    generator_name, batch_size, img_size, pixelwise, split_ratio, device, rng
):
    if generator_name in ("gaussian", "multiplicative") and len(img_size) < 3:
        pytest.skip(
            "Gaussian and multiplicative splitting mask not valid for images of shape smaller than (C, H, W)"
        )

    if generator_name == "multiplicative" and not pixelwise:
        pytest.skip("Multiplicative mask test not defined for non pixelwise masking.")

    gen = choose_inpainting_generator(
        generator_name, img_size, split_ratio, pixelwise, device, rng
    )  # Assume generator always receives "correct" img_size i.e. not one with dims missing

    def correct_ratio(ratio, rtol=1e-2, atol=1e-2):
        assert torch.isclose(
            ratio,
            torch.tensor([split_ratio], device=device),
            rtol=rtol,
            atol=atol,
        )

    def correct_pixelwise(mask):
        if pixelwise:
            assert torch.all(mask[:, 0, ...] == mask[:, 1, ...])
        else:
            assert not torch.all(mask[:, 0, ...] == mask[:, 1, ...])

    # Standard generate mask
    mask1 = gen.step(batch_size=batch_size, seed=0)["mask"]
    correct_ratio(mask1.sum() / np.prod((batch_size, *img_size)))
    correct_pixelwise(mask1)

    # Standard without batch dim
    mask1 = gen.step(batch_size=None, seed=0)["mask"]
    assert tuple(mask1.shape) == tuple(img_size)
    correct_ratio(mask1.sum() / np.prod(img_size))

    # Standard mask but by passing flat input_mask of ones
    input_mask = torch.ones(batch_size, *img_size)
    # should ignore batch_size
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(mask2.sum() / input_mask.sum())
    correct_pixelwise(mask2)

    # As above but with no batch dimension in input_mask
    input_mask = torch.ones(*img_size, device=device)
    mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)[
        "mask"
    ]  # should use batch_size
    correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # As above but with img_size missing channel dimension (bad practice)
    # Note: Multiplicative mask must have correct input mask shape
    # Note: 1D input_mask not compatible with pixelwise
    if generator_name != "multiplicative" and not (len(img_size) <= 2 and pixelwise):
        input_mask = torch.ones(*img_size[1:], device=device)
        mask2 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
        correct_ratio(mask2.sum() / input_mask.sum() / batch_size)

    # Generate splitting mask from already subsampled mask
    # Multiplicative splitting will rarely be exact
    input_mask = torch.zeros(batch_size, *img_size, device=device)
    input_mask[..., 10:20] = 1
    mask3 = gen.step(batch_size=batch_size, input_mask=input_mask, seed=0)["mask"]
    correct_ratio(
        mask3.sum() / input_mask.sum(),
        atol=1e-2 if generator_name != "multiplicative" else 2e-1,
    )
    correct_pixelwise(mask3)


@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rng", [torch.Generator().manual_seed(0)])
def test_inpainting_generator_random_ratio(num_channels, device, dtype, rng):
    # NOTE elements of this test are now redundant given above tests
    size = (100, 100)  # we take it large to have significant statistical numbers after
    physics = dinv.physics.Inpainting((num_channels, size[0], size[1]), 0.9)

    split_ratio = 0.6
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        (num_channels, size[0], size[1]), split_ratio=split_ratio
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    experimental_split_ratio = (mask[0] == 1).sum() / mask[0].numel()
    assert abs(experimental_split_ratio.item() - split_ratio) < 1e-2

    # check forward
    x = torch.randn((batch_size, num_channels, size[0], size[1]), generator=rng)
    y = physics(x, **params)
    experimental_split_ratio_obs = 1 - (y[0] == 0).sum() / y[0].numel()
    assert torch.allclose(
        experimental_split_ratio, experimental_split_ratio_obs, rtol=1e-4
    )

    # now we do the same with each element in the batch for random_split_ratio
    min_split_ratio = 0.001
    max_split_ratio = 0.5
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        (num_channels, size[0], size[1]),
        split_ratio=split_ratio,
        random_split_ratio=True,
        min_split_ratio=min_split_ratio,
        max_split_ratio=max_split_ratio,
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size, seed=0)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    x = torch.randn((batch_size, num_channels, size[0], size[1]), generator=rng)
    y = physics(x, **params)

    list_exp_split_ratio = []
    for b in range(batch_size):
        experimental_split_ratio = (mask[b] == 1).sum() / mask[b].numel()
        assert experimental_split_ratio.item() < max_split_ratio + 1e-2
        assert experimental_split_ratio.item() > min_split_ratio - 1e-2

        # check forward
        experimental_split_ratio_obs = 1 - (y[b] == 0).sum() / y[b].numel()
        assert torch.allclose(
            experimental_split_ratio, experimental_split_ratio_obs, rtol=1e-3
        )

        list_exp_split_ratio.append(experimental_split_ratio)

    # check that split ratios are different between batches
    assert abs(list_exp_split_ratio[0] - list_exp_split_ratio[1]) > 1e-2


def test_string_seed():
    # Dummy long paths
    paths = [f"{'deepinv/'*10}{p}" for p in Path("deepinv/tests").glob("*.py")]
    seeds = [seed_from_string(p) for p in paths]

    # Assert unique seeds
    assert len(set(seeds)) == len(seeds)

    # Assert seed in correct range for manual_seed
    for s in seeds:
        assert -0x8000_0000_0000_0000 < s < 0xFFFF_FFFF_FFFF_FFFF

    # Assert generators different
    states = [torch.Generator().manual_seed(s).get_state() for s in seeds]
    assert len(set(states)) == len(states)
