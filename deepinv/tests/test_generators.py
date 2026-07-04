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
    "GaussianBlurGenerator",
    "MotionBlurGenerator",
    "DiffractionBlurGenerator",
    "ProductConvolutionBlurGenerator",
    "SigmaGenerator",
]

MIXTURES = list(itertools.combinations(GENERATORS, 2))
# To test GeneratorMixture.use_batch_sampling feature, when compatible
# generators (same output keys and shapes), samples from different generators
# per batch element
MIXTURES += [("MotionBlurGenerator", "MotionBlurGenerator")]
MIXTURES += [("DiffractionBlurGenerator", "DiffractionBlurGenerator")]

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

DTYPES = [torch.float32, torch.float64]


# Fixture returns either None or a torch.Generator on the specified device, to test both cases:
# 1. an existing generator is passed to the physics generator
# 2. the physics generator creates its own default generator (according to the device values)
# with this fixture, generator.rng.device and device always match.
@pytest.fixture(params=[None, pytest.param("device", marks=pytest.mark.indirect)])
def rng(request, device):
    if request.param == "device":
        return torch.Generator(device=device).manual_seed(0)
    return None


def find_generator(name, size, device, dtype, psf_size=None, rng=None):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda:0
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    if name == "GaussianBlurGenerator":
        g = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=size, device=device, dtype=dtype
        )
        keys = ["filter"]
    elif name == "MotionBlurGenerator":
        g = dinv.physics.generator.MotionBlurGenerator(
            psf_size=size,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filter"]
    elif name == "DiffractionBlurGenerator":
        g = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filter", "coeff", "pupil", "fc"]
    elif name == "ProductConvolutionBlurGenerator":
        g = dinv.physics.generator.ProductConvolutionBlurGenerator(
            psf_generator=dinv.physics.generator.DiffractionBlurGenerator(
                psf_size=size,
                device=device,
                dtype=dtype,
                rng=rng,
            ),
            img_size=512,
            n_eigen_psf=10,
            device=device,
            dtype=dtype,
            rng=rng,
        )
        keys = ["filters", "multipliers"]
    elif name == "DownsamplingGenerator":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2, 4],
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator2":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator4":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[4],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "DownsamplingGenerator[2, 4]":
        g = dinv.physics.generator.DownsamplingGenerator(
            filters=["bilinear", "bicubic", "gaussian"],
            factors=[2, 4],
            psf_size=psf_size,
            rng=rng,
            device=device,
            dtype=dtype,
        )
        keys = ["filters", "factors"]
    elif name == "SigmaGenerator":
        g = dinv.physics.generator.SigmaGenerator(device=device, dtype=dtype, rng=rng)
        keys = ["sigma"]
    elif name == "GainGenerator":
        g = dinv.physics.generator.GainGenerator(device=device, dtype=dtype, rng=rng)
        keys = ["gain"]
    else:
        raise Exception("The generator chosen doesn't exist")
    return g, size, keys


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_shape(name, size, device, dtype, rng):
    r"""
    Tests generators shape. All blur generators produce single-channel output by default;
    multi-channel (colour) output is tested separately in test_diffraction_generator.
    """

    generator, size, keys = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 4

    params = generator.step(batch_size=batch_size)

    assert list(params.keys()) == keys

    if "filter" in params.keys():
        assert params["filter"].shape == (batch_size, 1, size[0], size[1])


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_newparams(name, device, dtype, rng):
    r"""
    Tests generators' ability to generate new parameters at each step.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 1

    if name == "GaussianBlurGenerator":
        param_key = ["filter"]
    elif name == "MotionBlurGenerator":
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
@pytest.mark.parametrize("dtype", DTYPES)
def test_generation_seed(name, device, dtype, rng):
    r"""
    Tests generators consistency with the same random seed.
    """
    size = (32, 32)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
    batch_size = 1

    if name == "GaussianBlurGenerator":
        param_key = ["filter"]
    elif name == "MotionBlurGenerator":
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


@pytest.mark.parametrize(
    "name", sorted(set(GENERATORS).difference(set(["ProductConvolutionBlurGenerator"])))
)
@pytest.mark.parametrize("dtype", [torch.float64])
def test_average(name, device, dtype, rng):
    r"""
    Tests generators average.
    """
    size = (5, 5)
    generator, size, _ = find_generator(name, size, device, dtype, rng=rng)
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
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("psf_size", [None, (31, 31)])
@pytest.mark.parametrize("fact", [None, 2, 4, [2, 4]])
def test_downsampling_generator(num_channels, device, dtype, psf_size, fact, rng):
    r"""
    Test downsampling generator.
    This test is different from the above ones because we do not generate a random kernel at each iteration, but
    we sample from a list.
    """
    # we need sufficiently large sizes to ensure well definedness of the operation
    size = (32, 32)

    str_fact = "" if fact is None else str(fact)

    physics = dinv.physics.Downsampling(
        img_size=(num_channels, size[0], size[1]),
        device=device,
        filter="bicubic",
        factor=4,
    )
    generator, _, _ = find_generator(
        "DownsamplingGenerator" + str_fact,
        size,
        device,
        dtype,
        psf_size=psf_size,
        rng=rng,
    )

    batch_size = (
        1 if fact is None else 128
    )  # Must be 1 as filters with different shapes can't be batched (case psf_size=None)

    if psf_size is None and batch_size > 1:
        # in this case, we have a generator that generates filters of different shapes
        with pytest.raises(ValueError):
            params = generator.step(batch_size=batch_size, seed=1)
    else:
        params = generator.step(batch_size=batch_size, seed=1)

        x = torch.randn(
            (batch_size, num_channels, size[0], size[1]),
            generator=generator.rng,
            device=device,
        )
        y = physics(x, **params)

        assert y.shape[-1] == x.shape[-1] // params["factor"].unique().item()

        if fact is not None and not isinstance(fact, list):
            assert fact == params["factor"].unique().item()


######################
### MRI GENERATORS ###
######################


@pytest.fixture
def batch_size():
    return 2


def choose_mri_generator(generator_name, img_size, acc, center_fraction, device, rng):
    if generator_name == "gaussian":
        g = GaussianMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "random":
        g = RandomMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "uniform":
        g = EquispacedMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            rng=rng,
            device=device,
        )
    elif generator_name == "poly":
        g = PolyOrderMaskGenerator(
            img_size,
            acceleration=acc,
            center_fraction=center_fraction,
            poly_order=2,
            rng=rng,
            device=device,
        )
    return g


@pytest.mark.parametrize("generator_name", MRI_GENERATORS)
@pytest.mark.parametrize("img_size", MRI_IMG_SIZES)
@pytest.mark.parametrize("acc", MRI_ACCELERATIONS)
@pytest.mark.parametrize("center_fraction", MRI_CENTER_FRACTIONS)
def test_mri_generator(
    generator_name, img_size, batch_size, acc, center_fraction, device, rng
):
    generator = choose_mri_generator(
        generator_name, img_size, acc, center_fraction, device, rng
    )
    # test across different accs and center fracations
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
        atol=1e-2 if generator_name != "multiplicative" else 2.5e-1,
    )
    correct_pixelwise(mask3)

    # Adapt to new img sizes
    assert gen.step(batch_size=batch_size, img_size=(73, 29))["mask"].shape[-2:] == (
        73,
        29,
    )

    # Raise error if input_mask and img_size both passed
    with pytest.raises(ValueError):
        gen.step(img_size=(20, 20), input_mask=(2, 20, 20))


@pytest.mark.parametrize("num_channels", NUM_CHANNELS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_inpainting_generator_random_ratio(num_channels, device, dtype, rng):
    # NOTE elements of this test are now redundant given above tests
    size = (100, 100)  # we take it large to have significant statistical numbers after
    physics = dinv.physics.Inpainting(
        (num_channels, size[0], size[1]), 0.9, device=device
    )

    split_ratio = 0.6
    generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
        (num_channels, size[0], size[1]),
        split_ratio=split_ratio,
        device=device,
        dtype=dtype,
        rng=rng,
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    experimental_split_ratio = (mask[0] == 1).sum() / mask[0].numel()
    assert abs(experimental_split_ratio.item() - split_ratio) < 1e-2

    # check forward
    x = torch.randn(
        (batch_size, num_channels, size[0], size[1]),
        generator=generator.rng,
        device=device,
    )
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
        device=device,
        rng=rng,
    )
    batch_size = 2
    params = generator.step(batch_size=batch_size, seed=0)

    mask = params["mask"]
    assert mask.shape == (batch_size, num_channels, size[0], size[1])

    x = torch.randn(
        (batch_size, num_channels, size[0], size[1]),
        generator=generator.rng,
        device=device,
    )
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


@pytest.mark.parametrize("apodize", [True, False])
@pytest.mark.parametrize("random_rotate", [True, False])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("convention", ["noll", "ansi"])
@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    "fc", [None, 0.2, (0.15, 0.2), torch.tensor([[0.10, 0.11], [0.2, 0.21]])]
)
@pytest.mark.parametrize("coeff", [None, torch.zeros(2, 35)])
def test_diffraction_generator(
    device,
    apodize,
    random_rotate,
    center,
    convention,
    is_3d,
    rng,
    fc,
    coeff,
):
    r"""
    Test diffraction generator.
    """

    dtype = torch.float32
    zernike_index = tuple(range(1, 36))  # All Zernike index up to 7th order
    pupil_size = (256, 256)
    if is_3d:
        size = (5, 5, 5)
        generator = dinv.physics.generator.DiffractionBlurGenerator3D(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )

    else:
        pupil_size = (256, 256)
        size = (5, 5)
        generator = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            center=center,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )

    batch_sizes = (1, 2)
    expected_keys = set(
        ["filter", "coeff", "pupil", "fc"]
        + (["angle"] if random_rotate else [])
        + (["coeff_tilt_x", "coeff_tilt_y"] if ((not is_3d) and center) else [])
    )
    for batch_size in batch_sizes:
        params = generator.step(
            batch_size=batch_size,
            seed=0,
            focal_length=0.004,
            aperture_diameter=0.002,
            apodize=apodize,
            random_rotate=random_rotate,
            fc=fc,
            coeff=coeff.to(device) if coeff is not None else None,
        )

        if fc is not None:
            if isinstance(fc, float):
                num_channels_out = 1
                batch_size_out = batch_size
            else:
                fc_tensor = torch.as_tensor(fc)
                if fc_tensor.ndim == 1:
                    fc_tensor = fc_tensor[None, :].expand(batch_size, -1)
                batch_size_out, num_channels_out = fc_tensor.shape
        else:
            batch_size_out = batch_size
            num_channels_out = 1

        if coeff is not None:
            if coeff.ndim == 2:
                batch_size_out = coeff.shape[0]
            elif coeff.ndim == 3:
                batch_size_out, num_channels_out = coeff.shape[:2]

        # print(fc, batch_size_out, num_channels_out)
        # print(params["filter"].shape, (batch_size_out, num_channels_out, *size))

        # Test keys and shapes
        assert set(params.keys()) == expected_keys
        assert params["filter"].shape == (batch_size_out, num_channels_out, *size)
        assert params["coeff"].shape == (
            batch_size_out,
            num_channels_out,
            len(zernike_index),
        )
        assert params["pupil"].shape == (batch_size_out, num_channels_out, *pupil_size)
        if random_rotate:
            assert params["angle"].shape == (batch_size_out,)
        if (not is_3d) and center:
            assert params["coeff_tilt_x"].shape == (batch_size_out, num_channels_out, 1)
            assert params["coeff_tilt_y"].shape == (batch_size_out, num_channels_out, 1)

        # Test generator consistency when coeff is None
        params2 = generator.step(
            batch_size=batch_size,
            seed=0,
            fc=fc,
        )
        if coeff is None:
            for key in params.keys():
                assert torch.allclose(params[key], params2[key])

        # Test generator variability when coeff is None
        params3 = generator.step(
            batch_size=batch_size,
            seed=1,
            fc=fc,
        )
        if coeff is None:
            for key in params.keys():
                if key == "fc":
                    assert torch.allclose(params[key], params3[key])
                else:
                    assert not torch.allclose(params[key], params3[key])

        # test raising ValueError when incompatible shapes
        if (
            fc is None
            and coeff is None
            and not apodize
            and not random_rotate
            and convention == "noll"
        ):
            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=2,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=3, K)  (B_f != B_c)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=2,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=2, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(1, 35).to(device),
                )  # (B_f=2, C_f=2) vs (1, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(3, 2, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=3, C_c=2, K)
            assert "does not match" in str(excinfo.value)

            with pytest.raises(ValueError) as excinfo:
                generator.step(
                    batch_size=5,
                    seed=1,
                    fc=0.2 * torch.ones(2, 2).to(device),
                    coeff=torch.zeros(2, 3, 35).to(device),
                )  # (B_f=2, C_f=2) vs (B_c=2, C_c=3, K)
            assert "does not match" in str(excinfo.value)

    # Test centering effect if center is True and psf size is large enough

    # Test only for 2D case as centering in 3D case is still under development
    if (not is_3d) and center and (not apodize) and (not random_rotate):
        batch_sizes = (1, 2)
        size = (71, 71)
        pupil_size = (256, 256)
        generator = dinv.physics.generator.DiffractionBlurGenerator(
            psf_size=size,
            device=device,
            zernike_index=zernike_index,
            index_convention=convention,
            apodize=apodize,
            random_rotate=random_rotate,
            center=center,
            dtype=dtype,
            pupil_size=pupil_size,
            rng=rng,
        )
        for batch_size in batch_sizes:
            generated_psf = generator.step(
                batch_size=batch_size,
                seed=0,
                focal_length=0.004,
                aperture_diameter=0.002,
                apodize=apodize,
                random_rotate=random_rotate,
            )["filter"]
            com_x, com_y = barycenter(generated_psf)
            assert torch.all(torch.round(com_x.abs(), decimals=1) <= 0.1)
            assert torch.all(torch.round(com_y.abs(), decimals=1) <= 0.1)


def barycenter(h):
    Ny, Nx = h.shape[-2:]
    centerx = (Nx / 2.0) - 0.5
    centery = (Ny / 2.0) - 0.5
    x = torch.arange(0, h.shape[-1]).to(h.device)
    y = torch.arange(0, h.shape[-2]).to(h.device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    com_x = (X[None, None, :, :] * h).sum(dim=(-2, -1)) - centerx
    com_y = (Y[None, None, :, :] * h).sum(dim=(-2, -1)) - centery
    return com_x, com_y


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("isotropic", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_gaussian_blur_generator(device, dim, isotropic, batch_size):
    r"""
    Validate GaussianBlurGenerator behaviors across 1D/2D/3D:
    - isotropic vs anisotropic sigma handling
    - float or tuple for sigma_min/max
    - float or tuple for angle_min/max
    """
    torch.manual_seed(0)

    # choose psf size according to dimension
    if dim == 1:
        psf_size = (7,)
    elif dim == 2:
        psf_size = (7, 7)
    else:
        psf_size = (5, 5, 5)

    if dim == 1:
        if isotropic:
            pytest.skip("Isotropic setting not relevant for 1D Gaussian blur.")
        # In 1D, isotropic should be ignored and sigma_min/max accept single float/integer or length-1 tuple
        generator = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=psf_size,
            sigma_min=0.5,
            sigma_max=1,
            device=device,
        )
        params = generator.step(batch_size=batch_size, seed=0)
        assert params["filter"].shape == (batch_size, 1, *psf_size)

        # providing length-1 tuple should also work
        generator = dinv.physics.generator.GaussianBlurGenerator(
            psf_size=psf_size,
            sigma_min=(0.5,),
            sigma_max=(1,),
            device=device,
        )
        params = generator.step(batch_size=batch_size, seed=0)
        assert params["filter"].shape == (batch_size, 1, *psf_size)

        # providing length-2 tuple should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=(0.5, 1.1),
                sigma_max=3.0,
                device=device,
            )

    elif dim == 2:
        # In 2D, generator can accept float, integer, length-1 or length-2 tuple for sigma_min/max. If different than length-2 tuple, the same min/max will be applied to both dimensions.

        for sigma_min, sigma_max in zip(
            [0.5, (0.5,), (0.5, 0.6)], [1.0, (1.0,), (1.0, 1.1)], strict=True
        ):
            generator = dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                angle_min=0.0,
                angle_max=(torch.pi,),
                device=device,
            )
            params = generator.step(batch_size=batch_size, seed=0)
            assert params["filter"].shape == (batch_size, 1, *psf_size)

        if isotropic:
            # check that the providing filter is indeed isotropic
            center = tuple(s // 2 for s in psf_size)
            for b in range(batch_size):
                assert torch.isclose(
                    params["filter"][b, 0, center[0] + 2, center[1] + 2],
                    params["filter"][b, 0, center[0] + 2, center[1] - 2],
                )
                assert torch.isclose(
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                )
                assert torch.isclose(
                    params["filter"][b, 0, center[0] - 2, center[1] - 2],
                    params["filter"][b, 0, center[0] - 2, center[1] + 2],
                )

        # providing length-2 tuple for angle_min should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(0.0, 0.5),
                angle_max=(1.0),
                device=device,
            )
        # providing length-2 tuple for angle_max should raise error
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(0.0),
                angle_max=(1.0, 1.5),
                device=device,
            )
        # angle_min should be less than angle_max
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=True,
                sigma_min=0.5,
                sigma_max=2.0,
                angle_min=(1.5),
                angle_max=(0.5),
                device=device,
            )

        # Angle constructor validation: 2D only accepts single float/integer or length-1 tuple for angle_min/max, not length-2 tuple
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size, angle_min=(0.1, 0.2), angle_max=(0.2, 0.3)
            )

    elif dim == 3:
        # In 3D, generator can accept float, integer, length-1 or length-3 tuple for sigma_min/max. If different than length-3 tuple, the same min/max will be applied to all dimensions.

        for sigma_min, sigma_max in zip(
            [0.5, (0.5,), (0.5, 0.6, 0.7)], [1.0, (1.0,), (1.0, 1.1, 1.2)], strict=True
        ):
            generator = dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size,
                isotropic=isotropic,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                angle_min=(-torch.pi, 0.0, 0.0),
                angle_max=(torch.pi, 0.5 * torch.pi, 2 * torch.pi),
                device=device,
            )
            params = generator.step(batch_size=batch_size, seed=0)
            assert params["filter"].shape == (batch_size, 1, *psf_size)

        # Angle constructor validation: 3D must accept length-3
        with pytest.raises(ValueError):
            dinv.physics.generator.GaussianBlurGenerator(
                psf_size=psf_size, angle_min=(0.1, 0.2)
            )

    # Single sigma for the whole batch -> pass an explicit sigma tensor with identical rows
    sigma_same = torch.tensor([[1.23] * dim] * batch_size, device=device)
    params_single = generator.step(batch_size=batch_size, sigma=sigma_same, seed=0)
    filt_single = params_single["filter"]
    if batch_size > 1:
        assert torch.allclose(filt_single[0], filt_single[1])

    # Different sigma per sample -> pass per-sample sigma tensor
    if batch_size > 1:
        sig0 = [(0.6 + 0.1 * i) for i in range(dim)]
        sig1 = [(1.6 + 0.1 * i) for i in range(dim)]
        sigma_tensor = torch.tensor([sig0, sig1], device=device, dtype=torch.float32)
        params_diff = generator.step(batch_size=batch_size, sigma=sigma_tensor, seed=0)
        filt_diff = params_diff["filter"]
        assert not torch.allclose(filt_diff[0], filt_diff[1])

    # Angle handling: for 2D and 3D, passing different angles per batch should change kernels
    if dim == 2 and batch_size > 1:
        # angle should change the kernel only when sigma is anisotropic
        sigma_aniso = torch.tensor([[0.6, 1.2]] * batch_size, device=device)
        angle_tensor = torch.tensor([0.0, 1.0], device=device)
        p_angle = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_aniso, seed=0
        )
        f_angle = p_angle["filter"]
        assert not torch.allclose(f_angle[0], f_angle[1])

        # check that if sigma is isotropic, angle does not change the kernel
        sigma_iso = torch.tensor([[0.9, 0.9]] * batch_size, device=device)
        p_angle_iso = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_iso, seed=0
        )
        f_angle_iso = p_angle_iso["filter"]
        assert torch.allclose(f_angle_iso[0], f_angle_iso[1])

    if dim == 3 and batch_size > 1:
        sigma_aniso = torch.tensor([[0.6, 0.8, 1.2]] * batch_size, device=device)
        angle_tensor = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.7]], device=device)
        p_angle = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_aniso, seed=0
        )
        f_angle = p_angle["filter"]
        assert not torch.allclose(f_angle[0], f_angle[1])

        # check that if sigma is isotropic, angle does not change the kernel
        sigma_iso = torch.tensor([[0.9, 0.9, 0.9]] * batch_size, device=device)
        p_angle_iso = generator.step(
            batch_size=batch_size, angle=angle_tensor, sigma=sigma_iso, seed=0
        )
        f_angle_iso = p_angle_iso["filter"]
        assert torch.allclose(f_angle_iso[0], f_angle_iso[1])


@pytest.mark.parametrize("generators", MIXTURES)
@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("use_batch_sampling", [True, False])
def test_generator_mixture(generators, size, dtype, use_batch_sampling, device, rng):

    generator_pair = []
    for name in generators:
        g, _, _ = find_generator(name, size, device, dtype, rng=rng)
        generator_pair.append(g)

    mixture = dinv.physics.generator.GeneratorMixture(
        generator_pair,
        [0.5, 0.5],
        use_batch_sampling=use_batch_sampling,
        device=device,
        rng=rng,
        verbose=True,
    )

    # When two generators belong to the same class and have same output keys and shapes
    # use_batch_sampling must be True if specified
    if type(generator_pair[0]) == type(generator_pair[1]):
        assert mixture.use_batch_sampling == use_batch_sampling

        # Check that the mixture functions properly when use_batch_sampling is True
        # and all params from the batch are from the same generator (force it by using batch_size=1)
        params = mixture.step(batch_size=1, seed=0)

    params = mixture.step(batch_size=4, seed=0)
    assert isinstance(params, dict)

    # Check the set keys of produced by the mixture are the same as the keys of the individual generators
    assert set(params.keys()).intersection(
        set.union(*[set(g.step(batch_size=1, seed=0).keys()) for g in generator_pair])
    ) == set(params.keys())


#################################
### CONFOCAL BLUR GENERATOR 3D ##
#################################


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "lambda_ill,lambda_coll,expected_channels",
    [
        (489e-9, 525e-9, 1),  # single-channel (scalar wavelengths)
        ([489e-9, 561e-9], [525e-9, 620e-9], 2),  # two-channel (list wavelengths)
    ],
)
def test_confocal_blur_generator_3d(
    device, batch_size, lambda_ill, lambda_coll, expected_channels
):
    r"""
    Test ConfocalBlurGenerator3D output shapes and keys for single- and multi-channel cases.
    """
    psf_size = (5, 11, 11)
    zernike_index = (3,)  # minimal: one coefficient for speed

    generator = dinv.physics.generator.ConfocalBlurGenerator3D(
        psf_size=psf_size,
        zernike_index=zernike_index,
        lambda_ill=lambda_ill,
        lambda_coll=lambda_coll,
        device=device,
    )

    params = generator.step(batch_size=batch_size, seed=0)

    expected_keys = {
        "filter",
        "coeff_ill",
        "coeff_coll",
        "pupil_ill",
        "pupil_coll",
        "fc_ill",
        "fc_coll",
    }
    assert set(params.keys()) == expected_keys
    assert params["filter"].shape == (batch_size, expected_channels, *psf_size)
    assert params["fc_ill"].shape == (batch_size, expected_channels)
    assert params["fc_coll"].shape == (batch_size, expected_channels)

    # Reproducibility
    params2 = generator.step(batch_size=batch_size, seed=0)
    assert torch.allclose(params["filter"], params2["filter"])


########################################
### DIFFRACTION USED_ZERNIKE_INDEX TEST #
########################################


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("n_used", [3, 10])
def test_diffraction_used_zernike_index(device, batch_size, n_used):
    r"""
    Test DiffractionBlurGenerator.step(used_zernike_index=...) feature.

    Verifies:
    - output shape is (B, 1, H, W) regardless of subset size
    - coeff shape last dim equals n_used
    - different subsets produce different PSFs (not degenerate)
    - passing indices outside self.zernike_index raises ValueError
    """
    psf_size = (15, 15)
    full_index = list(range(3, 37))  # 34 Noll indices

    generator = dinv.physics.generator.DiffractionBlurGenerator(
        psf_size=psf_size,
        zernike_index=full_index,
        device=device,
    )

    used = full_index[:n_used]
    params = generator.step(batch_size=batch_size, seed=0, used_zernike_index=used)

    assert params["filter"].shape == (batch_size, 1, *psf_size)
    assert params["coeff"].shape[-1] == n_used

    # Different subset → different PSF
    other_used = full_index[-n_used:]
    params_other = generator.step(
        batch_size=batch_size, seed=0, used_zernike_index=other_used
    )
    assert not torch.allclose(params["filter"], params_other["filter"])

    # Passing an index not in self.zernike_index must raise
    with pytest.raises(ValueError, match="not in self.zernike_index"):
        generator.step(
            batch_size=1, used_zernike_index=[1, 2]
        )  # 1,2 not in range(3,37)
