import os
import shutil
import copy
from math import sqrt
from typing import Optional
import pytest
import torch
import numpy as np
from deepinv.physics.forward import adjoint_function
import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.physics.mri import MRI, MRIMixin, DynamicMRI, MultiCoilMRI
from deepinv.utils import TensorList


# Linear forward operators to test (make sure they appear in find_operator as well)
# We do not include operators for which padding is involved, they are tested separately
OPERATORS = [
    "CS",
    "fastCS",
    "inpainting",
    "inpainting_clone",
    "demosaicing",
    "denoising",
    "colorize",
    "fftdeblur",
    "singlepixel",
    "deblur_valid",
    "deblur_circular",
    "deblur_reflect",
    "deblur_replicate",
    "deblur_constant",
    "composition",
    "composition2",
    "space_deblur_valid",
    "space_deblur_circular",
    "space_deblur_reflect",
    "space_deblur_replicate",
    "space_deblur_constant",
    "hyperspectral_unmixing",
    "3Ddeblur_valid",
    "3Ddeblur_circular",
    "super_resolution_valid",
    "super_resolution_circular",
    "super_resolution_reflect",
    "super_resolution_replicate",
    "super_resolution_constant",
    "down_resolution_circular",
    "down_resolution_reflect",
    "down_resolution_replicate",
    "down_resolution_constant",
    "aliased_super_resolution",
    "super_resolution_matlab",
    "fast_singlepixel",
    "fast_singlepixel_cake_cutting",
    "fast_singlepixel_zig_zag",
    "fast_singlepixel_xy",
    "fast_singlepixel_old_sequency",
    "MRI",
    "DynamicMRI",
    "MultiCoilMRI",
    "3DMRI",
    "3DMultiCoilMRI",
    "aliased_pansharpen",
    "pansharpen_valid",
    "pansharpen_circular",
    "pansharpen_reflect",
    "pansharpen_replicate",
    "complex_compressed_sensing",
    "radio",
    "radio_weighted",
    "structured_random",
    "cassi",
    "ptychography_linear",
]

NONLINEAR_OPERATORS = ["haze", "lidar"]

PHASE_RETRIEVAL_OPERATORS = [
    "random_phase_retrieval",
    "structured_random_phase_retrieval",
    "ptychography",
]

NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "Neighbor2Neighbor",
    "LogPoisson",
    "Gamma",
    "SaltPepper",
]


WRAPPERS = [
    None,
    "LinearPhysicsMultiScaler",
    "PhysicsCropper",
]


def find_operator(name, device, imsize=None, get_physics_param=False):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    img_size = (3, 16, 8) if imsize is None else imsize
    norm = 1
    dtype = torch.float
    padding = next(
        (
            p
            for p in ["valid", "circular", "reflect", "replicate", "constant"]
            if p in name
        ),
        None,
    )

    rng = torch.Generator(device).manual_seed(0)
    if name == "CS":
        m = 30
        p = dinv.physics.CompressedSensing(
            m=m, img_size=img_size, device=device, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 0.75  # Marcenko-Pastur law, second term is a small n correction
        params = []
    elif name == "fastCS":
        p = dinv.physics.CompressedSensing(
            m=20,
            fast=True,
            channelwise=True,
            img_size=img_size,
            device=device,
            rng=rng,
        )
        params = []
    elif name == "colorize":
        p = dinv.physics.Decolorize(device=device)
        norm = 0.4468
        params = ["srf"]
    elif name == "cassi":
        img_size = (7, 37, 31) if imsize is None else imsize
        p = dinv.physics.CompressiveSpectralImaging(img_size, device=device, rng=rng)
        norm = 1 / img_size[0]
        params = ["mask"]
    elif name == "inpainting":
        p = dinv.physics.Inpainting(img_size=img_size, mask=0.5, device=device, rng=rng)
        params = ["mask"]
    elif name == "inpainting_clone":
        p = dinv.physics.Inpainting(img_size=img_size, mask=0.5, device=device, rng=rng)
        p = p.clone()
        params = ["mask"]
    elif name == "demosaicing":
        p = dinv.physics.Demosaicing(img_size=img_size, device=device)
        norm = 1.0
        params = []
    elif name == "MRI":
        img_size = (2, 17, 11) if imsize is None else imsize  # C,H,W
        p = MRI(img_size=img_size, device=device)
        params = ["mask"]
    elif name == "3DMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,D,H,W where D is depth
        p = MRI(img_size=img_size, three_d=True, device=device)
        params = ["mask"]
    elif name == "DynamicMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,T,H,W where T is time
        p = DynamicMRI(img_size=img_size, device=device)
        params = ["mask"]
    elif name == "MultiCoilMRI":
        img_size = (2, 17, 11) if imsize is None else imsize  # C,H,W
        n_coils = 7
        maps = torch.ones(
            (1, n_coils, img_size[-2], img_size[-1]),
            dtype=torch.complex64,
            device=device,
        ) / sqrt(
            n_coils
        )  # B,N,H,W where N is coil dimension
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, device=device)
        params = ["mask", "coil_maps"]
    elif name == "3DMultiCoilMRI":
        img_size = (
            (2, 5, 17, 11) if imsize is None else imsize
        )  # C,D,H,W where D is depth
        n_coils = 15
        maps = torch.ones(
            (1, n_coils, img_size[-3], img_size[-2], img_size[-1]),
            dtype=torch.complex64,
            device=device,
        ) / sqrt(
            n_coils
        )  # B,N,D,H,W where N is coils and D is depth
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, three_d=True, device=device)
        params = ["mask"]
    elif name == "Tomography":
        img_size = (1, 16, 16) if imsize is None else imsize  # C,H,W
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=img_size[-1], device=device
        )
        params = ["theta"]
    elif name == "composition":
        img_size = (3, 16, 16) if imsize is None else imsize
        p1 = dinv.physics.Downsampling(
            img_size=img_size, factor=2, device=device, padding="same", filter=None
        )
        p2 = dinv.physics.BlurFFT(
            img_size=img_size,
            device=device,
            filter=dinv.physics.blur.gaussian_blur(sigma=(1.0)),
        )
        p = p1 * p2
        norm = 1 / 2**2
        params = ["filter"]
    elif name == "composition2":
        img_size = (3, 16, 16) if imsize is None else imsize
        p1 = dinv.physics.Downsampling(
            img_size=img_size, factor=2, device=device, filter=None
        )
        p2 = dinv.physics.BlurFFT(
            img_size=(3, 8, 8),
            device=device,
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.5)),
        )
        p = p2 * p1
        params = ["filter"]
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1, rng=rng))
        params = []
    elif name.startswith("pansharpen"):
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(
            img_size=img_size,
            device=device,
            padding=padding,
            filter="bilinear",
            use_brovey=False,
        )
        norm = 0.4
        params = []
    elif name == "aliased_pansharpen":
        img_size = (3, 30, 32) if imsize is None else imsize
        p = dinv.physics.Pansharpen(
            img_size=img_size, device=device, filter=None, use_brovey=False
        )
        norm = 1.4
        params = ["filter"]
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_size=img_size, device=device, rng=rng
        )
        params = ["mask"]
    elif name == "fast_singlepixel_cake_cutting":
        p = dinv.physics.SinglePixelCamera(
            m=20,
            fast=True,
            img_size=img_size,
            device=device,
            rng=rng,
            ordering="cake_cutting",
        )
        params = ["mask"]
    elif name == "fast_singlepixel_zig_zag":
        p = dinv.physics.SinglePixelCamera(
            m=20,
            fast=True,
            img_size=img_size,
            device=device,
            rng=rng,
            ordering="zig_zag",
        )
        params = ["mask"]
    elif name == "fast_singlepixel_xy":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_size=img_size, device=device, rng=rng, ordering="xy"
        )
        params = []
    elif name == "fast_singlepixel_old_sequency":
        p = dinv.physics.SinglePixelCamera(
            m=20,
            fast=True,
            img_size=img_size,
            device=device,
            rng=rng,
            ordering="old_sequency",
        )
        params = ["mask"]
    elif name == "singlepixel":
        m = 20
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=False, img_size=img_size, device=device, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 3.7  # Marcenko-Pastur law, second term is a small n correction
        params = ["mask"]
    elif name.startswith("deblur"):
        img_size = (3, 17, 19) if imsize is None else imsize
        p = dinv.physics.Blur(
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.25, 0.1), angle=45.0),
            padding=padding,
            device=device,
        )
        params = ["filter"]
    elif name == "fftdeblur":
        img_size = (3, 17, 19) if imsize is None else imsize
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.bicubic_filter(),
            device=device,
        )
        params = ["filter"]
    elif name.startswith("space_deblur"):
        img_size = (3, 20, 13) if imsize is None else imsize
        h = dinv.physics.blur.bilinear_filter(factor=2).unsqueeze(0).to(device)
        h /= torch.sum(h)
        h = torch.cat([h, h], dim=2)
        p = dinv.physics.SpaceVaryingBlur(
            filters=h,
            multipliers=torch.ones(
                (
                    1,
                    img_size[0],
                    2,
                )
                + img_size[-2:],
                device=device,
            ).to(device)
            * 0.5,
            padding=padding,
            device=device,
        )
        params = ["filters", "multipliers"]
    elif name == "hyperspectral_unmixing":
        img_size = (15, 32, 32) if imsize is None else imsize  # x (E, H, W)
        p = dinv.physics.HyperSpectralUnmixing(E=15, C=64, device=device)
        params = ["M"]
    elif name.startswith("3Ddeblur"):
        img_size = (1, 7, 6, 8) if imsize is None else imsize  # C,D,H,W
        h_size = (1, 1, 4, 3, 5)
        h = torch.rand(h_size)
        h /= h.sum()
        p = dinv.physics.Blur(
            filter=h,
            padding=padding,
            device=device,
        )
        params = ["filter"]
    elif name == "aliased_super_resolution":
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0
        p = dinv.physics.Downsampling(
            img_size=img_size,
            factor=factor,
            padding=padding,
            device=device,
            filter=None,
        )
        params = []
    elif name == "super_resolution_matlab":
        img_size = (1, 32, 32)
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.DownsamplingMatlab(factor=factor)
        params = []
    elif name.startswith("super_resolution"):
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.Downsampling(
            img_size=img_size,
            factor=factor,
            padding=padding,
            device=device,
            filter="bilinear",
            dtype=dtype,
        )
        params = ["filter"]
    elif name.startswith("down_resolution"):
        img_size = (1, 32, 32) if imsize is None else imsize
        factor = 2
        norm = 1.0 / factor**2
        p = dinv.physics.Upsampling(
            img_size=(img_size[0], img_size[1] * factor, img_size[2] * factor),
            factor=factor,
            padding=padding,
            device=device,
            filter="bilinear",
            dtype=dtype,
        )
        params = ["filter"]
    elif name == "complex_compressed_sensing":
        img_size = (1, 8, 8) if imsize is None else imsize
        m = 50
        p = dinv.physics.CompressedSensing(
            m=m,
            img_size=img_size,
            dtype=torch.cdouble,
            device=device,
            compute_inverse=True,
            rng=rng,
        )
        dtype = p.dtype
        norm = (1 + np.sqrt(np.prod(img_size) / m)) ** 2
        params = ["mask"]
    elif "radio" in name:
        dtype = torch.cfloat
        img_size = (1, 64, 64) if imsize is None else imsize
        pytest.importorskip(
            "torchkbnufft",
            reason="This test requires torchkbnufft. It should be "
            "installed with `pip install torchkbnufft`",
        )

        # Generate regular grid for sampling
        y = torch.linspace(-1, 1, img_size[-2])
        x = torch.linspace(-1, 1, img_size[-1])
        grid_y, grid_x = torch.meshgrid(y, x)
        uv = torch.stack((grid_y, grid_x), dim=-1) * torch.pi  # normalize [-pi, pi]

        # Reshape to [nb_points x 2]
        uv = uv.view(-1, 2)
        uv = uv.to(device)

        if "weighted" in name:
            dataWeight = torch.linspace(
                0.01, 0.99, uv.shape[0], device=device
            )  # take a non-trivial weight
        else:
            dataWeight = torch.tensor(
                [
                    1.0,
                ]
            )

        p = dinv.physics.RadioInterferometry(
            img_size=img_size[1:],
            samples_loc=uv.permute((1, 0)),
            dataWeight=dataWeight,
            real_projection=False,
            dtype=dtype,
            device=device,
            noise_model=dinv.physics.GaussianNoise(0.0, rng=rng),
        )
        params = []
    elif name == "structured_random":
        img_size = (1, 8, 8) if imsize is None else imsize
        p = dinv.physics.StructuredRandom(
            img_size=img_size, output_size=img_size, device=device
        )
        params = []
    elif name == "ptychography_linear":
        img_size = (1, 32, 32) if imsize is None else imsize
        dtype = torch.complex64
        norm = 1.32
        p = dinv.physics.PtychographyLinearOperator(
            img_size=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
        params = ["probe", "shifts"]
    else:
        raise Exception("The inverse problem chosen doesn't exist")

    if not get_physics_param:
        return p, img_size, norm, dtype
    else:
        return p, img_size, norm, dtype, params


def find_nonlinear_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    if name == "haze":
        x = dinv.utils.TensorList(
            [
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, 1, 16, 16, device=device),
                torch.randn(1, device=device),
            ]
        )
        p = dinv.physics.Haze()

    elif name == "lidar":
        x = torch.rand(1, 3, 16, 16, device=device)
        p = dinv.physics.SinglePhotonLidar(device=device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, x


def wrap_physics(wrapper_name, physics, img_size, device):
    if wrapper_name == "LinearPhysicsMultiScaler":
        factors = [2, 4, 8]
        p = dinv.physics.LinearPhysicsMultiScaler(
            physics=physics, img_shape=img_size, factors=factors, device=device
        )
        img_size_out = (img_size[0], img_size[-2] // 4, img_size[-1] // 4)
    elif wrapper_name == "PhysicsCropper":
        crop = (2, 4)
        p = dinv.physics.PhysicsCropper(physics=physics, crop=crop)
        img_size_out = (
            *img_size[:-2],
            img_size[-2] + crop[-2],
            img_size[-1] + crop[-1],
        )
    else:
        raise Exception(
            f"The wrapper {wrapper_name} is not in the `wrap_physics` function"
        )
    return p, img_size_out


def find_phase_retrieval_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.PhaseRetrieval) forward operator.
    """
    if name == "random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.RandomPhaseRetrieval(m=500, img_size=img_size, device=device)
    elif name == "ptychography":
        img_size = (1, 32, 32)
        p = dinv.physics.Ptychography(
            img_size=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
    elif name == "structured_random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.StructuredRandomPhaseRetrieval(
            img_size=img_size, output_size=img_size, n_layers=2, device=device
        )
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size


def test_stacking(device):
    r"""
    Tests if stacking physics operators is consistent with applying them sequentially.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    imsize = (2, 5, 5)
    p1 = dinv.physics.Inpainting(mask=0.5, img_size=imsize, device=device)
    p2 = dinv.physics.Physics(A=lambda x: x**2)
    p3 = p1.stack(p2)

    x = torch.randn(imsize, device=device).unsqueeze(0)
    y1 = p1.A(x)
    y2 = p2.A(x)
    y = p3.A(x)

    assert torch.allclose(y[0], y1)
    assert torch.allclose(y[1], y2)

    assert not isinstance(p3, dinv.physics.StackedLinearPhysics)
    assert isinstance(p3, dinv.physics.StackedPhysics)

    p4 = p1.stack(p1)
    y = p4(x)
    assert isinstance(p4, dinv.physics.StackedLinearPhysics)
    assert len(y) == 2
    assert p4.A_adjoint(y).shape == x.shape

    p5 = p4.stack(p4)
    y = p5(x)
    assert len(p5) == 4
    assert len(y) == 4


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_adjointness(name, device, rng):
    r"""
    Tests if a linear forward operator has a well defined adjoint.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts adjointness
    """
    physics, imsize, _, dtype = find_operator(name, device)

    if name == "radio":
        dtype = torch.cfloat

    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    error = physics.adjointness_test(x).abs()
    assert error < 1e-3

    if (
        "pansharpen" in name or "radio" in name
    ):  # automatic adjoint does not work for inputs that are not torch.tensors
        return
    f = adjoint_function(physics.A, x.shape, x.device, x.dtype)

    y = physics.A(x)
    error2 = (f(y) - physics.A_adjoint(y)).flatten().mean().abs()

    assert error2 < 1e-3


LIST_DOWN_OP = [
    "down_resolution_circular",
    "down_resolution_reflect",
    "down_resolution_replicate",
    "down_resolution_constant",
]


@pytest.mark.parametrize("name", LIST_DOWN_OP)
@pytest.mark.parametrize("kernel", ["bilinear", "bicubic", "sinc", "gaussian"])
def test_upsampling(device, rng, name, kernel):
    r"""
    This function tests that the Upsampling and Downsampling operators are effectively adjoint to each other.

    Note that the test does not hold when the padding is not 'valid', as the Upsampling operator
    does not support 'valid' padding.
    """
    padding = name.split("_")[-1]  # get padding type from name
    physics, imsize, _, dtype = find_operator(name, device)
    physics_adjoint, _, _, dtype = find_operator(
        "super_resolution_" + padding, device, imsize=imsize
    )

    # physics.register_buffer("filter", None)
    physics.update_parameters(filter=kernel)

    # physics_adjoint.register_buffer("filter", None)
    physics_adjoint.update_parameters(filter=kernel)

    factor = physics.factor

    x = torch.randn(
        (1, imsize[0], imsize[1], imsize[2]),
        device=device,
        dtype=dtype,
        generator=rng,
    )

    out = physics(x)
    assert out.shape == (1, imsize[0], imsize[1] * factor, imsize[2] * factor)

    y = physics(x)
    err1 = (physics.A_adjoint(y) - physics_adjoint(y)).flatten().mean().abs()
    assert err1 < 1e-6

    imsize_new = (*imsize[:1], imsize[1] * factor, imsize[2] * factor)
    physics_adjoint, _, _, dtype = find_operator(
        "super_resolution_" + padding, device, imsize=imsize_new
    )  # we need to redefine the adjoint operator with the new image size

    # physics_adjoint.register_buffer("filter", None)
    physics_adjoint.update_parameters(filter=kernel)

    x = torch.randn(imsize_new, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    y = physics_adjoint(x)
    err2 = (physics.A(y) - physics_adjoint.A_adjoint(y)).flatten().mean().abs()
    assert err2 < 1e-6


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_multiscale_wrapper(name, device, rng):
    r"""
    Tests if a linear physics operator can be wrapped with a multiscale wrapper.
    """

    # defining a list of exceptions to skip  # TODO: fix for those?
    list_exceptions = [
        "pansharpen",  # shape handling
        "radio",  # data type (complex)
        "3d",  # shape handling
        "ptychography",  # ?
        "composition2",  # shape handling
        "dynamicmri",  # shape handling
        "complex_compressed_sensing",  # data type (complex)
    ]

    if any(exc in name.lower() for exc in list_exceptions):
        pytest.skip(f"Skipping test for operator '{name}' as it matches an exception.")

    base_shape = (32, 32)
    scale = 2

    _, img_size_orig, _, _ = find_operator(
        name,
        device,
    )  # get img_size for the operator
    physics, img_size_orig, _, dtype = find_operator(
        name,
        device,
        imsize=(*img_size_orig[:-2], base_shape[-2], base_shape[-1]),
    )  # get physics for the operator with base img size

    image_shape = (
        *img_size_orig[:-2],
        base_shape[-2] // (scale**2),
        base_shape[-1] // (scale**2),
    )
    x = torch.rand((1, *image_shape), dtype=dtype)  # add batch dim

    new_physics = dinv.physics.LinearPhysicsMultiScaler(
        physics, (*image_shape[:-2], *base_shape), factors=[2, 4, 8], dtype=dtype
    )  # define a multiscale physics with base img size (1, 32, 32)
    y = new_physics(x, scale=scale)
    Aty = new_physics.A_adjoint(y, scale=scale)

    assert Aty.shape == x.shape


@pytest.mark.parametrize("name", OPERATORS)
def test_operator_cropper(name, device, rng):
    r"""
    Tests if a linear physics operator can be wrapped with a crop wrapper.
    """

    physics, image_shape, _, dtype = find_operator(
        name,
        device,
    )  # get physics for the operator with base img size

    x = torch.rand((1, *image_shape), dtype=dtype)  # add batch dim
    padding_shape = (2, 5)
    x_new = torch.nn.functional.pad(x, (padding_shape[1], 0, padding_shape[0], 0))

    new_physics = dinv.physics.PhysicsCropper(
        physics,
        padding_shape,
    )
    y = new_physics(x_new)
    Aty = new_physics.A_adjoint(y)

    assert Aty.shape == x_new.shape


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_norm(name, device, rng):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.8,1.2)
    """
    if name == "radio_weighted":  # weighted nufft norm is not tested
        return

    if name == "singlepixel" or name == "CS":
        device = torch.device("cpu")
        rng = torch.Generator("cpu")

    torch.manual_seed(0)
    physics, imsize, norm_ref, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)
    norm = physics.compute_norm(x, max_iter=1000, tol=1e-6)
    bound = 1e-2
    # if theoretical bound relies on Marcenko-Pastur law, or if pansharpening, relax the bound
    if (
        name in ["singlepixel", "CS", "complex_compressed_sensing", "radio"]
        or "pansharpen" in name
    ):
        bound = 0.2
    # convolution norm is not simple in those cases
    if (
        "reflect" in name
        or "replicate" in name
        or "constant" in name
        or "valid" in name
    ):
        pass
    else:
        assert torch.abs(norm - norm_ref) < bound


@pytest.mark.parametrize("name", NONLINEAR_OPERATORS)
def test_nonlinear_operators(name, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts correct shapes
    """
    physics, x = find_nonlinear_operator(name, device)
    y = physics(x)
    xhat = physics.A_dagger(y)
    assert x.shape == xhat.shape


@pytest.mark.parametrize("name", OPERATORS)
def test_pseudo_inverse(name, device, rng):
    r"""
    Tests if a linear physics operator has a well-defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize, _, dtype = find_operator(name, device)

    x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)

    r = physics.A_adjoint(physics.A(x))  # project to range of A^T
    y = physics.A(r)
    error = torch.linalg.vector_norm(
        physics.A_dagger(y, solver="lsqr", tol=0.0001, max_iter=50, verbose=True) - r
    ) / torch.linalg.vector_norm(r)
    assert error < 0.05


@pytest.mark.parametrize("name", OPERATORS)
def test_decomposable(name, device, rng):
    physics, imsize, _, dtype = find_operator(name, device)
    if isinstance(physics, dinv.physics.DecomposablePhysics):
        x = torch.randn(imsize, device=device, dtype=dtype, generator=rng).unsqueeze(0)

        proj = lambda u: physics.V(physics.V_adjoint(u))
        r = proj(x)  # project
        assert (
            torch.linalg.vector_norm(proj(r) - r) / torch.linalg.vector_norm(r) < 1e-3
        )

        y = physics.A(x)
        proj = lambda u: physics.U(physics.U_adjoint(u))
        r = proj(y)
        assert (
            torch.linalg.vector_norm(proj(r) - r) / torch.linalg.vector_norm(r) < 1e-3
        )


@pytest.fixture
def mri_img_size():
    return 1, 2, 3, 16, 16  # B, C, T, H, W


@pytest.mark.parametrize("mri", [MRI, DynamicMRI, MultiCoilMRI])
def test_MRI(mri, mri_img_size, device, rng):
    r"""
    Test MRI and DynamicMRI functions

    Assert mask is applied to physics wherever it is passed.

    :param mri_img_size: (tuple) image size tuple (B, C, T, H, W)
    :param device: (torch.device) cpu or cuda:x
    :param rng: (torch.Generator)
    """

    B, C, T, H, W = mri_img_size
    if rng.device != device:
        rng = torch.Generator(device=device).manual_seed(0)
    x, y = (
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
    )

    coil_maps_kwarg = {}

    if mri is MRI:
        x = x[:, :, 0, :, :]
        y = y[:, :, 0, :, :]
    elif mri is MultiCoilMRI:
        # y treat T as coil dim for tests
        x = x[:, :, 0, :, :]
        coil_maps_kwarg = {"coil_maps": T}

    for mask_size in [(H, W), (T, H, W), (C, T, H, W), (B, C, T, H, W)]:
        # Remove time dim for static MRI
        _mask_size = mask_size if mri is DynamicMRI else mask_size[:-3] + mask_size[-2:]

        mask, mask2 = (
            torch.ones(_mask_size, device=device)
            - torch.eye(*_mask_size[-2:], device=device),
            torch.zeros(_mask_size, device=device)
            + torch.eye(*_mask_size[-2:], device=device),
        )

        # Empty mask
        physics = mri(img_size=x.shape, device=device, **coil_maps_kwarg)
        y1 = physics(x)
        x1 = physics.A_adjoint(y)
        assert torch.sum(y1 == 0) == 0
        assert torch.sum(x1 == 0) == 0

        # Set mask in constructor
        physics = mri(
            mask=mask, img_size=mri_img_size, device=device, **coil_maps_kwarg
        )
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (physics.mask == 0))

        # Set mask in forward
        y1 = physics(x, mask=mask2)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask2 == 0))

        # Mask retained in previous forward
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask2 == 0))

        # Set mask via update
        physics.update(mask=mask)
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil
        assert torch.all((y1 == 0) == (mask == 0))

        # Check mag/rss reduces channel dim
        x_hat = physics.A_adjoint(
            y, **{("rss" if isinstance(physics, MultiCoilMRI) else "mag"): True}
        )
        # (B, 2, ...) -> (B, 1, ...)
        assert x_hat.shape[:2] == (x.shape[0], 1) and y.shape[1] == 2

        # Check rss works for multi-coil
        if isinstance(physics, MultiCoilMRI):
            assert y.shape[:3] == (x.shape[0], 2, T)  # B,C,N(=T)
            xrss = physics.A_adjoint(y, rss=True)
            assert xrss.shape == (x.shape[0], 1, *x.shape[2:])  # B,1,H,W


@pytest.mark.parametrize("mri", [MRI, DynamicMRI, MultiCoilMRI])
def test_MRI_noise_domain(mri, mri_img_size, device, rng):
    r"""
    Test that MRI noise addition is 0 where mask is 0

    :param mri_img_size: (tuple) image size tuple (B, C, T, H, W)
    :param device: (torch.device) cpu or cuda:x
    :param rng: (torch.Generator)
    """

    B, C, T, H, W = mri_img_size
    if rng.device != device:
        rng = torch.Generator(device=device).manual_seed(0)
    x, y = (
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
        torch.rand(mri_img_size, generator=rng, device=device) + 1,
    )

    coil_maps_kwarg = {}

    if mri is MRI:
        x = x[:, :, 0, :, :]
        y = y[:, :, 0, :, :]
    elif mri is MultiCoilMRI:
        # y treat T as coil dim for tests
        x = x[:, :, 0, :, :]
        coil_maps_kwarg = {"coil_maps": T}

    for mask_size in [(H, W), (T, H, W), (C, T, H, W), (B, C, T, H, W)]:
        # Remove time dim for static MRI
        _mask_size = mask_size if mri is DynamicMRI else mask_size[:-3] + mask_size[-2:]

        mask = torch.ones(_mask_size, device=device) - torch.eye(
            *_mask_size[-2:], device=device
        )

        # Set mask in constructor
        physics = mri(
            mask=mask,
            img_size=mri_img_size,
            device=device,
            noise_model=dinv.physics.noise.GaussianNoise(sigma=0.1).to(device),
            **coil_maps_kwarg,
        )
        y1 = physics(x)
        if isinstance(physics, MultiCoilMRI):
            y1 = y1[:, :, 0]  # check 0th coil

        assert torch.all((y1 == 0) == (physics.mask == 0))


@pytest.mark.parametrize("name", OPERATORS)
def test_concatenation(name, device):
    if "pansharpen" in name:  # TODO: fix pansharpening
        return
    physics, imsize, _, dtype = find_operator(name, device)

    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    y = physics(x)
    physics = (
        dinv.physics.Inpainting(
            img_size=y.size()[1:], mask=0.5, pixelwise=False, device=device
        )
        * physics
    )

    r = physics.A_adjoint(physics.A(x))  # project to range of A^T
    y = physics.A(r)
    error = torch.linalg.vector_norm(
        physics.A_dagger(y, solver="lsqr", tol=0.0001) - r
    ) / torch.linalg.vector_norm(r)
    assert error < 0.01


@pytest.mark.parametrize("name", PHASE_RETRIEVAL_OPERATORS)
def test_phase_retrieval(name, device):
    r"""
    Tests to ensure the phase retrieval operator is behaving as expected.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize = find_phase_retrieval_operator(name, device)
    x = torch.randn(imsize, dtype=torch.cfloat, device=device).unsqueeze(0)

    # nonnegativity
    assert (physics(x) >= 0).all()
    # same outputes for x and -x
    assert torch.equal(physics(x), physics(-x))


def test_phase_retrieval_Avjp(device):
    r"""
    Tests if the gradient computed with A_vjp method of phase retrieval is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    torch.set_grad_enabled(True)
    x = torch.randn((1, 1, 3, 3), dtype=torch.cfloat, device=device, requires_grad=True)
    physics = dinv.physics.RandomPhaseRetrieval(m=10, img_size=(1, 3, 3), device=device)
    loss = L2()
    func = lambda x: loss(x, torch.ones_like(physics(x)), physics)[0]
    grad_value = torch.func.grad(func)(x)
    jvp_value = loss.grad(x, torch.ones_like(physics(x)), physics)
    assert torch.isclose(grad_value[0], jvp_value, rtol=1e-5).all()


def test_linear_physics_Avjp(device, rng):
    r"""
    Tests if the gradient computed with A_vjp method of linear physics is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    torch.set_grad_enabled(True)
    x = torch.randn(
        (1, 1, 3, 3),
        dtype=torch.float,
        device=device,
        generator=rng,
        requires_grad=True,
    )
    physics = dinv.physics.CompressedSensing(m=10, img_size=(1, 3, 3), device=device)
    loss = L2()
    func = lambda x: loss(x, torch.ones_like(physics(x)), physics)[0]
    grad_value = torch.func.grad(func)(x)
    jvp_value = loss.grad(x, torch.ones_like(physics(x)), physics)
    assert torch.isclose(grad_value[0], jvp_value, rtol=1e-5).all()


def test_physics_Avjp(device):
    r"""
    Tests if the vector Jacobian product computed by A_vjp method of physics is correct.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the computed gradients and expected values is more than 1e-5
    """
    A = torch.eye(3, dtype=torch.float64)

    def A_forward(v):
        return A @ v

    physics = dinv.physics.Physics(A=A_forward)
    for _ in range(100):
        x = torch.randn(3, dtype=torch.float64)
        v = torch.randn(3, dtype=torch.float64)
        # Jacobian in this case should be identity
        assert torch.allclose(physics.A_vjp(x, v), v)


def choose_noise(noise_type, device="cpu"):
    gain = 0.1
    sigma = 0.1
    mu = 0.2
    N0 = 1024.0
    l = torch.ones((1), device=device)
    p, s = 0.025, 0.025
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise()
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "Neighbor2Neighbor":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "LogPoisson":
        noise_model = dinv.physics.LogPoissonNoise(N0, mu)
    elif noise_type == "Gamma":
        noise_model = dinv.physics.GammaNoise(l)
    elif noise_type == "SaltPepper":
        noise_model = dinv.physics.SaltPepperNoise(p=p, s=s)
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("noise_type", NOISES)
def test_noise(device, noise_type):
    r"""
    Tests noise models.
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = choose_noise(noise_type, device)
    x = torch.ones((1, 3, 2), device=device).unsqueeze(0)

    y1 = physics(
        x
        # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    )
    assert y1.shape == x.shape


def test_noise_domain(device):
    r"""
    Tests that there is no noise outside the domain of the measurement operator, i.e. that in y = Ax+n, we have
    n=0 where Ax=0.
    """
    x = torch.ones((1, 3, 12, 7), device=device)
    mask = torch.ones_like(x[0])
    # mask[:, x.shape[-2]//2-3:x.shape[-2]//2+3, x.shape[-1]//2-3:x.shape[-1]//2+3] = 0
    mask[0, 0, 0] = 0
    mask[1, 1, 1] = 0
    mask[2, 2, 2] = 0

    physics = dinv.physics.Inpainting(img_size=x.shape, mask=mask, device=device)
    physics.noise_model = choose_noise("Gaussian")
    y1 = physics(
        x
        # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    )
    assert y1.shape == x.shape

    assert y1[0, 0, 0, 0] == 0
    assert y1[0, 1, 1, 1] == 0
    assert y1[0, 2, 2, 2] == 0


def test_blur(device):
    r"""
    Tests that there is no noise outside the domain of the measurement operator, i.e. that in y = Ax+n, we have
    n=0 where Ax=0.
    """
    torch.manual_seed(0)
    x = torch.randn((3, 128, 128), device=device).unsqueeze(0)
    h = torch.ones((1, 1, 5, 5)) / 25.0

    physics_blur = dinv.physics.Blur(
        filter=h,
        device=device,
        padding="circular",
    )

    physics_blurfft = dinv.physics.BlurFFT(
        img_size=(1, x.shape[-2], x.shape[-1]),
        filter=h,
        device=device,
    )

    y1 = physics_blur(x)
    y2 = physics_blurfft(x)

    back1 = physics_blur.A_adjoint(y1)
    back2 = physics_blurfft.A_adjoint(y2)

    assert y1.shape == y2.shape
    assert back1.shape == back2.shape

    error_A = (y1 - y2).flatten().abs().max()
    error_At = (back1 - back2).flatten().abs().max()

    assert error_A < 1e-6
    assert error_At < 1e-6


def test_reset_noise(device):
    r"""
    Tests that the reset function works.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is > 0
    """
    x = torch.ones((1, 3, 3), device=device).unsqueeze(0)
    rng = torch.Generator(device)
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(0.1, rng=rng)

    y1 = physics(x)
    y2 = physics(x, sigma=0.2)

    assert physics.noise_model.sigma == 0.2

    physics.noise_model = dinv.physics.PoissonNoise(0.1, rng=rng)

    y1 = physics(x)
    y2 = physics(x, gain=0.2)

    assert physics.noise_model.gain == 0.2

    physics.noise_model = dinv.physics.PoissonGaussianNoise(0.5, 0.3, rng=rng)
    y1 = physics(x)
    y2 = physics(x, sigma=0.2, gain=0.2)

    assert physics.noise_model.gain == 0.2
    assert physics.noise_model.sigma == 0.2


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("parallel_computation", [True, False])
@pytest.mark.parametrize("fan_beam", [True, False])
@pytest.mark.parametrize("circle", [True, False])
@pytest.mark.parametrize("adjoint_via_backprop", [True, False])
@pytest.mark.parametrize("fbp_interpolate_boundary", [True, False])
def test_tomography(
    normalize,
    parallel_computation,
    fan_beam,
    circle,
    adjoint_via_backprop,
    fbp_interpolate_boundary,
    device,
):
    r"""
    Tests tomography operator which does not have a numerically precise adjoint.

    :param device: (torch.device) cpu or cuda:x
    """
    imsize = (1, 16, 16)
    physics = dinv.physics.Tomography(
        img_width=imsize[-1],
        angles=imsize[-1],
        device=device,
        circle=circle,
        fan_beam=fan_beam,
        normalize=normalize,
        adjoint_via_backprop=adjoint_via_backprop,
        fbp_interpolate_boundary=fbp_interpolate_boundary,
        parallel_computation=parallel_computation,
    )

    x = torch.randn(imsize, device=device).unsqueeze(0)
    if adjoint_via_backprop:
        assert physics.adjointness_test(x).abs() < 1e-3
    r = physics.A_adjoint(physics.A(x)) * torch.pi / (2 * len(physics.radon.theta))
    y = physics.A(r)
    error = (physics.A_dagger(y) - r).flatten().mean().abs()
    epsilon = 0.2 if device == "cpu" else 0.3  # Relax a bit of GPU
    assert error < epsilon


@pytest.mark.parametrize(
    "padding", ("valid", "constant", "circular", "reflect", "replicate")
)
def test_downsampling_adjointness(padding, device):
    r"""
    Tests downsampling+blur operator adjointness for various image and filter sizes

    :param device: (torch.device) cpu or cuda:x
    """
    torch.manual_seed(0)

    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = (
            [nchan_im, 5, 5],
            [nchan_im, 6, 6],
            [nchan_im, 5, 6],
            [nchan_im, 6, 5],
        )
        size_filt = (
            [nchan_filt, 3, 3],
            [nchan_filt, 4, 4],
            [nchan_filt, 3, 4],
            [nchan_filt, 4, 3],
        )

        for sim in size_im:
            for sfil in size_filt:
                x = torch.rand(1, *sim).to(device)
                h = torch.rand(1, *sfil).to(device)

                physics = dinv.physics.Downsampling(
                    sim, filter=h, padding=padding, device=device
                )

                Ax = physics.A(x)
                y = torch.rand_like(Ax)
                Aty = physics.A_adjoint(y)
                Axy = torch.sum(Ax * y)
                Atyx = torch.sum(Aty * x)

                assert torch.abs(Axy - Atyx) < 1e-3


def test_prox_l2_downsampling(device):
    nchannels = ((1, 1), (3, 1), (3, 3))

    for nchan_im, nchan_filt in nchannels:
        size_im = ([nchan_im, 16, 16],)
        filters = ["bicubic", "bilinear", "sinc"]

        paddings = ("circular",)

        for pad in paddings:
            for sim in size_im:
                for h in filters:
                    x = torch.rand(sim)[None].to(device)

                    physics = dinv.physics.Downsampling(
                        sim, filter=h, padding=pad, device=device
                    )

                    y = physics(x)
                    # next we test the speedup formula of prox with fft
                    x_prox1 = physics.prox_l2(
                        physics.A_adjoint(y) * 0.0, y, gamma=1e5, use_fft=True
                    )
                    x_prox2 = physics.prox_l2(
                        physics.A_adjoint(y) * 0.0, y, gamma=1e5, use_fft=False
                    )

                    assert torch.abs(x_prox1 - x_prox2).max() < 1e-2


@pytest.mark.parametrize("imsize", ((8, 16),))  # must be even here
@pytest.mark.parametrize("channels", (1, 2))
@pytest.mark.parametrize("factor", (2, 4))
@pytest.mark.parametrize(
    "downsampling", (dinv.physics.Downsampling, dinv.physics.DownsamplingMatlab)
)
def test_downsampling_imsize(imsize, channels, device, factor, downsampling):
    # Test downsampling can update imsize on the fly
    x = torch.rand(1, channels, *imsize, device=device)
    physics = downsampling(device=device, factor=factor)
    assert physics(x).shape == (1, channels, imsize[0] // factor, imsize[1] // factor)
    assert physics.A_adjoint(x).shape == (
        1,
        channels,
        imsize[0] * factor,
        imsize[1] * factor,
    )
    assert physics.adjointness_test(x).abs() < 1e-3


def test_mri_fft():
    """
    Test that our torch FFT is the same as FastMRI FFT implementation.
    The following 5 functions are taken from
    from https://github.com/facebookresearch/fastMRI/blob/main/fastmri/fftc.py
    """

    def fft2c_new(data: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
        if not data.shape[-1] == 2:
            raise ValueError("Tensor does not have separate complex dim.")

        data = ifftshift(data, dim=[-3, -2])
        data = torch.view_as_real(
            torch.fft.fftn(  # type: ignore
                torch.view_as_complex(data), dim=(-2, -1), norm=norm
            )
        )
        data = fftshift(data, dim=[-3, -2])

        return data

    def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
        shift = shift % x.size(dim)
        if shift == 0:
            return x

        left = x.narrow(dim, 0, x.size(dim) - shift)
        right = x.narrow(dim, x.size(dim) - shift, shift)

        return torch.cat((right, left), dim=dim)

    def roll(x: torch.Tensor, shift: list[int], dim: list[int]) -> torch.Tensor:
        for s, d in zip(shift, dim, strict=True):
            x = roll_one_dim(x, s, d)

        return x

    def fftshift(x: torch.Tensor, dim: Optional[list[int]] = None) -> torch.Tensor:
        if dim is None:
            # this weird code is necessary for toch.jit.script typing
            dim = [0] * (x.dim())
            for i in range(1, x.dim()):
                dim[i] = i

        # also necessary for torch.jit.script
        shift = [0] * len(dim)
        for i, dim_num in enumerate(dim):
            shift[i] = x.shape[dim_num] // 2

        return roll(x, shift, dim)

    def ifftshift(x: torch.Tensor, dim: Optional[list[int]] = None) -> torch.Tensor:
        if dim is None:
            # this weird code is necessary for toch.jit.script typing
            dim = [0] * (x.dim())
            for i in range(1, x.dim()):
                dim[i] = i

        # also necessary for torch.jit.script
        shift = [0] * len(dim)
        for i, dim_num in enumerate(dim):
            shift[i] = (x.shape[dim_num] + 1) // 2

        return roll(x, shift, dim)

    x = torch.randn(4, 2, 16, 8)  # B,C,H,W

    # Our FFT
    xf1 = MRIMixin.from_torch_complex(MRIMixin.fft(MRIMixin.to_torch_complex(x)))

    # FastMRI FFT
    xf2 = fft2c_new(x.moveaxis(1, -1).contiguous()).moveaxis(-1, 1)

    assert torch.all(xf1 == xf2)


@pytest.fixture
def multispectral_channels():
    return 7


@pytest.mark.parametrize("srf", ("flat", "random", "rec601", "list"))
def test_decolorize(srf, device, imsize, multispectral_channels):
    channels = multispectral_channels
    if srf == "list":
        srf = list(range(channels))
        srf = [s / sum(srf) for s in srf]

    physics = dinv.physics.Decolorize(channels=channels, srf=srf, device=device)
    x = torch.ones((1, channels, *imsize[-2:]), device=device)
    x2 = physics.A_adjoint_A(x)

    assert x2.shape == x.shape
    assert torch.allclose(
        physics.srf.sum(), torch.tensor(1.0, device=device), rtol=1e-3
    )
    assert physics.srf.shape[1] == channels


@pytest.mark.parametrize("shear_dir", ["h", "w"])
@pytest.mark.parametrize("cassi_mode", ["ss", "sd"])
def test_CASSI(shear_dir, imsize, device, multispectral_channels, rng, cassi_mode):
    channels = multispectral_channels

    x = torch.ones(1, channels, *imsize[-2:]).to(device)
    physics = dinv.physics.CompressiveSpectralImaging(
        (channels, *imsize[-2:]),
        mask=None,
        mode=cassi_mode,
        shear_dir=shear_dir,
        device=device,
        rng=rng,
    )
    y = physics(x)
    if cassi_mode == "ss":
        assert y.shape == (x.shape[0], 1, *x.shape[2:])
    elif cassi_mode == "sd":
        if shear_dir == "h":
            assert y.shape == (x.shape[0], 1, x.shape[-2] + channels - 1, x.shape[-1])
        elif shear_dir == "w":
            assert y.shape == (x.shape[0], 1, x.shape[-2], x.shape[-1] + channels - 1)

    x_hat = physics.A_adjoint(y)
    assert x_hat.shape == x.shape


def test_unmixing(device):
    physics = dinv.physics.HyperSpectralUnmixing(
        M=torch.tensor(
            [
                [0.5, 0.5, 0.0],  # yellow endmember
                [0.0, 0.0, 1.0],  # blue endmember
            ],
            device=device,
        ),
        device=device,
    )
    # Image of shape B,C,H,W
    # Image consists of 2 pixels, one yellow and one blue
    y = (
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            device=device,
        )
        .unsqueeze(-1)
        .unsqueeze(0)
    )
    x_hat = physics.A_adjoint(y)

    assert torch.all(x_hat[:, 0].squeeze() == torch.tensor([1.0, 0.0], device=device))
    assert torch.all(x_hat[:, 1].squeeze() == torch.tensor([0.0, 1.0], device=device))


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_differentiability(name, device):
    r"""
    Tests if a forward operator is differentiable (can perform back-propagation).

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts differentiability
    """

    physics, imsize, _, dtype, params = find_operator(
        name, device, get_physics_param=True
    )

    if name == "radio":
        dtype = torch.cfloat

    # Only test for floating point tensor
    valid_dtype = [torch.float16, torch.float32, torch.float64]
    if dtype in valid_dtype:
        # Differentiate w.r.t to input image
        x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
        y = physics.A(x)
        x_hat = (
            torch.randn(imsize, device=device, dtype=dtype)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        with torch.enable_grad():
            y_hat = physics.A(x_hat)
            if isinstance(y_hat, TensorList):
                for y_hat_item, y_item in zip(y_hat.x, y.x, strict=True):
                    loss = torch.nn.functional.mse_loss(y_hat_item, y_item)
                    loss.backward()
                    assert x_hat.requires_grad == True
                    assert x_hat.grad is not None
                    assert torch.all(~torch.isnan(x_hat.grad))
            else:
                loss = torch.nn.functional.mse_loss(y_hat, y)
                loss.backward()
                assert x_hat.requires_grad == True
                assert x_hat.grad is not None
                assert torch.all(~torch.isnan(x_hat.grad))

        # Differentiate w.r.t to physics parameters
        if (
            not physics._buffers == dict() and len(params) > 0
        ):  # If the buffers are not empty (i.e. there is a parameter)
            x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
            buffers = copy.deepcopy(dict(physics.named_buffers()))
            parameters = {k: v for k, v in buffers.items() if k in params}
            # Set requires grad
            for k, v in parameters.items():
                if v.dtype in valid_dtype:
                    parameters[k] = v.requires_grad_(True)

            with torch.enable_grad():
                y_hat = physics.A(x, **parameters)
                if isinstance(y_hat, TensorList):
                    for y_hat_item, y_item in zip(y_hat.x, y.x, strict=True):
                        loss = torch.nn.functional.mse_loss(y_hat_item, y_item)
                        loss.backward()

                        for k, v in parameters.items():
                            if v.dtype in valid_dtype:
                                assert v.requires_grad == True
                                assert v.grad is not None
                                assert torch.all(~torch.isnan(v.grad))

                else:
                    loss = torch.nn.functional.mse_loss(y_hat, y)
                    loss.backward()

                    for k, v in parameters.items():
                        if v.dtype in valid_dtype:
                            assert v.requires_grad == True
                            assert v.grad is not None
                            assert torch.all(~torch.isnan(v.grad))


@pytest.mark.parametrize(
    "name", OPERATORS + NONLINEAR_OPERATORS + PHASE_RETRIEVAL_OPERATORS
)
def test_device_consistency(name):
    r"""
    Tests if a physics can be moved properly between devices.

    :param name: operator name (see find_operator)
    :return: asserts
    """

    def try_find_operator(name):
        physics, imsize, _, dtype = find_operator(name, "cpu")
        return physics, imsize, dtype

    def try_find_nonlinear_operator(name):
        physics, x = find_nonlinear_operator(name, "cpu")
        return physics, x, torch.float32

    def try_find_phase_retrieval_operator(name):
        (
            physics,
            imsize,
        ) = find_phase_retrieval_operator(name, "cpu")
        return physics, imsize, torch.complex64

    for finder in (
        try_find_operator,
        try_find_nonlinear_operator,
        try_find_phase_retrieval_operator,
    ):
        try:
            physics, imsize, dtype = finder(name)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not find an operator for {name}")

    # The current radio physics depends on torchkbnufft, which seems to be not compatible.
    if "radio" in name:
        pytest.skip(
            "Skip 'radio' operator for device consistency test, since the current implementation depends on torchkbnufft, which seems to be not compatible."
        )
    else:
        # Test CPU
        torch.manual_seed(11)
        # For non linear operators
        if not isinstance(imsize, (torch.Tensor, TensorList)):
            x = torch.randn(imsize, device="cpu", dtype=dtype).unsqueeze(0)
        else:
            x = imsize
        y1 = physics.A(x)
        assert y1.device == torch.device("cpu")
        # Move to GPU if cuda is available
        if torch.cuda.is_available():
            torch.manual_seed(11)
            cuda = torch.device("cuda:0")
            physics = physics.to(cuda)
            x = x.to(cuda)
            y2 = physics.A(x)
            assert y2.device == cuda

            # skip denoising that adds random noise in each forward call
            if not isinstance(physics, dinv.physics.Denoising):
                if isinstance(y2, TensorList):
                    for y11, y22 in zip(y1, y2, strict=True):
                        assert torch.linalg.norm((y11.to(cuda) - y22).ravel()) < 1e-5
                else:
                    assert torch.linalg.norm((y1.to(cuda) - y2).ravel()) < 1e-5


@pytest.mark.parametrize("name", OPERATORS)
def test_physics_state_dict(name, device):
    r"""
    Tests if the physics state dict is well behaved.

    :param name: operator name (see find_operator)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts state dict is saved.
    """

    def get_all_tensor_attrs(module, prefix=""):
        tensor_attrs = {}

        # Check direct attributes
        for name in dir(module):
            try:
                attr = getattr(module, name)
            except Exception:
                continue  # skip attributes that raise exceptions on access

            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(attr, torch.Tensor):
                tensor_attrs[full_name] = attr
            elif isinstance(attr, torch.nn.Module):
                # Recurse into submodules
                tensor_attrs.update(get_all_tensor_attrs(attr, prefix=full_name))

        return tensor_attrs

    physics, imsize, _, dtype = find_operator(name, device)
    if name == "radio":
        dtype = torch.cfloat

    # A cache dir for saving state dict
    cache_dir = "./cache_test_physics"
    os.makedirs(cache_dir, exist_ok=True)

    # If the buffers are not empty (i.e. there is a parameter)
    if not physics._buffers == dict():
        state_dict = physics.state_dict()
        # Check that all tensor attributes are in the state dict
        params = get_all_tensor_attrs(physics)

        assert set(state_dict.keys()) == set(params.keys())
        for k, v in params.items():
            assert torch.allclose(state_dict[k], v)

        # Save the state_dict
        torch.save(state_dict, os.path.join(cache_dir, f"{name}.pt"))

        # Reinitialize the physics
        new_physics, _, _, _ = find_operator(name, device)
        # Change to random parameters

        loaded_state_dict = torch.load(os.path.join(cache_dir, f"{name}.pt"))
        new_physics.load_state_dict(loaded_state_dict)
        new_state_dict = new_physics.state_dict()
        # Check that the state dict are identical
        assert set(state_dict.keys()) == set(new_state_dict.keys())
        for k, v in state_dict.items():
            assert torch.equal(v, new_state_dict[k])

        # Check two physics have the same output
        x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
        assert torch.allclose(physics(x), new_physics(x))

        # Remove the cache dir
        shutil.rmtree(cache_dir)


def test_composed_physics(device):
    img_size = (3, 32, 32)
    # First physics
    mask_1 = torch.ones(img_size, device=device).unsqueeze(0)
    mask_1[..., 10:15, 13:17] = 0.0
    physics_1 = dinv.physics.Inpainting(img_size=img_size, mask=mask_1, device=device)
    # Second physics
    mask_2 = torch.ones(img_size, device=device).unsqueeze(0)
    mask_2[..., 5:7, 9:13] = 0.0
    physics_2 = dinv.physics.Inpainting(img_size=img_size, mask=mask_2, device=device)

    composed_physics = physics_1 * physics_2  # physics_1(physics_2(.))
    x = torch.randn(img_size, device=device).unsqueeze(0)
    assert torch.equal(composed_physics.A(x), physics_1.A(physics_2.A(x)))
    assert torch.equal(composed_physics.A(torch.ones_like(x)), mask_1 * mask_2)

    # A blur physics
    physics_3 = dinv.physics.BlurFFT(
        img_size=img_size, filter=dinv.physics.blur.bilinear_filter(2.0), device=device
    )

    composed_physics = physics_1 * physics_3
    assert torch.equal(composed_physics.A(x), physics_1.A(physics_3.A(x)))
    assert torch.equal(
        composed_physics.A_adjoint(x), physics_3.A_adjoint(physics_1.A_adjoint(x))
    )

    # Compose with Transform:
    physics = dinv.physics.Blur(filter=dinv.physics.blur.bicubic_filter(3.0))
    T = dinv.transform.Shift()
    T_kwargs = {"x_shift": torch.tensor([1]), "y_shift": torch.tensor([1])}

    physics_mul = physics * dinv.physics.LinearPhysics(
        A=lambda x: T.inverse(x, **T_kwargs),
        A_adjoint=lambda y: T(y, **T_kwargs),
    )
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(physics_mul.A(x), physics.A(T.inverse(x, **T_kwargs)))
    y = physics_mul.A(x)
    assert torch.allclose(physics_mul.A_adjoint(y), T(physics.A_adjoint(y), **T_kwargs))
    assert torch.allclose(
        physics_mul.A_dagger(y),
        T(physics.A_dagger(y), **T_kwargs),
        atol=1e-4,
        rtol=1e-4,
    )

    # test non-linear physics - checking for possible bugs in noise model
    non_lin_physics = dinv.physics.Physics(
        A=lambda x: x**2, noise_model=dinv.physics.GaussianNoise(0.0)
    )
    p = physics * non_lin_physics

    y_2 = physics(non_lin_physics.A(x))
    assert torch.allclose(y_2, p(x))

    assert isinstance(p.noise_model, dinv.physics.ZeroNoise)
    assert isinstance(p, dinv.physics.Physics) and not isinstance(
        p, dinv.physics.LinearPhysics
    )

    p = non_lin_physics * physics

    y_2 = non_lin_physics.A(physics(x))
    assert torch.allclose(y_2, p(x))
    assert p.noise_model.sigma == 0.0


@pytest.mark.parametrize("name", OPERATORS)
def test_adjoint_autograd(name, device):
    # NOTE: The current implementation of adjoint_function does not support
    # physics that return tensor lists or complex tensors. It also does not
    # support RadioInterferometry although it is not entirely clear why.
    if name in {
        "aliased_pansharpen",
        "pansharpen_valid",
        "pansharpen_circular",
        "pansharpen_reflect",
        "pansharpen_replicate",
        "complex_compressed_sensing",
        "ptychography_linear",
        "radio",
        "radio_weighted",
    }:
        pytest.skip(f"Operator {name} is not supported by adjoint_function.")

    physics, imsize, _, dtype = find_operator(name, device)

    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    y = physics.A(x)

    A_adjoint = adjoint_function(physics.A, x.shape, x.device, x.dtype)

    # Compute Df^\top(z) using autograd where f(z) = A^\top z.
    y.requires_grad_()
    z = torch.randn_like(x, device=device, dtype=dtype)
    l = (z * A_adjoint(y)).sum()
    l.backward()
    # \delta y := \delta_y <z, A^\top y> = Az
    delta_y = y.grad
    Az = physics.A(z)
    assert torch.allclose(delta_y, Az, rtol=1e-5)


@pytest.mark.parametrize("name", OPERATORS)
def test_clone(name, device):
    physics, imsize, _, dtype = find_operator(name, device)

    # Add a dummy parameter used for further testing
    dummy_tensor = torch.randn(
        imsize,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device).manual_seed(0),
    )
    dummy_parameter = torch.nn.Parameter(dummy_tensor)
    physics.register_parameter("dummy", dummy_parameter)

    physics_clone = physics.clone()

    # Test clone type (parent class)
    assert type(physics_clone) == type(physics), "Clone is not of the same type."

    # Check parameters
    parameter_names = set(name for name, _ in physics.named_parameters())
    parameter_names_clone = set(name for name, _ in physics_clone.named_parameters())

    assert parameter_names == parameter_names_clone, "Parameter names do not match."

    for name in parameter_names.intersection(parameter_names_clone):
        param = physics.get_parameter(name)
        param_clone = physics_clone.get_parameter(name)

        # Check that params have been reallocated somewhere else in the memory space
        assert (
            param.data_ptr() != param_clone.data_ptr()
        ), f"Parameter {name} has not been cloned properly."

        # Check that changing one parameter does not change the other
        # NOTE: no_grad is necessary because autograd prevents in-place modifications
        # of leaf variables.
        with torch.no_grad():
            param.fill_(0)
            param_clone.fill_(1)
        assert not torch.allclose(param, param_clone), f"Expected different values"

    # Check buffers
    buffer_names = set(name for name, _ in physics.named_buffers())
    buffer_names_clone = set(name for name, _ in physics_clone.named_buffers())

    assert buffer_names == buffer_names_clone, "Buffer names do not match."

    for name in buffer_names.intersection(buffer_names_clone):
        buffer = physics.get_buffer(name)
        buffer_clone = physics_clone.get_buffer(name)

        # Check that buffers have been reallocated somewhere else in the memory space
        assert (
            buffer.data_ptr() != buffer_clone.data_ptr()
        ), f"Buffer {name} has not been cloned properly."

        # Check that changing one buffer does not change the other
        buffer.fill_(0)
        buffer_clone.fill_(1)
        assert not torch.allclose(buffer, buffer_clone), f"Expected different values"

    # Test that RNGs have been cloned successfully
    rng = getattr(physics, "rng", None)
    rng_clone = getattr(physics_clone, "rng", None)

    assert (rng is not None) == (
        rng_clone is not None
    ), "RNGs are not both set or unset."

    if rng is not None:
        assert torch.all(
            rng.get_state() == rng_clone.get_state()
        ), "RNG state does not match."

        arr = torch.randn(16, device=rng.device, generator=rng)
        arr_clone = torch.randn(16, device=rng_clone.device, generator=rng_clone)
        assert torch.allclose(
            arr, arr_clone
        ), "RNGs do not produce the same random numbers after cloning."

    # Additional tests
    if hasattr(physics, "mask") and physics.mask.dtype != torch.bool:
        # Save original values
        saved_mask = physics.mask
        saved_physics_clone = physics_clone

        physics.mask += 7
        physics_clone = physics.clone()
        assert torch.allclose(
            physics_clone.mask, physics.mask
        ), "Mask has not been cloned properly."

        # Restore original values
        physics_clone.mask = saved_mask
        physics_clone = physics_clone

    # Test other attributes than parameters and buffers
    attr_name = "img_size"
    is_attr = hasattr(physics, attr_name)
    is_parameter = attr_name in [name for name, _ in physics.named_parameters()]
    is_buffer = attr_name in [name for name, _ in physics.named_buffers()]
    if is_attr and not is_parameter and not is_buffer:
        # Save original values
        attr_val = getattr(physics, attr_name)
        attr_val_clone = getattr(physics_clone, attr_name)

        setattr(physics, attr_name, 42)
        physics_clone = physics.clone()
        assert getattr(physics_clone, attr_name) == getattr(
            physics, attr_name
        ), "Attribute has not been cloned properly."

        # Restore original values
        setattr(physics, attr_name, attr_val)
        setattr(physics_clone, attr_name, attr_val_clone)

    # Save original values
    saved_A = physics.A
    physics.A = lambda *args, **kwargs: "hi"

    x = torch.randn(
        imsize,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device).manual_seed(0),
    ).unsqueeze(0)
    assert physics.A(x) == "hi"
    assert physics_clone.A(x) != "hi"

    # Restore original values
    physics.A = saved_A

    # Check requires_grad in parameters and buffers

    saved_physics = physics
    saved_physics_clone = physics_clone

    # Use a clone as the base to avoid mutations across different tests as it
    # may happen when modifying parameters and buffers
    physics = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = True

    physics_clone = physics.clone()

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        assert param.requires_grad, "Cloned parameter does not require grad."

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = False

    physics_clone = physics.clone()

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        assert not param.requires_grad, "Cloned parameter should not require grad."

    for buffer in physics.buffers():
        if not torch.is_floating_point(buffer) and not torch.is_complex(buffer):
            continue
        buffer.requires_grad = True

    physics_clone = physics.clone()

    for buffer in physics_clone.buffers():
        if not torch.is_floating_point(buffer) and not torch.is_complex(buffer):
            continue
        assert buffer.requires_grad, "Cloned buffer does not require grad."

    for buffer in physics.buffers():
        buffer.requires_grad = False

    physics_clone = physics.clone()

    for buffer in physics_clone.buffers():
        assert not buffer.requires_grad, "Cloned buffer should not require grad."

    # Restore original values
    physics = saved_physics
    physics_clone = saved_physics_clone

    # Test autograd
    saved_physics = physics
    saved_physics_clone = physics_clone

    # Use a clone as the base to avoid mutations across different tests as it
    # may happen when modifying parameters and buffers
    physics = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param) and not torch.is_complex(param):
            continue
        param.requires_grad = True

    physics_clone = physics.clone()

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        l = param.flatten()[0]
        l.backward()
        assert param.grad is not None, "Parameter gradient is None after backward."

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        assert param.grad is None, "Cloned parameter should not have a gradient."

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        param.grad = None  # Reset gradients

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        param.grad = None  # Reset gradients

    for param in physics_clone.parameters():
        if not torch.is_floating_point(param):
            continue
        l = param.flatten()[0]
        l.backward()
        assert param.grad is not None, "Parameter gradient is None after backward."

    for param in physics.parameters():
        if not torch.is_floating_point(param):
            continue
        assert param.grad is None, "Original parameter should not have a gradient."

    # Restore original values
    physics = saved_physics
    physics_clone = saved_physics_clone


def test_physics_warn_extra_kwargs():
    with pytest.warns(
        UserWarning, match="Arguments {'sigma': 0.5} are passed to Denoising"
    ):
        dinv.physics.Denoising(sigma=0.5)


def test_automatic_A_adjoint(device):
    x = torch.randn((2, 3, 8, 8), device=device)
    physics = dinv.physics.LinearPhysics(
        A=lambda x: x.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for LinearPhysics with automatic A_adjoint."

    # test decomposable physics
    physics = dinv.physics.DecomposablePhysics(
        V_adjoint=lambda s: s.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)

    assert torch.allclose(
        physics.U(x), physics.U_adjoint(x)
    ), "U and U_adjoint should be identity if not provided."
    assert torch.allclose(physics.U(x), x), "U should be identity if not provided."
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch for DecomposablePhysics."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for DecomposablePhysics with automatic A_adjoint."

    physics = dinv.physics.DecomposablePhysics(
        U=lambda x: x.mean(dim=1, keepdim=True), img_size=(3, 8, 8)
    )

    y = physics(x)
    x_adj = physics.A_adjoint(y)

    assert torch.allclose(
        physics.V(x), physics.V_adjoint(x)
    ), "V and V_adjoint should be identity if not provided."
    assert torch.allclose(physics.V(x), x), "V should be identity if not provided."
    assert x_adj.shape == x.shape, "A_adjoint shape mismatch for DecomposablePhysics."
    assert (
        physics.adjointness_test(x) < 1e-4
    ), "Adjointness test failed for DecomposablePhysics with automatic A_adjoint."
