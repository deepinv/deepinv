from math import sqrt
from typing import Optional, List
import pytest
import torch
import numpy as np
from deepinv.physics.forward import adjoint_function
import deepinv as dinv
from deepinv.optim.data_fidelity import L2
from deepinv.physics.mri import MRI, MRIMixin, DynamicMRI, MultiCoilMRI, SequentialMRI

# Linear forward operators to test (make sure they appear in find_operator as well)
# We do not include operators for which padding is involved, they are tested separately
OPERATORS = [
    "CS",
    "fastCS",
    "inpainting",
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
    "aliased_super_resolution",
    "fast_singlepixel",
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
]


def find_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (:class:`deepinv.physics.Physics`) forward operator.
    """
    img_size = (3, 16, 8)
    norm = 1
    dtype = torch.float
    padding = None
    paddings = ["valid", "circular", "reflect", "replicate", "constant"]
    for p in paddings:
        if p in name:
            padding = p
            break

    rng = torch.Generator(device).manual_seed(0)
    if name == "CS":
        m = 30
        p = dinv.physics.CompressedSensing(
            m=m, img_shape=img_size, device=device, compute_inverse=True, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 0.75  # Marcenko-Pastur law, second term is a small n correction
    elif name == "fastCS":
        p = dinv.physics.CompressedSensing(
            m=20,
            fast=True,
            channelwise=True,
            img_shape=img_size,
            device=device,
            rng=rng,
        )
    elif name == "colorize":
        p = dinv.physics.Decolorize(device=device)
        norm = 0.4468
    elif name == "cassi":
        img_size = (7, 37, 31)
        p = dinv.physics.CompressiveSpectralImaging(img_size, device=device, rng=rng)
        norm = 1 / img_size[0]
    elif name == "inpainting":
        p = dinv.physics.Inpainting(
            tensor_size=img_size, mask=0.5, device=device, rng=rng
        )
    elif name == "demosaicing":
        p = dinv.physics.Demosaicing(img_size=img_size, device=device)
        norm = 1.0
    elif name == "MRI":
        img_size = (2, 17, 11)  # C,H,W
        p = MRI(img_size=img_size, device=device)
    elif name == "3DMRI":
        img_size = (2, 5, 17, 11)  # C,D,H,W where D is depth
        p = MRI(img_size=img_size, three_d=True, device=device)
    elif name == "DynamicMRI":
        img_size = (2, 5, 17, 11)  # C,T,H,W where T is time
        p = DynamicMRI(img_size=img_size, device=device)
    elif name == "MultiCoilMRI":
        img_size = (2, 17, 11)  # C,H,W
        n_coils = 7
        maps = torch.ones(
            (1, n_coils, 17, 11), dtype=torch.complex64, device=device
        ) / sqrt(
            n_coils
        )  # B,N,H,W where N is coil dimension
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, device=device)
    elif name == "3DMultiCoilMRI":
        img_size = (2, 5, 17, 11)  # C,D,H,W where D is depth
        n_coils = 15
        maps = torch.ones(
            (1, n_coils, 5, 17, 11), dtype=torch.complex64, device=device
        ) / sqrt(
            n_coils
        )  # B,N,D,H,W where N is coils and D is depth
        p = MultiCoilMRI(coil_maps=maps, img_size=img_size, three_d=True, device=device)
    elif name == "Tomography":
        img_size = (1, 16, 16)
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=img_size[-1], device=device
        )
    elif name == "composition":
        img_size = (3, 16, 16)
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
    elif name == "composition2":
        img_size = (3, 16, 16)
        p1 = dinv.physics.Downsampling(
            img_size=img_size, factor=2, device=device, filter=None
        )
        p2 = dinv.physics.BlurFFT(
            img_size=(3, 8, 8),
            device=device,
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.5)),
        )
        p = p2 * p1
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1, rng=rng))
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
    elif name == "aliased_pansharpen":
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(
            img_size=img_size, device=device, filter=None, use_brovey=False
        )
        norm = 1.4
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_shape=img_size, device=device, rng=rng
        )
    elif name == "singlepixel":
        m = 20
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=False, img_shape=img_size, device=device, rng=rng
        )
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 3.7  # Marcenko-Pastur law, second term is a small n correction
    elif name.startswith("deblur"):
        img_size = (3, 17, 19)
        p = dinv.physics.Blur(
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.25, 0.1), angle=45.0),
            padding=padding,
            device=device,
        )
    elif name == "fftdeblur":
        img_size = (3, 17, 19)
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.bicubic_filter(),
            device=device,
        )
    elif name.startswith("space_deblur"):
        img_size = (3, 20, 13)
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
            )
            * 0.5,
            padding=padding,
        )
    elif name == "hyperspectral_unmixing":
        img_size = (15, 32, 32)  # x (E, H, W)
        p = dinv.physics.HyperSpectralUnmixing(E=15, C=64, device=device)

    elif name.startswith("3Ddeblur"):
        img_size = (1, 7, 6, 8)
        h_size = (1, 1, 4, 3, 5)
        h = torch.rand(h_size)
        h /= h.sum()
        p = dinv.physics.Blur(
            filter=h,
            padding=padding,
            device=device,
        )

    elif name == "aliased_super_resolution":
        img_size = (1, 32, 32)
        factor = 2
        norm = 1.0
        p = dinv.physics.Downsampling(
            img_size=img_size,
            factor=factor,
            padding=padding,
            device=device,
            filter=None,
        )
    elif name.startswith("super_resolution"):
        img_size = (1, 32, 32)
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
    elif name == "complex_compressed_sensing":
        img_size = (1, 8, 8)
        m = 50
        p = dinv.physics.CompressedSensing(
            m=m,
            img_shape=img_size,
            dtype=torch.cdouble,
            device=device,
            compute_inverse=True,
            rng=rng,
        )
        dtype = p.dtype
        norm = (1 + np.sqrt(np.prod(img_size) / m)) ** 2
    elif "radio" in name:
        dtype = torch.cfloat
        img_size = (1, 64, 64)
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
            dtype=torch.float,
            device=device,
            noise_model=dinv.physics.GaussianNoise(0.0, rng=rng),
        )
    elif name == "structured_random":
        img_size = (1, 8, 8)
        p = dinv.physics.StructuredRandom(
            input_shape=img_size, output_shape=img_size, device=device
        )
    elif name == "ptychography_linear":
        img_size = (1, 32, 32)
        dtype = torch.complex64
        norm = 1.32
        p = dinv.physics.PtychographyLinearOperator(
            img_size=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size, norm, dtype


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


def find_phase_retrieval_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.PhaseRetrieval) forward operator.
    """
    if name == "random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.RandomPhaseRetrieval(m=500, img_shape=img_size, device=device)
    elif name == "ptychography":
        img_size = (1, 32, 32)
        p = dinv.physics.Ptychography(
            in_shape=img_size,
            probe=None,
            shifts=None,
            device=device,
        )
    elif name == "structured_random_phase_retrieval":
        img_size = (1, 10, 10)
        p = dinv.physics.StructuredRandomPhaseRetrieval(
            input_shape=img_size, output_shape=img_size, n_layers=2, device=device
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
    p1 = dinv.physics.Inpainting(mask=0.5, tensor_size=imsize, device=device)
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
        assert torch.all((y1 == 0) == (mask == 0))

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

        # Set mask via update_parameters
        physics.update_parameters(mask=mask)
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


@pytest.mark.parametrize("name", OPERATORS)
def test_concatenation(name, device):
    if "pansharpen" in name:  # TODO: fix pansharpening
        return
    physics, imsize, _, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    y = physics(x)
    physics = (
        dinv.physics.Inpainting(
            tensor_size=y.size()[1:], mask=0.5, pixelwise=False, device=device
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
    physics = dinv.physics.RandomPhaseRetrieval(
        m=10, img_shape=(1, 3, 3), device=device
    )
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
    physics = dinv.physics.CompressedSensing(m=10, img_shape=(1, 3, 3), device=device)
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
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("noise_type", NOISES)
def test_noise(device, noise_type):
    r"""
    Tests noise models.
    """
    physics = dinv.physics.DecomposablePhysics(device=device)
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

    physics = dinv.physics.Inpainting(tensor_size=x.shape, mask=mask, device=device)
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
        img_size=(1, x.shape[-2], x.shape[-1]),
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
    physics = dinv.physics.Denoising()
    physics.noise_model = dinv.physics.GaussianNoise(0.1)

    y1 = physics(x)
    y2 = physics(x, sigma=0.2)

    assert physics.noise_model.sigma == 0.2

    physics.noise_model = dinv.physics.PoissonNoise(0.1)

    y1 = physics(x)
    y2 = physics(x, gain=0.2)

    assert physics.noise_model.gain == 0.2

    physics.noise_model = dinv.physics.PoissonGaussianNoise(0.5, 0.3)
    y1 = physics(x)
    y2 = physics(x, sigma=0.2, gain=0.2)

    assert physics.noise_model.gain == 0.2
    assert physics.noise_model.sigma == 0.2


def test_tomography(device):
    r"""
    Tests tomography operator which does not have a numerically precise adjoint.

    :param device: (torch.device) cpu or cuda:x
    """
    PI = 4 * torch.ones(1).atan()
    for normalize in [True, False]:
        for parallel_computation in [True, False]:
            for fan_beam in [True, False]:
                for circle in [True, False]:
                    imsize = (1, 16, 16)
                    physics = dinv.physics.Tomography(
                        img_width=imsize[-1],
                        angles=imsize[-1],
                        device=device,
                        circle=circle,
                        fan_beam=fan_beam,
                        normalize=normalize,
                        parallel_computation=parallel_computation,
                    )

                    x = torch.randn(imsize, device=device).unsqueeze(0)
                    r = (
                        physics.A_adjoint(physics.A(x))
                        * PI.item()
                        / (2 * len(physics.radon.theta))
                    )
                    y = physics.A(r)
                    error = (physics.A_dagger(y) - r).flatten().mean().abs()
                    assert error < 0.2


def test_downsampling_adjointness(device):
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

        paddings = ("valid", "constant", "circular", "reflect", "replicate")

        for pad in paddings:
            for sim in size_im:
                for sfil in size_filt:
                    x = torch.rand(sim)[None].to(device)
                    h = torch.rand(sfil)[None].to(device)

                    physics = dinv.physics.Downsampling(
                        sim, filter=h, padding=pad, device=device
                    )

                    Ax = physics.A(x)
                    y = torch.rand_like(Ax)
                    Aty = physics.A_adjoint(y)

                    Axy = torch.sum(Ax * y)
                    Atyx = torch.sum(Aty * x)

                    assert torch.abs(Axy - Atyx) < 1e-3


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

    def roll(x: torch.Tensor, shift: List[int], dim: List[int]) -> torch.Tensor:
        if len(shift) != len(dim):
            raise ValueError("len(shift) must match len(dim)")

        for s, d in zip(shift, dim):
            x = roll_one_dim(x, s, d)

        return x

    def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
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

    def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
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

    x = torch.ones(1, channels, *imsize[-2:])
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

    assert torch.all(x_hat[:, 0].squeeze() == torch.tensor([1.0, 0.0]))
    assert torch.all(x_hat[:, 1].squeeze() == torch.tensor([0.0, 1.0]))
