import pytest
import torch
import numpy as np
from deepinv.physics.forward import adjoint_function
import deepinv as dinv
from deepinv.optim.data_fidelity import L2

# Linear forward operators to test (make sure they appear in find_operator as well)
# We do not include operators for which padding is involved, they are tested separately
OPERATORS = [
    "CS",
    "fastCS",
    "inpainting",
    "denoising",
    "fftdeblur",
    "singlepixel",
    "deblur_valid",
    "deblur_circular",
    "deblur_reflect",
    "deblur_replicate",
    "space_deblur_valid",
    "space_deblur_circular",
    "space_deblur_reflect",
    "space_deblur_replicate",
    "super_resolution_valid",
    "super_resolution_circular",
    "super_resolution_reflect",
    "super_resolution_replicate",
    "aliased_super_resolution",
    "fast_singlepixel",
    "MRI",
    "aliased_pansharpen",
    "pansharpen_valid",
    "pansharpen_circular",
    "pansharpen_reflect",
    "pansharpen_replicate",
    "complex_compressed_sensing",
]

NONLINEAR_OPERATORS = ["haze", "lidar"]

NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "Neighbor2Neighbor",
    "LogPoisson",
]


def find_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    img_size = (3, 16, 8)
    norm = 1
    dtype = torch.float
    padding = None
    paddings = ["valid", "circular", "reflect", "replicate"]
    for p in paddings:
        if p in name:
            padding = p
            break

    if name == "CS":
        m = 30
        p = dinv.physics.CompressedSensing(m=m, img_shape=img_size, device=device)
        norm = (
            1 + np.sqrt(np.prod(img_size) / m)
        ) ** 2 - 0.75  # Marcenko-Pastur law, second term is a small n correction
    elif name == "fastCS":
        p = dinv.physics.CompressedSensing(
            m=20, fast=True, channelwise=True, img_shape=img_size, device=device
        )
    elif name == "inpainting":
        p = dinv.physics.Inpainting(tensor_size=img_size, mask=0.5, device=device)
    elif name == "MRI":
        img_size = (2, 16, 8)
        p = dinv.physics.MRI(mask=torch.ones(img_size[-2], img_size[-1]), device=device)
    elif name == "Tomography":
        img_size = (1, 16, 16)
        p = dinv.physics.Tomography(
            img_width=img_size[-1], angles=img_size[-1], device=device
        )
    elif name == "denoising":
        p = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.1))
    elif name.startswith("pansharpen"):
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(
            img_size=img_size, device=device, padding=padding, filter="bilinear"
        )
        norm = 0.4
    elif name == "aliased_pansharpen":
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(img_size=img_size, device=device, filter=None)
        norm = 1.4
    elif name == "fast_singlepixel":
        p = dinv.physics.SinglePixelCamera(
            m=20, fast=True, img_shape=img_size, device=device
        )
    elif name == "singlepixel":
        m = 20
        p = dinv.physics.SinglePixelCamera(
            m=m, fast=False, img_shape=img_size, device=device
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
        h = torch.cat([h, h], dim=0)
        p = dinv.physics.SpaceVaryingBlur(
            filters=h,
            multipliers=torch.ones(
                (
                    2,
                    1,
                )
                + img_size,
                device=device,
            )
            * 0.5,
            padding=padding,
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
        )
    elif name == "complex_compressed_sensing":
        img_size = (1, 8, 8)
        m = 50
        p = dinv.physics.CompressedSensing(
            m=m, img_shape=img_size, dtype=torch.cfloat, device=device
        )
        dtype = p.dtype
        norm = (1 + np.sqrt(np.prod(img_size) / m)) ** 2
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size, norm, dtype


def find_nonlinear_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
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


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_adjointness(name, device):
    r"""
    Tests if a linear forward operator has a well defined adjoint.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts adjointness
    """
    physics, imsize, _, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    error = physics.adjointness_test(x).abs()
    assert error < 1e-3

    if (
        "pansharpen" in name
    ):  # automatic adjoint does not work for inputs that are not torch.tensors
        return
    f = adjoint_function(physics.A, x.shape, x.device, x.dtype)

    y = physics.A(x)
    error2 = (f(y) - physics.A_adjoint(y)).flatten().mean().abs()

    assert error2 < 1e-3


@pytest.mark.parametrize("name", OPERATORS)
def test_operators_norm(name, device):
    r"""
    Tests if a linear physics operator has a norm close to 1.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts norm is in (.8,1.2)
    """
    if name == "singlepixel" or name == "CS":
        device = torch.device("cpu")

    torch.manual_seed(0)
    physics, imsize, norm_ref, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)
    norm = physics.compute_norm(x, max_iter=1000, tol=1e-6)
    bound = 1e-2
    # if theoretical bound relies on Marcenko-Pastur law, or if pansharpening, relax the bound
    if (
        name in ["singlepixel", "CS", "complex_compressed_sensing"]
        or "pansharpen" in name
        or "space_deblur"
    ):
        bound = 0.2
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
def test_pseudo_inverse(name, device):
    r"""
    Tests if a linear physics operator has a well-defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize, _, dtype = find_operator(name, device)
    x = torch.randn(imsize, device=device, dtype=dtype).unsqueeze(0)

    r = physics.A_adjoint(physics.A(x))
    y = physics.A(r)
    error = (physics.A_dagger(y) - r).flatten().mean().abs()
    assert error < 0.01


def test_MRI(device):
    r"""
    Test MRI function

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    mask = torch.ones((32, 32), device=device)
    physics = dinv.physics.MRI(mask=mask, device=device, acceleration_factor=4)
    x = torch.randn((2, 32, 32), device=device).unsqueeze(0)
    y1 = physics.A(x)
    x2 = physics.A_adjoint(y1)
    assert x2.shape == x.shape

    generator = dinv.physics.generator.AccelerationMaskGenerator(
        (32, 32), device=device
    )
    mask = generator.step()
    y2 = physics.A(x, **mask)
    if y1.shape == y2.shape:
        error = (y1.abs() - y2.abs()).flatten().mean().abs()
        assert error > 0.0


def test_phase_retrieval(device):
    r"""
    Tests to ensure the phase retrieval operator is behaving as expected.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    x = torch.randn((1, 1, 10, 10), dtype=torch.cfloat, device=device)
    physics = dinv.physics.RandomPhaseRetrieval(
        m=500, img_shape=(1, 10, 10), device=device
    )
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


def test_linear_physics_Avjp(device):
    r"""
    Tests if the gradient computed with A_vjp method of linear physics is consistent with the autograd gradient.

    :param device: (torch.device) cpu or cuda:x
    :return: assertion error if the relative difference between the two gradients is more than 1e-5
    """
    # essential to enable autograd
    torch.set_grad_enabled(True)
    x = torch.randn((1, 1, 3, 3), dtype=torch.float, device=device, requires_grad=True)
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


def choose_noise(noise_type):
    gain = 0.1
    sigma = 0.1
    mu = 0.2
    N0 = 1024.0
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
    else:
        raise Exception("Noise model not found")

    return noise_model


@pytest.mark.parametrize("noise_type", NOISES)
def test_noise(device, noise_type):
    r"""
    Tests noise models.
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = choose_noise(noise_type)
    x = torch.ones((1, 3, 2), device=device).unsqueeze(0)

    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    assert y1.shape == x.shape


def test_noise_domain(device):
    r"""
    Tests that there is no noise outside the domain of the measurement operator, i.e. that in y = Ax+n, we have
    n=0 where Ax=0.
    """
    x = torch.ones((3, 12, 7), device=device).unsqueeze(0)
    mask = torch.ones_like(x[0])
    # mask[:, x.shape[-2]//2-3:x.shape[-2]//2+3, x.shape[-1]//2-3:x.shape[-1]//2+3] = 0
    mask[0, 0, 0] = 0
    mask[1, 1, 1] = 0
    mask[2, 2, 2] = 0

    physics = dinv.physics.Inpainting(tensor_size=x.shape, mask=mask, device=device)
    physics.noise_model = choose_noise("Gaussian")
    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
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
    for circle in [True, False]:
        imsize = (1, 16, 16)
        physics = dinv.physics.Tomography(
            img_width=imsize[-1], angles=imsize[-1], device=device, circle=circle
        )

        x = torch.randn(imsize, device=device).unsqueeze(0)
        r = physics.A_adjoint(physics.A(x))
        y = physics.A(r)
        error = (physics.A_dagger(y) - r).flatten().mean().abs()
        assert error < 0.2
