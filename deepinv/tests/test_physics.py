import pytest
import torch
import numpy as np

import deepinv as dinv


# Linear forward operators to test (make sure they appear in find_operator as well)
OPERATORS = [
    "CS",
    "fastCS",
    "inpainting",
    "denoising",
    "deblur_fft",
    "deblur",
    "singlepixel",
    "fast_singlepixel",
    "super_resolution",
    "MRI",
    "pansharpen",
]
NONLINEAR_OPERATORS = ["haze", "blind_deblur", "lidar"]

NOISES = [
    "Gaussian",
    "Poisson",
    "PoissonGaussian",
    "UniformGaussian",
    "Uniform",
    "Neighbor2Neighbor",
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
    elif name == "pansharpen":
        img_size = (3, 30, 32)
        p = dinv.physics.Pansharpen(img_size=img_size, device=device)
        norm = 0.4
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
    elif name == "deblur":
        img_size = (3, 17, 19)
        p = dinv.physics.Blur(
            dinv.physics.blur.gaussian_blur(sigma=(2, 0.1), angle=45.0), device=device
        )
    elif name == "deblur_fft":
        img_size = (3, 17, 19)
        p = dinv.physics.BlurFFT(
            img_size=img_size,
            filter=dinv.physics.blur.gaussian_blur(sigma=(0.1, 0.5), angle=45.0),
            device=device,
        )
    elif name == "super_resolution":
        img_size = (1, 32, 32)
        factor = 2
        norm = 1 / factor**2
        p = dinv.physics.Downsampling(img_size=img_size, factor=factor, device=device)
    else:
        raise Exception("The inverse problem chosen doesn't exist")
    return p, img_size, norm


def find_nonlinear_operator(name, device):
    r"""
    Chooses operator

    :param name: operator name
    :param device: (torch.device) cpu or cuda
    :return: (deepinv.physics.Physics) forward operator.
    """
    if name == "blind_deblur":
        x = dinv.utils.TensorList(
            [
                torch.randn(1, 3, 16, 16, device=device),
                torch.randn(1, 1, 3, 3, device=device),
            ]
        )
        p = dinv.physics.BlindBlur(kernel_size=3)
    elif name == "haze":
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
    physics, imsize, _ = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    error = physics.adjointness_test(x).abs()
    assert error < 1e-3


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
    physics, imsize, norm_ref = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)
    norm = physics.compute_norm(x)
    assert torch.abs(norm - norm_ref) < 0.2


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
    Tests if a linear physics operator has a well defined pseudoinverse.
    Warning: Only test linear operators, non-linear ones will fail the test.

    :param name: operator name (see find_operator)
    :param imsize: (tuple) image size tuple in (C, H, W)
    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is less than 1e-3
    """
    physics, imsize, _ = find_operator(name, device)
    x = torch.randn(imsize, device=device).unsqueeze(0)

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
    physics = dinv.physics.MRI(mask=None, device=device, acceleration_factor=4)
    x = torch.randn((2, 320, 320), device=device).unsqueeze(0)
    x2 = physics.A_adjoint(physics.A(x))
    assert x2.shape == x.shape

    physics = dinv.physics.MRI(mask=None, device=device, acceleration_factor=8, seed=0)
    y1 = physics.A(x)
    physics.reset()
    y2 = physics.A(x)
    if y1.shape == y2.shape:
        error = (y1.abs() - y2.abs()).flatten().mean().abs()
        assert error > 0.0


def choose_noise(noise_type):
    gain = 0.1
    sigma = 0.1
    if noise_type == "PoissonGaussian":
        noise_model = dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain)
    elif noise_type == "Gaussian":
        noise_model = dinv.physics.GaussianNoise(sigma)
    elif noise_type == "UniformGaussian":
        noise_model = dinv.physics.UniformGaussianNoise(
            sigma=sigma
        )  # This is equivalent to GaussianNoise when sigma is fixed
    elif noise_type == "Uniform":
        noise_model = dinv.physics.UniformNoise(a=gain)
    elif noise_type == "Poisson":
        noise_model = dinv.physics.PoissonNoise(gain)
    elif noise_type == "Neighbor2Neighbor":
        noise_model = dinv.physics.PoissonNoise(gain)
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
    x = torch.ones((1, 12, 7), device=device).unsqueeze(0)

    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    assert y1.shape == x.shape

    if noise_type == "UniformGaussian":
        physics.reset()
        y2 = physics(x)
        error = (y1 - y2).flatten().abs().sum()
        assert error > 0.0


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


def test_reset_noise(device):
    r"""
    Tests that the reset function works.

    :param device: (torch.device) cpu or cuda:x
    :return: asserts error is > 0
    """
    physics = dinv.physics.DecomposablePhysics()
    physics.noise_model = dinv.physics.UniformGaussianNoise(
        sigma=None
    )  # Should be 20/255 (to check)
    x = torch.ones((1, 12, 7), device=device).unsqueeze(0)

    y1 = physics(
        x
    )  # Note: this works but not physics.A(x) because only the noise is reset (A does not encapsulate noise)
    physics.reset()
    y2 = physics(x)
    error = (y1 - y2).flatten().abs().sum()
    assert error > 0.0


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


from deepinv.physics.blur import gaussian_blur


def test_gaussian_blur_sigma_conversion():
    # Test when sigma is a single value (int or float)
    sigma = 2.0
    result = gaussian_blur(sigma)
    expected_size = (1, 1, 15, 15)
    assert result.size() == expected_size

    # Test when sigma is a tuple
    sigma = (2.0, 2.0)
    result = gaussian_blur(sigma)
    expected_size = (1, 1, 15, 15)
    assert result.size() == expected_size


###################################################################
# Test forward  :


from deepinv.physics import LinearPhysics, GaussianNoise


# Test case 1: Multiplying LinearPhysics instances
def test_linear_physics_mul():
    r"""
    Test the multiplication of two LinearPhysics instances.

    This test validates the functionality of concatenating two LinearPhysics models via multiplication.
    :param A1: Lambda function representing the first linear operation (identity function in this case).
    :param A2: Lambda function representing the second linear operation (doubling the input in this case).
    :param physics1: LinearPhysics initialized with A1 and GaussianNoise with sigma=0.1.
    :param physics2: LinearPhysics initialized with A2 and GaussianNoise with sigma=0.2.
    :return: Ensures that the concatenated model is an instance of LinearPhysics, correctly combines the effects of A1 and A2, and inherits the noise model from physics1.
    """

    A1 = lambda x: x
    A2 = lambda x: x * 2
    physics1 = LinearPhysics(A=A1, noise_model=GaussianNoise(sigma=0.1))
    physics2 = LinearPhysics(A=A2, noise_model=GaussianNoise(sigma=0.2))

    result_physics = physics1 * physics2

    assert isinstance(result_physics, LinearPhysics)
    assert result_physics.A(torch.tensor(3.0)) == 6.0  # A1(A2(x)) = x * 2 * 2 = 4x
    assert result_physics.noise_model.sigma == 0.1


from deepinv.physics.blur import bilinear_filter
from deepinv.physics.blur import bicubic_filter


def test_bilinear_filter_values():
    r"""
    This test checks the validity of the bilinear filter kernel by ensuring that the sum of all its values equals 1.

    :param kernel: A bilinear filter kernel generated by the bilinear_filter() function.
    :return: Asserts that the sum of all values in the kernel equals 1 with a tolerance of 1e-6, confirming it is a valid bilinear filter.
    """

    kernel = bilinear_filter()

    assert torch.isclose(
        torch.sum(kernel), torch.tensor(1.0), atol=1e-6
    ), "The sum of kernel values should equal 1"


def test_A_dagger_convergence():
    r"""

    :param A: Lambda function representing a simple linear operation (identity function in this test).
    :param A_adjoint: Lambda function representing the adjoint of A.
    :param physics: An instance of the LinearPhysics class initialized with A and A_adjoint.
    :param x_true: Original input tensor.
    :param y: Measurement tensor obtained by applying A to x_true.
    :return: Asserts that the difference between the reconstructed tensor and the original tensor is less than a predefined threshold, indicating successful convergence.
    """

    A = lambda x: x
    A_adjoint = lambda y: y
    physics = LinearPhysics(A=A, A_adjoint=A_adjoint, max_iter=10, tol=1e-4)

    x_true = torch.randn(1, 10, 10)
    y = physics.A(x_true)

    x_reconstructed = physics.A_dagger(y)

    assert (
        torch.norm(x_reconstructed - x_true) < 1.0
    ), "The method should converge to a reasonable solution"


def test_bicubic_filter_values():
    r"""
    Test the bicubic filter's kernel values to ensure they are non-zero and sum up to 1.

    :param kernel: A bicubic filter kernel generated by the bicubic_filter() function.
    :return: Asserts that the kernel contains non-zero values and that the sum of all values equals 1 with a tolerance of 1e-6.
    """

    kernel = bicubic_filter()

    assert torch.sum(kernel) > 0
    assert torch.isclose(torch.sum(kernel), torch.tensor(1.0), atol=1e-6)


from deepinv.physics.blur import Blur


def test_prox_l2_fft_circular():
    r"""
    Test the prox_l2 function of the Blur class with FFT and circular padding.

    :param blur: An instance of the Blur class initialized with circular padding and a specific filter.
    :param z: Input tensor representing the signal.
    :param y: Input tensor representing the measurements.
    :param gamma: Hyperparameter of the proximal operator.
    :return: Asserts that the shape of the output from prox_l2 matches the shape of the input tensor z, indicating correct processing.
    """

    blur = Blur(padding="circular", filter=torch.ones((1, 1, 5, 5)) / 25)

    z = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    gamma = 0.1

    result = blur.prox_l2(z, y, gamma)
    assert result.shape == z.shape
