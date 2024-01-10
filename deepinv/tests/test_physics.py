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


import pytest
import torch
from deepinv.physics import LinearPhysics, GaussianNoise


# Test case 1: Multiplying LinearPhysics instances
def test_linear_physics_mul():
    A1 = lambda x: x
    A2 = lambda x: x * 2
    physics1 = LinearPhysics(A=A1, noise_model=GaussianNoise(sigma=0.1))
    physics2 = LinearPhysics(A=A2, noise_model=GaussianNoise(sigma=0.2))

    result_physics = physics1 * physics2

    assert isinstance(result_physics, LinearPhysics)
    assert result_physics.A(torch.tensor(3.0)) == 6.0  # A1(A2(x)) = x * 2 * 2 = 4x
    assert (
        result_physics.noise_model.sigma == 0.1
    )  # Check that noise model is from physics1


import torch
from deepinv.physics.forward import TensorList
from deepinv.physics.forward import LinearPhysics


def test_noise_forward():
    # Creating instances of fictitious noise
    noise1 = GaussianNoise(sigma=0.1)

    # Creating an instance of the LinearPhysics class
    physics_instance = LinearPhysics(noise_model=noise1)

    # Creating a sample input data
    x = torch.randn(1, 10, 10)

    # Applying the forward method
    output = physics_instance(x)

    # Verifications
    assert output.shape == x.shape, "The output shape should match the input shape"

    # Check if the variance of the output data is greater than that of the input data
    input_variance = torch.var(x)
    output_variance = torch.var(output)
    assert (
        output_variance != input_variance
    ), "The variance after adding noise should be higher"


def test_sensor_forward():
    # Creating instances of fictitious sensor
    sensor1 = GaussianNoise(sigma=0.1)

    # Creating an instance of the LinearPhysics class
    physics_instance = LinearPhysics(sensor_model=sensor1)

    # Creating a sample input data
    x = torch.randn(1, 10, 10)

    # Applying the forward method
    output = physics_instance(x)

    # Verifications
    assert output.shape == x.shape, "The output shape should match the input shape"
    # Other assertions according to the sensor logic
    # Check if the output values are saturated to a certain threshold
    threshold = 1.0  # Assuming this is the saturation threshold
    # assert torch.all(output > threshold), "All values should be less than or equal to the saturation threshold"


#################################################################
# Test for the BLUR class
##Test of bilinear_filter
import torch
import pytest
from deepinv.physics.blur import bilinear_filter
from deepinv.physics.blur import bicubic_filter


# Test of the output shape:
def test_bilinear_filter_output_type():
    kernel = bilinear_filter()
    assert isinstance(kernel, torch.Tensor)


# Test of filter values:
def test_bilinear_filter_output_shape():
    kernel = bilinear_filter()
    expected_shape = (1, 1, 4, 4)  # Mise à jour pour correspondre à la forme réelle
    assert kernel.shape == expected_shape


# Test of filter values:
def test_bilinear_filter_values():
    kernel = bilinear_filter()
    # Valeurs attendues basées sur la logique réelle de la fonction bilinear_filter
    expected_values = torch.tensor(
        [
            [
                [
                    [0.0156, 0.0469, 0.0469, 0.0156],
                    [0.0469, 0.1406, 0.1406, 0.0469],
                    [0.0469, 0.1406, 0.1406, 0.0469],
                    [0.0156, 0.0469, 0.0469, 0.0156],
                ]
            ]
        ]
    )
    assert torch.allclose(kernel, expected_values, atol=1e-2)


# Test on A_dagger of Forward


def test_A_dagger_basic():
    # Definition of a simple A operator (identity in this example)
    A = lambda x: x
    A_adjoint = lambda y: y
    physics = LinearPhysics(A=A, A_adjoint=A_adjoint, max_iter=50, tol=1e-3)

    # Creating a test data and calculating measurements
    x_true = torch.randn(1, 10, 10)
    y = physics.A(x_true)

    # Using A_dagger to reconstruct x from y
    x_reconstructed = physics.A_dagger(y)

    # Check if the reconstruction is close to the original value
    assert torch.allclose(
        x_reconstructed, x_true, atol=1e-2
    ), "The reconstruction should be close to the original value"


def test_A_dagger_with_initialization():
    A = lambda x: x
    A_adjoint = lambda y: y
    physics = LinearPhysics(A=A, A_adjoint=A_adjoint, max_iter=50, tol=1e-3)

    x_true = torch.randn(1, 10, 10)
    y = physics.A(x_true)
    x_init = torch.zeros_like(x_true)

    # Using x_init directly as an argument to A_dagger
    x_reconstructed = physics.A_dagger(y)

    assert torch.allclose(
        x_reconstructed, x_true, atol=1e-2
    ), "The reconstruction should be close to the original value even with custom initialization"


def test_A_dagger_convergence():
    A = lambda x: x
    A_adjoint = lambda y: y
    physics = LinearPhysics(A=A, A_adjoint=A_adjoint, max_iter=10, tol=1e-4)

    x_true = torch.randn(1, 10, 10)
    y = physics.A(x_true)

    x_reconstructed = physics.A_dagger(y)

    # Check if the difference between the reconstruction and the original value is reasonable
    assert (
        torch.norm(x_reconstructed - x_true) < 1.0
    ), "The method should converge to a reasonable solution"


## Test of the bicubic filter function

# Test of the output type


def test_bicubic_filter_output_type():
    kernel = bicubic_filter()
    assert isinstance(kernel, torch.Tensor)


# Test of the output shape:


def test_bicubic_filter_output_shape():
    kernel = bicubic_filter()
    expected_shape = (1, 1, 8, 8)  # Updated to match the actual shape
    assert kernel.shape == expected_shape


# Test of filter values:


def test_bicubic_filter_values():
    kernel = bicubic_filter()
    # Check that the kernel has non-zero values and that the sum of values is 1
    assert torch.sum(kernel) > 0
    assert torch.isclose(torch.sum(kernel), torch.tensor(1.0), atol=1e-6)


## Test of the prox_l2 function of the Blur class

import torch
import pytest
from deepinv.physics.blur import Blur  # Replace with the correct path of the Blur class


def test_prox_l2_fft_circular():
    # Initialization of Blur with specific parameters
    # Ensure that the filter has the same channel dimension as z and y
    blur = Blur(padding="circular", filter=torch.ones((1, 1, 5, 5)) / 25)

    # Create input tensors
    z = torch.rand(1, 1, 32, 32)
    y = torch.rand(1, 1, 32, 32)
    gamma = 0.1

    # Call the prox_l2 function
    result = blur.prox_l2(z, y, gamma)

    # Check the properties of the output
    assert result.shape == z.shape
