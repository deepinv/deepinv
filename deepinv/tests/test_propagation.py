import math

import pytest
import torch

from deepinv.physics import (
    AngularSpectrumPropagation,
    FresnelPropagation,
    LinearPhysics,
    PhaseRetrieval,
    Physics,
)


def test_angular_spectrum_tensor_and_callable_interfaces():
    img_size = (2, 9, 12)
    pixel_size = (2e-6, 3e-6)

    def transfer_function(fx, fy):
        return torch.exp(-1j * 1e-10 * (fx.square() + 2 * fy.square()))

    physics_callable = AngularSpectrumPropagation(
        img_size=img_size,
        H=transfer_function,
        pixel_size=pixel_size,
        dtype=torch.cdouble,
    )
    fx, fy = AngularSpectrumPropagation.frequency_grid(
        img_size,
        pixel_size=pixel_size,
        dtype=torch.float64,
    )
    H = transfer_function(fx, fy)
    physics_tensor = AngularSpectrumPropagation(
        img_size=img_size,
        H=H,
        pixel_size=pixel_size,
        dtype=torch.cdouble,
    )

    x = torch.randn(3, *img_size, dtype=torch.cdouble)
    torch.testing.assert_close(physics_callable.H, H)
    torch.testing.assert_close(physics_callable.A(x), physics_tensor.A(x))
    assert "H" in physics_callable.state_dict()


def test_angular_spectrum_adjoint_and_pseudoinverse():
    img_size = (1, 8, 10)
    fx, fy = AngularSpectrumPropagation.frequency_grid(img_size, dtype=torch.float64)
    H = torch.exp(1j * (0.3 * fx + 0.2 * fy))
    physics = AngularSpectrumPropagation(img_size=img_size, H=H, dtype=torch.cdouble)

    x = torch.randn(2, *img_size, dtype=torch.cdouble)
    v = torch.randn_like(x)
    lhs = (physics.A(x).conj() * v).sum()
    rhs = (x.conj() * physics.A_adjoint(v)).sum()

    torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        physics.A_dagger(physics.A(x)), x, rtol=1e-10, atol=1e-10
    )


def test_angular_spectrum_pseudoinverse_zeros_nullspace():
    H = torch.ones(7, 6, dtype=torch.cfloat)
    H[0, 0] = 0
    physics = AngularSpectrumPropagation(img_size=(1, 7, 6), H=H)
    y = torch.ones(1, 1, 7, 6, dtype=torch.cfloat)

    x = physics.A_dagger(y)

    assert torch.isfinite(x).all()
    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    torch.testing.assert_close(
        spectrum[..., 0, 0], torch.zeros_like(spectrum[..., 0, 0])
    )


def test_fresnel_transfer_function_and_phase_retrieval():
    img_size = (1, 11, 14)
    wavelength = 500e-9
    distance = 0.08
    pixel_size = (4e-6, 5e-6)
    propagation = FresnelPropagation(
        img_size=img_size,
        wavelength=wavelength,
        distance=distance,
        pixel_size=pixel_size,
        dtype=torch.cdouble,
    )
    fx, fy = propagation.frequency_grid(
        img_size,
        pixel_size=pixel_size,
        dtype=torch.float64,
    )
    expected_H = torch.exp(
        -1j * math.pi * wavelength * distance * (fx.square() + fy.square())
    )

    assert isinstance(propagation, LinearPhysics)
    torch.testing.assert_close(propagation.H, expected_H)

    transmission = torch.randn(2, *img_size, dtype=torch.cdouble)
    physics = PhaseRetrieval(B=propagation)
    torch.testing.assert_close(
        physics.A(transmission), propagation.A(transmission).abs().square()
    )


def test_nonlinear_transmission_composes_before_phase_retrieval():
    img_size = (1, 8, 9)
    propagation = FresnelPropagation(
        img_size=img_size,
        wavelength=500e-9,
        distance=0.05,
        pixel_size=4e-6,
    )
    phase_retrieval = PhaseRetrieval(B=propagation)
    wavenumber = 2 * math.pi / 500e-9

    def transmission(parameters):
        projected_delta = parameters[:, 0:1]
        projected_beta = parameters[:, 1:2]
        return torch.exp(
            -1j * wavenumber * projected_delta - wavenumber * projected_beta
        )

    transmission_physics = Physics(A=transmission)
    with pytest.warns(UserWarning, match="composing two physics"):
        physics = phase_retrieval * transmission_physics
    assert not isinstance(physics, LinearPhysics)

    parameters = (torch.rand(2, 2, *img_size[-2:]) * 1e-8).requires_grad_()
    measurements = physics.A(parameters)
    measurements.mean().backward()

    assert measurements.shape == (2, *img_size)
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()


@pytest.mark.parametrize("H_shape", [(2, 8, 7), (1, 1, 8, 7), (8, 6)])
def test_angular_spectrum_rejects_output_or_incompatible_dimensions(H_shape):
    with pytest.raises(ValueError, match="H with shape"):
        AngularSpectrumPropagation(img_size=(1, 8, 7), H=torch.ones(H_shape))
