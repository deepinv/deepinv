"""Shared PGD test setup and a small deterministic problem."""

import pytest
import torch

import deepinv as dinv


def wavelet_prior():
    pytest.importorskip("ptwt", reason="WaveletPrior requires ptwt")
    pytest.importorskip("pywt", reason="WaveletPrior requires PyWavelets")
    return dinv.optim.WaveletPrior(wv="db1", level=1)


@pytest.fixture
def optim_problem(device):
    """Positive batched images avoid singularities in likelihood gradients."""
    y = torch.linspace(0.5, 1.5, 32, dtype=torch.float64, device=device).reshape(
        2, 1, 4, 4
    )
    physics = dinv.physics.LinearPhysics(
        A=lambda x: 0.8 * x, A_adjoint=lambda y: 0.8 * y
    )
    return physics, y, y.flip(-1).clone()
