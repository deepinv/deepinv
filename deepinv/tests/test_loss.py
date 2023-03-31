import pytest

import math
import torch

import deepinv as dinv
from deepinv.loss.regularisers import JacobianSpectralNorm, FNEJacobianSpectralNorm

@pytest.fixture
def device():
    return dinv.device


@pytest.fixture
def toymatrix():
    w = 50
    A = torch.diag(torch.Tensor(range(1, w + 1)))
    return A


def test_jacobian_spectral_values(toymatrix):

    # Define the Jacobian regularisers we want to check
    reg_l2 = JacobianSpectralNorm(max_iter=100, tol=1e-3, eval_mode=False, verbose=True)
    reg_FNE_l2 = FNEJacobianSpectralNorm(max_iter=100, tol=1e-3, eval_mode=False, verbose=True)

    # Setup our toy example; here y = A@x
    x_detached = torch.randn_like(toymatrix).requires_grad_()
    out = toymatrix @ x_detached

    def model(x):
        return toymatrix @ x

    regl2 = reg_l2(out, x_detached)
    regfnel2 = reg_FNE_l2(out, x_detached, model, interpolation=False)

    assert(math.isclose(regl2.item(), toymatrix.size(0), rel_tol=1e-3))
    assert(math.isclose(regfnel2.item(), 2*toymatrix.size(0)-1, rel_tol=1e-3))
