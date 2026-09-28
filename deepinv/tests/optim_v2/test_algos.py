"""Check the PGD update independently of the legacy optimizer."""

from functools import partial

import pytest
import torch

import deepinv as dinv
from conftest import wavelet_prior

DIFFERENTIABLE_PRIORS = [
    pytest.param(dinv.optim.ZeroPrior, id="ZeroPrior"),
    pytest.param(dinv.optim.Tikhonov, id="Tikhonov"),
    pytest.param(
        lambda: dinv.optim.Prior(g=lambda x, *args: x.square().flatten(1).sum(1) / 2),
        id="Prior-autograd",
    ),
    pytest.param(
        lambda: dinv.optim.PatchPrior(
            negative_patch_log_likelihood=lambda patches: patches.square().sum(-1),
            patch_size=2,
            n_patches=-1,
        ),
        id="PatchPrior",
    ),
]

NONDIFFERENTIABLE_PRIORS = [
    pytest.param(dinv.optim.L1Prior, id="L1Prior"),
    pytest.param(partial(dinv.optim.L12Prior, l2_axis=1), id="L12Prior"),
    pytest.param(dinv.optim.TVPrior, id="TVPrior"),
    pytest.param(dinv.optim.TVL1Prior, id="TVL1Prior"),
    pytest.param(wavelet_prior, id="WaveletPrior"),
]


@pytest.mark.parametrize("use_init", [False, True])
@pytest.mark.parametrize("prior_type", DIFFERENTIABLE_PRIORS)
def test_pgd_differentiable_priors(optim_problem, prior_type, use_init):
    physics, y, init = optim_problem
    stepsize, lambda_reg = 0.05, 0.3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=prior_type(),
        stepsize=stepsize,
        lambda_reg=lambda_reg,
        max_iter=1,
    )
    x = init if use_init else 0.8 * y
    # The L2 gradient for A = 0.8 I is 0.8 * (0.8 * x - y).
    with torch.no_grad():
        expected = prior_type().prox(
            x - stepsize * 0.8 * (0.8 * x - y), gamma=stepsize * lambda_reg
        )
        result = model(y, physics, init=init if use_init else None)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("use_init", [False, True])
@pytest.mark.parametrize("prior_type", NONDIFFERENTIABLE_PRIORS)
def test_pgd_nondifferentiable_priors(optim_problem, prior_type, use_init):
    physics, y, init = optim_problem
    stepsize, lambda_reg = 0.05, 0.3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=prior_type(),
        stepsize=stepsize,
        lambda_reg=lambda_reg,
        max_iter=1,
    )
    x = init if use_init else 0.8 * y
    with torch.no_grad():
        expected = prior_type().prox(
            x - stepsize * 0.8 * (0.8 * x - y), gamma=stepsize * lambda_reg
        )
        result = model(y, physics, init=init if use_init else None)
    torch.testing.assert_close(result, expected)
