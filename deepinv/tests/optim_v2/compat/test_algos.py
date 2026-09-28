"""Check PGD against the legacy optimizer for representative prior and fidelity terms."""

from functools import partial

import pytest
import torch

import deepinv as dinv


@pytest.mark.parametrize(
    ("prior_type", "fidelity_type"),
    [
        pytest.param(dinv.optim.Tikhonov, dinv.optim.L2, id="Tikhonov"),
        pytest.param(dinv.optim.L1Prior, dinv.optim.L2, id="L1Prior"),
        pytest.param(
            dinv.optim.ZeroPrior,
            lambda: dinv.optim.DataFidelity(
                d=lambda u, y: (u - y).square().flatten(1).sum(1) / 2
            ),
            id="DataFidelity-autograd",
        ),
        pytest.param(dinv.optim.ZeroPrior, dinv.optim.L2, id="L2"),
        pytest.param(dinv.optim.ZeroPrior, dinv.optim.ZeroFidelity, id="ZeroFidelity"),
        pytest.param(
            dinv.optim.ZeroPrior,
            partial(dinv.optim.PoissonLikelihood, bkg=0.1),
            id="Poisson",
        ),
        pytest.param(
            dinv.optim.ZeroPrior,
            partial(dinv.optim.LogPoissonLikelihood, N0=1.0, mu=1.0),
            id="LogPoisson",
        ),
        pytest.param(
            dinv.optim.ZeroPrior, dinv.optim.AmplitudeLoss, id="AmplitudeLoss"
        ),
        pytest.param(dinv.optim.ZeroPrior, dinv.optim.ItohFidelity, id="ItohFidelity"),
        pytest.param(
            dinv.optim.ZeroPrior,
            lambda: dinv.optim.StackedPhysicsDataFidelity(
                [dinv.optim.L2(), dinv.optim.L2()]
            ),
            id="StackedPhysicsDataFidelity",
        ),
    ],
)
@pytest.mark.parametrize(("use_init", "max_iter"), [(False, 1), (True, 5)])
def test_pgd_compat(optim_problem, prior_type, fidelity_type, use_init, max_iter):
    physics, y, init = optim_problem
    fidelity = fidelity_type()
    if isinstance(fidelity, dinv.optim.ItohFidelity):
        physics = dinv.physics.SpatialUnwrapping()
    elif isinstance(fidelity, dinv.optim.StackedPhysicsDataFidelity):
        physics = dinv.physics.StackedLinearPhysics(
            [dinv.physics.Denoising(), dinv.physics.Denoising()]
        )
        y = dinv.utils.TensorList([y, 2 * y])

    params = dict(stepsize=0.05, lambda_reg=0.2, max_iter=max_iter)
    old = dinv.optim.PGD(prior=prior_type(), data_fidelity=fidelity, **params)
    new = dinv.optim_v2.PGD(prior=prior_type(), data_fidelity=fidelity_type(), **params)
    assert isinstance(new, dinv.models.Reconstructor)
    with torch.no_grad():
        expected = old(y, physics, init=init if use_init else None)
        actual = new(y, physics, init=init if use_init else None)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
