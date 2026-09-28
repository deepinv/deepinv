"""Check the PGD update independently of the legacy optimizer."""

from functools import partial

import pytest
import torch

import deepinv as dinv
from .conftest import wavelet_prior

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
    stepsize, prior_weight = 0.05, 0.3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=prior_type(),
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=1,
    )
    x = init if use_init else 0.8 * y
    # The L2 gradient for A = 0.8 I is 0.8 * (0.8 * x - y).
    with torch.no_grad():
        expected = prior_type().prox(
            x - stepsize * 0.8 * (0.8 * x - y), gamma=stepsize * prior_weight
        )
        result, _ = model(y, physics, init=init if use_init else None)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("use_init", [False, True])
@pytest.mark.parametrize("prior_type", NONDIFFERENTIABLE_PRIORS)
def test_pgd_nondifferentiable_priors(optim_problem, prior_type, use_init):
    physics, y, init = optim_problem
    stepsize, prior_weight = 0.05, 0.3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=prior_type(),
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=1,
    )
    x = init if use_init else 0.8 * y
    with torch.no_grad():
        expected = prior_type().prox(
            x - stepsize * 0.8 * (0.8 * x - y), gamma=stepsize * prior_weight
        )
        result, _ = model(y, physics, init=init if use_init else None)
    torch.testing.assert_close(result, expected)


def test_pgd_metrics(optim_problem, capsys):
    physics, y, init = optim_problem
    init.requires_grad_()
    stepsize, prior_weight = 0.05, 0.3
    metrics = (dinv.loss.MSE(), dinv.loss.MAE())
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=2,
        show_progress_bar=True,
        metrics=metrics,
    )

    result, params = model(y, physics, init=init, x_gt=init.detach())
    assert result.requires_grad
    assert params["stepsize"] == [stepsize, stepsize]
    with torch.no_grad():
        x = init
        for i in range(2):
            x = (x - stepsize * 0.8 * (0.8 * x - y)) / (1 + stepsize * prior_weight)
            expected_objective = model.data_fidelity(
                x, y, physics
            ) + prior_weight * model.prior(x)
            torch.testing.assert_close(params["objective"][i], expected_objective)
            for metric in metrics:
                torch.testing.assert_close(
                    params["metrics"][type(metric)][i], metric(x, init.detach())
                )
        torch.testing.assert_close(result, x)

    progress = capsys.readouterr().err
    assert "100%" in progress
    assert "MSE" not in progress
    assert all(not value.requires_grad for value in params["objective"])
    assert all(
        not value.requires_grad
        for history in params["metrics"].values()
        for value in history
    )
    model.max_iter = 1
    _, second_params = model(y, physics, init=init, x_gt=init.detach())
    assert len(second_params["objective"]) == 1
    assert all(len(history) == 1 for history in second_params["metrics"].values())
    assert all(len(history) == 2 for history in params["metrics"].values())


def test_pgd_objective_without_metrics(optim_problem):
    physics, y, _ = optim_problem
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(), prior=dinv.optim.ZeroPrior(), max_iter=2
    )

    result, params = model(y, physics)

    assert torch.isfinite(result).all()
    assert params["metrics"] == {}
    assert len(params["objective"]) == 2
    torch.testing.assert_close(
        params["objective"][-1], model.data_fidelity(result, y, physics)
    )


@pytest.mark.parametrize("prior_type", [dinv.optim.ZeroPrior, dinv.optim.Tikhonov])
def test_pgd_armijo_backtracking(optim_problem, prior_type):
    physics, y, init = optim_problem
    prior_weight = 0.3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=prior_type(),
        stepsize=10.0,
        prior_weight=prior_weight,
        max_iter=2,
        backtracking="armijo",
    )

    with torch.no_grad():
        result, params = model(y, physics, init=init)
        x = init
        for stepsize, recorded_objective in zip(
            params["stepsize"], params["objective"], strict=True
        ):
            objective_prev = model.data_fidelity(
                x, y, physics
            ) + prior_weight * model.prior(x)
            x_next = prior_type().prox(
                x - stepsize * model.data_fidelity.grad(x, y, physics),
                gamma=stepsize * prior_weight,
            )
            objective_next = model.data_fidelity(
                x_next, y, physics
            ) + prior_weight * model.prior(x_next)
            torch.testing.assert_close(recorded_objective, objective_next)
            squared_norm = (x_next - x).abs().square().flatten(1).sum(1).mean()
            assert (
                objective_prev - objective_next
            ).mean() >= 0.1 / stepsize * squared_norm
            x = x_next
        torch.testing.assert_close(result, x)
    assert len(params["stepsize"]) == 2
    if prior_type is dinv.optim.ZeroPrior:
        assert params["stepsize"][0] < model.stepsize
