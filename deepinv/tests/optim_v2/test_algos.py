"""Check the PGD update independently of the legacy optimizer."""

from functools import partial

import pytest
import torch
from deepinv.optim_v2 import Armijo
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
        result = model(y, physics, init=init if use_init else None)
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
        result = model(y, physics, init=init if use_init else None)
    torch.testing.assert_close(result, expected)


def test_pgd_metrics(optim_problem, capsys):
    physics, y, init = optim_problem
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

    result, diag = model(y, physics, init=init, x_gt=init.detach(), get_diagnostic=True)
    batch_size = y.shape[0]
    torch.testing.assert_close(diag["stepsize"], torch.full((batch_size, 2), stepsize))
    torch.testing.assert_close(
        diag["prior_weight"], torch.full((batch_size, 2), prior_weight)
    )
    with torch.no_grad():
        x = init
        for i in range(2):
            x = (x - stepsize * 0.8 * (0.8 * x - y)) / (1 + stepsize * prior_weight)
            expected_objective = model.data_fidelity(
                x, y, physics
            ) + prior_weight * model.prior(x)
            torch.testing.assert_close(
                diag["objective"][:, i + 1], expected_objective.cpu()
            )
            for metric in metrics:
                torch.testing.assert_close(
                    diag["metrics"][type(metric).__name__][:, i + 1],
                    metric(x, init.detach()).cpu(),
                )
        torch.testing.assert_close(result, x)

    progress = capsys.readouterr().err
    assert "100%" in progress
    assert "MSE" not in progress
    assert not diag["objective"].requires_grad
    assert all(not values.requires_grad for values in diag["metrics"].values())
    model.max_iter = 1
    _, second_diag = model(
        y, physics, init=init, x_gt=init.detach(), get_diagnostic=True
    )
    assert second_diag["objective"].shape[1] == 2  # F(x_0), F(x_1)
    assert all(values.shape[1] == 2 for values in second_diag["metrics"].values())
    assert all(values.shape[1] == 3 for values in diag["metrics"].values())


def test_pgd_objective_without_metrics(optim_problem):
    physics, y, _ = optim_problem
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(), prior=dinv.optim.ZeroPrior(), max_iter=2
    )

    result, diag = model(y, physics, get_diagnostic=True)

    assert torch.isfinite(result).all()
    assert diag["metrics"] == {}
    assert diag["objective"].shape == (y.shape[0], 3)  # F(x_0), F(x_1), F(x_2)
    torch.testing.assert_close(
        diag["objective"][:, -1], model.data_fidelity(result, y, physics).cpu()
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
        backtracking=Armijo(),
    )

    with torch.no_grad():
        result, diag = model(y, physics, init=init, get_diagnostic=True)
        x = init
        for k in range(diag["stepsize"].shape[1]):
            stepsize = diag["stepsize"][0, k].item()  # the same for the whole batch
            recorded_objective = diag["objective"][:, k + 1]
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
            torch.testing.assert_close(recorded_objective, objective_next.cpu())
            change = (x_next - x).reshape(x_next.shape[0], -1)
            squared_norm = torch.linalg.vector_norm(change, dim=1).square().mean()
            assert (
                objective_prev - objective_next
            ).mean() >= 0.1 / stepsize * squared_norm
            x = x_next
        torch.testing.assert_close(result, x)
    assert diag["stepsize"].shape[1] == 2
    if prior_type is dinv.optim.ZeroPrior:
        assert diag["stepsize"][0, 0] < model.stepsize


@pytest.mark.parametrize("schedule_type", ["iterable", "callable"])
def test_pgd_parameter_schedules(optim_problem, schedule_type):
    physics, y, init = optim_problem
    stepsizes = [y.new_tensor(0.05), y.new_tensor(0.02)]
    prior_weights = [y.new_tensor(0.3), y.new_tensor(0.1)]
    calls = []
    if schedule_type == "iterable":
        stepsize = stepsizes
        prior_weight = iter(prior_weights)
    else:

        def stepsize(i):
            calls.append(("stepsize", i))
            return stepsizes[i]

        def prior_weight(i):
            calls.append(("prior_weight", i))
            return prior_weights[i]

    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=2,
    )

    result, diag = model(y, physics, init=init, get_diagnostic=True)

    x = init
    batch_size = y.shape[0]
    with torch.no_grad():
        for i, (step, weight) in enumerate(zip(stepsizes, prior_weights, strict=True)):
            x = (x - step * 0.8 * (0.8 * x - y)) / (1 + step * weight)
            torch.testing.assert_close(
                diag["stepsize"][:, i], step.expand(batch_size).cpu()
            )
            torch.testing.assert_close(
                diag["prior_weight"][:, i], weight.expand(batch_size).cpu()
            )
            assert "objective" not in diag
    torch.testing.assert_close(result, x)
    if schedule_type == "callable":
        assert calls == [
            ("stepsize", 0),
            ("prior_weight", 0),
            ("stepsize", 1),
            ("prior_weight", 1),
        ]


@pytest.mark.parametrize("backtracking", [None, Armijo()])
def test_pgd_tensor_parameters_preserve_gradients(optim_problem, backtracking):
    physics, y, init = optim_problem
    stepsize = y.new_tensor(10.0 if backtracking else 0.05)
    prior_weight = y.new_tensor(0.3)
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=1,
        backtracking=backtracking,
        unfold=True,
    )

    result, diag = model(y, physics, init=init, get_diagnostic=True)
    result.sum().backward()

    assert model.stepsize.grad is not None
    assert model.prior_weight.grad is not None
    assert not diag["stepsize"].requires_grad
    assert not diag["prior_weight"].requires_grad
    torch.testing.assert_close(
        diag["prior_weight"][:, 0], prior_weight.expand(y.shape[0]).cpu()
    )


def test_pgd_iteration_dependent_terms(optim_problem):
    physics, y, init = optim_problem
    stepsize, prior_weight = 0.05, 0.3
    priors = [dinv.optim.Tikhonov(), dinv.optim.L1Prior()]
    data_fidelities = [dinv.optim.L2(), dinv.optim.L2(sigma=0.5)]
    model = dinv.optim_v2.PGD(
        data_fidelity=data_fidelities,
        prior=priors,
        stepsize=stepsize,
        prior_weight=prior_weight,
        max_iter=2,
    )

    result, diag = model(y, physics, init=init, get_diagnostic=True)

    x = init
    with torch.no_grad():
        for i, (data_fidelity, prior) in enumerate(zip(data_fidelities, priors)):
            x = prior.prox(
                x - stepsize * data_fidelity.grad(x, y, physics),
                gamma=stepsize * prior_weight,
            )
            assert "objective" not in diag
    torch.testing.assert_close(result, x)


def test_pgd_backtracking_failure_stops(optim_problem):
    physics, y, init = optim_problem
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.ZeroPrior(),  # a pure gradient step diverges with a huge stepsize
        stepsize=1e6,
        max_iter=5,
        backtracking=Armijo(max_backtracking_steps=1),
    )
    with pytest.warns(UserWarning, match="Stopping"):
        result, diag = model(y, physics, init=init, get_diagnostic=True)
    torch.testing.assert_close(result, init)
    assert diag["stepsize"].shape[1] == 0


VARYING_OBJECTIVES = [
    pytest.param({"prior_weight": [0.1, 0.2, 0.3]}, id="prior_weight-sequence"),
    pytest.param({"prior_weight": lambda iteration: 0.1}, id="prior_weight-callable"),
    pytest.param({"prior_param": [0.1, 0.2, 0.3]}, id="prior_param-sequence"),
    pytest.param(
        {"prior": [dinv.optim.Tikhonov(), dinv.optim.L1Prior(), dinv.optim.Tikhonov()]},
        id="prior-sequence",
    ),
    pytest.param(
        {
            "data_fidelity": [
                dinv.optim.L2(),
                dinv.optim.L2(sigma=0.5),
                dinv.optim.L2(),
            ]
        },
        id="data_fidelity-sequence",
    ),
]


@pytest.mark.parametrize("varying", VARYING_OBJECTIVES)
def test_pgd_varying_objective_disables_objective(optim_problem, varying):
    physics, y, init = optim_problem
    kwargs = {
        "data_fidelity": dinv.optim.L2(),
        "prior": dinv.optim.Tikhonov(),
        "stepsize": 0.05,
        "max_iter": 3,
        **varying,
    }
    with pytest.warns(UserWarning, match="Backtracking is disabled"):
        model = dinv.optim_v2.PGD(**kwargs, backtracking=Armijo())
    assert model.backtracking is None
    assert not model.has_objective

    result, diag = model(y, physics, init=init, get_diagnostic=True)
    assert torch.isfinite(result).all()
    assert "objective" not in diag

    with pytest.raises(ValueError, match="crit_conv='cost'"):
        dinv.optim_v2.PGD(**kwargs, crit_conv="cost")


class FakeDenoiser(torch.nn.Module):
    """Fake denoiser recording the noise levels it receives."""

    def __init__(self):
        super().__init__()
        self.sigmas = []

    def forward(self, x, sigma):
        self.sigmas.append(float(sigma))
        return 0.9 * x


def test_pgd_non_explicit_prior_disables_objective(optim_problem):
    physics, y, init = optim_problem
    prior = dinv.optim.PnP(FakeDenoiser())
    with pytest.warns(UserWarning, match="not explicit"):
        model = dinv.optim_v2.PGD(
            data_fidelity=dinv.optim.L2(),
            prior=prior,
            sigma_denoiser=0.1,
            stepsize=0.05,
            max_iter=2,
            backtracking=Armijo(),
        )
    assert model.backtracking is None
    assert not model.has_objective

    result, diag = model(y, physics, init=init, get_diagnostic=True)
    assert torch.isfinite(result).all()
    assert "objective" not in diag

    with pytest.raises(ValueError, match="crit_conv='cost'"):
        dinv.optim_v2.PGD(data_fidelity=dinv.optim.L2(), prior=prior, crit_conv="cost")


@pytest.mark.parametrize("unfold", [False, True])
def test_pgd_gradients_only_when_unfolded(optim_problem, unfold):
    physics, y, init = optim_problem
    init = init.clone().requires_grad_()
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=0.05,
        max_iter=2,
        unfold=unfold,
        trainable_params=[],
    )

    result = model(y, physics, init=init)

    assert result.requires_grad is unfold
    if unfold:
        result.sum().backward()
        assert init.grad is not None


@pytest.mark.parametrize("get_trajectory", [False, True])
@pytest.mark.parametrize("get_diagnostic", [False, True])
def test_pgd_return_format(optim_problem, get_diagnostic, get_trajectory):
    physics, y, init = optim_problem
    batch_size, max_iter = y.shape[0], 3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=0.05,
        max_iter=max_iter,
        metrics=[dinv.loss.MSE()],
    )

    out = model(
        y,
        physics,
        init=init,
        x_gt=init,
        get_diagnostic=get_diagnostic,
        get_trajectory=get_trajectory,
    )

    if not get_diagnostic and not get_trajectory:
        assert isinstance(out, torch.Tensor)
        return

    result, diag = out
    assert isinstance(result, torch.Tensor)

    if get_diagnostic:
        assert diag["stepsize"].shape == (batch_size, max_iter)
        assert diag["prior_weight"].shape == (batch_size, max_iter)
        assert diag["residual"].shape == (batch_size, max_iter)
        assert diag["objective"].shape == (batch_size, max_iter + 1)
        assert diag["metrics"]["MSE"].shape == (batch_size, max_iter + 1)
        assert "prior_param" not in diag

    if get_trajectory:
        assert len(diag["trajectory"]) == max_iter + 1
        torch.testing.assert_close(diag["trajectory"][0], init)
        torch.testing.assert_close(diag["trajectory"][-1], result)
    else:
        assert "trajectory" not in diag


@pytest.mark.parametrize("crit_conv", ["residual", "cost"])
def test_pgd_early_stop(optim_problem, crit_conv):
    physics, y, init = optim_problem
    max_iter, thres_conv = 500, 1e-3
    model = dinv.optim_v2.PGD(
        data_fidelity=dinv.optim.L2(),
        prior=dinv.optim.Tikhonov(),
        stepsize=0.5,
        prior_weight=0.3,
        max_iter=max_iter,
        early_stop=True,
        crit_conv=crit_conv,
        thres_conv=thres_conv,
    )

    _, diag = model(y, physics, init=init, get_diagnostic=True)

    num_iter = diag["stepsize"].shape[1]
    assert 0 < num_iter < max_iter
    if crit_conv == "residual":
        assert diag["residual"][:, -1].max() < thres_conv
    else:
        objective = diag["objective"]
        crit = (objective[:, -2] - objective[:, -1]).abs() / (
            objective[:, -2].abs() + 1e-06
        )
        assert crit.max() < thres_conv
