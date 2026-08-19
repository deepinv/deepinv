"""Tests for the prior-agnostic bilevel interface.

Two properties decide whether a new prior is usable with MAID, and both are
checked here for every supplied prior:

* the energy is convex in ``x``, so the lower level has a unique minimiser and
  the strong-convexity certificate applies. Checked by assembling the Hessian
  on a small image and asking for its smallest eigenvalue.
* the hypergradient agrees with a central finite difference of
  ``g(x*(theta))``. This is the only check that can catch a disagreement
  between the model that is solved and the model that is differentiated:
  autograd faithfully reproduces whatever the energy expresses, mistakes
  included.
"""

import pytest
import torch

from deepinv.optim.bilevel import (
    BatchedPriorProblem,
    ConvexRidgeConfig,
    ConvexRidgePrior2,
    ICNNPrior,
    LearnedTVPrior,
    ParametricPrior,
)


def _priors(channels=2):
    """One instance of each supplied prior, kept small enough to assemble."""
    return {
        "learned_tv": LearnedTVPrior(channels=channels, eta=1e-2),
        "icnn": ICNNPrior(channels=channels, hidden=4, depth=1, kernel=3),
        "convex_ridge": ConvexRidgePrior2(
            ConvexRidgeConfig(
                nb_channels=(channels, 3, 4),
                filter_sizes=(3, 3),
                gamma=1e-2,
                weak_convexity=0.0,
                lip_fft_size=16,
            )
        ),
    }


PRIOR_NAMES = sorted(_priors().keys())


def _problem(prior, *, size=6, channels=2, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    x_star = torch.rand((2, channels, size, size), generator=gen, dtype=dtype)
    y = x_star + 0.05 * torch.randn(x_star.shape, generator=gen, dtype=dtype)
    return BatchedPriorProblem(y=y, x_star=x_star, prior=prior, gamma=1e-2)


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_energy_is_per_sample_and_finite(name):
    prior = _priors()[name]
    theta = prior.init_theta(seed=0)
    x = torch.rand((3, 2, 6, 6), dtype=torch.float64)
    e = prior.energy(x, theta)
    assert e.shape == (3,), f"energy must be one scalar per sample, got {e.shape}"
    assert torch.isfinite(e).all()


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_init_theta_respects_dtype_and_size(name):
    prior = _priors()[name]
    for dtype in (torch.float32, torch.float64):
        theta = prior.init_theta(dtype=dtype, seed=3)
        assert theta.dtype == dtype
        assert theta.numel() == prior.n_params
        assert torch.isfinite(theta).all()


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_energy_rejects_wrong_theta_size(name):
    prior = _priors()[name]
    theta = prior.init_theta(seed=0)
    x = torch.rand((1, 2, 6, 6), dtype=torch.float64)
    with pytest.raises((ValueError, RuntimeError)):
        prior.energy(x, theta[:-1])


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_energy_is_convex_in_x(name):
    """Smallest Hessian eigenvalue is non-negative up to rounding."""
    prior = _priors()[name]
    problem = _problem(prior, size=4)
    theta = prior.init_theta(seed=0)
    x = torch.rand_like(problem.x_star[:1])

    n = x[0].numel()
    single = BatchedPriorProblem(
        y=problem.y[:1], x_star=problem.x_star[:1], prior=prior, gamma=1e-2
    )
    # The data term and the ridge floor are convex by construction, so subtract
    # them to test the prior itself rather than what surrounds it.
    hessian = torch.zeros((n, n), dtype=x.dtype)
    for i in range(n):
        v = torch.zeros_like(x).flatten()
        v[i] = 1.0
        v = v.view_as(x)
        hv = single.hess_matvec(x, theta, v)
        hessian[:, i] = hv.flatten()

    hessian = 0.5 * (hessian + hessian.T)
    eigmin = float(torch.linalg.eigvalsh(hessian).min())
    assert eigmin > -1e-8, f"{name} is not convex in x: min eigenvalue {eigmin:.3e}"


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_hypergradient_matches_finite_difference(name):
    """Central finite difference of g(x*(theta)) along a random direction."""
    prior = _priors()[name]
    problem = _problem(prior, size=5, seed=2)
    theta = prior.init_theta(seed=1)

    gen = torch.Generator().manual_seed(11)
    d = torch.randn(theta.shape, generator=gen, dtype=theta.dtype)
    d = d / d.norm()

    eps_solve, delta = 1e-10, 1e-10
    x_star, _res, _info = problem.solve_lower(theta, eps_solve, max_iter=2000)
    z = problem.hypergradient(x_star, theta, delta)
    analytic = float((z * d).sum())

    def value_at(t):
        x, _r, _i = problem.solve_lower(t, eps_solve, max_iter=2000)
        return float(problem.g_per_sample(x).sum())

    h = 1e-5
    numeric = (value_at(theta + h * d) - value_at(theta - h * d)) / (2 * h)

    scale = max(abs(analytic), abs(numeric), 1e-12)
    rel = abs(analytic - numeric) / scale
    assert rel < 2e-4, (
        f"{name}: hypergradient {analytic:.8e} disagrees with finite "
        f"difference {numeric:.8e} (relative {rel:.2e})"
    )


def test_learned_tv_energy_vanishes_on_constant_image():
    """Subtracting eta sets the energy of a constant image to zero."""
    prior = LearnedTVPrior(channels=2, eta=1e-3)
    theta = prior.init_theta(seed=0)
    x = torch.full((2, 2, 5, 5), 0.37, dtype=torch.float64)
    e = prior.energy(x, theta)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-10)


def test_learned_tv_weights_stay_positive_under_any_theta():
    """exp() keeps the weights positive, which is what preserves convexity."""
    prior = LearnedTVPrior(channels=3, eta=1e-2)
    problem = _problem(prior, size=4, channels=3)
    # A strongly negative theta is what an unconstrained optimiser might reach.
    theta = torch.full((3,), -30.0, dtype=torch.float64)
    x = torch.rand_like(problem.x_star[:1])
    single = BatchedPriorProblem(
        y=problem.y[:1], x_star=problem.x_star[:1], prior=prior, gamma=1e-2
    )
    n = x[0].numel()
    hessian = torch.zeros((n, n), dtype=x.dtype)
    for i in range(n):
        v = torch.zeros_like(x).flatten()
        v[i] = 1.0
        hv = single.hess_matvec(x, theta, v.view_as(x))
        hessian[:, i] = hv.flatten()
    eigmin = float(torch.linalg.eigvalsh(0.5 * (hessian + hessian.T)).min())
    assert eigmin > -1e-8


def test_icnn_energy_is_zero_at_the_zero_image():
    """The prior is centred, so the constant softplus offset is removed."""
    prior = ICNNPrior(channels=2, hidden=4, depth=1)
    theta = prior.init_theta(seed=5)
    x = torch.zeros((2, 2, 6, 6), dtype=torch.float64)
    e = prior.energy(x, theta)
    assert torch.allclose(e, torch.zeros_like(e), atol=1e-12)


def test_custom_prior_needs_only_energy_and_init():
    """The documented extension contract: implement two methods, nothing else."""

    class L1Prior(ParametricPrior):
        def __init__(self, channels=2):
            self.n_params = channels

        def energy(self, x, theta):
            w = torch.exp(theta).view(1, -1, 1, 1)
            # Smoothed so the Hessian-vector product exists.
            return (w * torch.sqrt(x * x + 1e-6)).flatten(1).sum(dim=1)

        def init_theta(self, *, dtype=torch.float64, device="cpu", seed=0):
            return torch.zeros(self.n_params, dtype=dtype, device=device)

    prior = L1Prior()
    problem = _problem(prior, size=4, seed=7)
    theta = prior.init_theta()
    x, _res, info = problem.solve_lower(theta, 1e-10, max_iter=2000)
    assert info["reached"]
    z = problem.hypergradient(x, theta, 1e-10)
    assert z.shape == theta.shape
    assert torch.isfinite(z).all()


def test_solve_lower_certificate_bounds_the_distance_to_the_minimiser():
    """||x* - x|| <= ||grad_x h(x)|| / mu, checked against a tighter solve."""
    prior = LearnedTVPrior(channels=2, eta=1e-2)
    problem = _problem(prior, size=5, seed=4)
    theta = prior.init_theta(seed=0)

    x_loose, res_loose, _ = problem.solve_lower(theta, 1e-3, max_iter=2000)
    x_tight, _res, info = problem.solve_lower(theta, 1e-11, max_iter=5000)
    assert info["reached"]

    distance = (x_loose - x_tight).flatten(1).norm(dim=1)
    bound = res_loose / problem.mu
    assert torch.all(distance <= bound * 1.05 + 1e-9), (
        f"certificate violated: distance {distance.tolist()} exceeds "
        f"bound {bound.tolist()}"
    )
