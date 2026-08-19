"""Tests for the batched lower-level solves and hypergradient accumulation.

The property that matters is exactness: batching is a memory strategy, so
splitting a sample set into sub-batches must change peak memory and nothing
else. That is checked directly against the sequential oracle, which is the
reference implementation.
"""

import pytest
import torch

import deepinv as dinv
from deepinv.optim.bilevel import (
    BatchedCRR,
    BatchedMinibatchOracle,
    ConvexRidgeConfig,
    MinibatchOracle,
    auto_batch_size,
    auto_initial_accuracy,
    auto_initial_step,
    pack_init_theta,
)
from deepinv.optim.bilevel.cg_utils import cg_solve_batched


def _tiny_cfg(**kwargs):
    defaults = dict(
        nb_channels=(1, 2, 4),
        filter_sizes=(3, 3),
        gamma=1e-2,
        weak_convexity=0.0,
        lip_fft_size=32,
    )
    defaults.update(kwargs)
    return ConvexRidgeConfig(**defaults)


def _denoising(n=4, size=8, seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    x_star = torch.rand((n, 1, size, size), generator=gen, dtype=dtype)
    y = x_star + 0.05 * torch.randn(x_star.shape, generator=gen, dtype=dtype)
    physics = dinv.physics.Denoising(noise_model=dinv.physics.GaussianNoise(0.0))
    return physics, y, x_star


def test_cg_solve_batched_matches_a_direct_solve_per_sample():
    """Each sample must get its own scalars, not the batch's."""
    torch.manual_seed(0)
    B, n = 3, 12
    mats, rhs = [], []
    for i in range(B):
        m = torch.randn((n, n), dtype=torch.float64)
        # Different conditioning per sample, so shared scalars would show up.
        mats.append(m @ m.T + (i + 1) * torch.eye(n, dtype=torch.float64))
        rhs.append(torch.randn((n,), dtype=torch.float64))
    A = torch.stack(mats)
    b = torch.stack(rhs).view(B, 1, 1, n)

    def matvec(v):
        flat = v.view(B, n, 1)
        return torch.bmm(A, flat).view(B, 1, 1, n)

    out = cg_solve_batched(
        matvec, b, tol=torch.full((B,), 1e-12, dtype=torch.float64), max_iter=200
    )
    expected = torch.stack(
        [torch.linalg.solve(A[i], b.view(B, n)[i]) for i in range(B)]
    )
    assert torch.allclose(
        out.x.view(B, n), expected, atol=1e-8
    ), f"max error {(out.x.view(B, n) - expected).abs().max():.3e}"


def test_solve_lower_reaches_tolerance_and_certificate_holds():
    physics, y, x_star = _denoising(n=3, size=8)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=1)
    problem = BatchedCRR(y=y, x_star=x_star, cfg=cfg, physics=physics)

    x_loose, res_loose, _info = problem.solve_lower(theta, 1e-3, max_iter=2000)
    # Above the float64 floor, so "reached" is a meaningful assertion. Asking
    # for 1e-11 puts the solve at the precision floor, where stalling short of
    # eps is by design not an error: the certificate still holds, only wider.
    x_tight, _res, info = problem.solve_lower(theta, 1e-9, max_iter=5000)
    assert info["reached"]

    distance = (x_loose - x_tight).flatten(1).norm(dim=1)
    bound = res_loose / problem.mu
    assert torch.all(distance <= bound * 1.05 + 1e-9)


def test_stalling_at_the_precision_floor_is_not_an_error():
    """A solve that cannot reach eps returns a valid, wider certificate."""
    physics, y, x_star = _denoising(n=3, size=8)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=1)
    problem = BatchedCRR(y=y, x_star=x_star, cfg=cfg, physics=physics)

    # Far below what float64 can deliver on this problem.
    x, res, info = problem.solve_lower(theta, 1e-14, max_iter=3000)
    assert not info["reached"]
    assert torch.isfinite(res).all() and torch.all(res >= 0)
    assert torch.isfinite(x).all()


def test_batching_does_not_change_the_hypergradient():
    """Splitting a sample set must move z only by rounding."""
    physics, y, x_star = _denoising(n=6, size=8, seed=3)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=2)
    groups = [(physics, y, x_star)]

    zs = []
    for batch_size in (1, 2, 3, 6):
        oracle = BatchedMinibatchOracle(groups, cfg, batch_size=batch_size)
        lower = oracle.solve_lower_level(theta, 1e-9)
        zs.append(oracle.hypergradient(theta, lower, 1e-9).z)

    reference = zs[-1]
    for batch_size, z in zip((1, 2, 3, 6), zs, strict=True):
        rel = float((z - reference).norm() / max(float(reference.norm()), 1e-30))
        assert rel < 1e-10, (
            f"batch_size={batch_size} changed the hypergradient by {rel:.3e} "
            "relative; accumulation is not exact"
        )


def test_batched_matches_the_sequential_oracle():
    """The batched path must agree with the reference implementation."""
    physics, y, x_star = _denoising(n=4, size=8, seed=5)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=4)

    batched = BatchedMinibatchOracle([(physics, y, x_star)], cfg, batch_size=2)
    zb = batched.hypergradient(theta, batched.solve_lower_level(theta, 1e-10), 1e-10).z

    from deepinv.optim.bilevel import CRRSampleProblem, CRRSampleOracle

    samples = [
        CRRSampleProblem(
            physics=physics,
            y=y[i : i + 1],
            x_star=x_star[i : i + 1],
            cfg=cfg,
            max_iter=5000,
        )
        for i in range(x_star.shape[0])
    ]
    sequential = MinibatchOracle([CRRSampleOracle(s) for s in samples], chunk_size=1)
    zs = sequential.hypergradient(
        theta, sequential.solve_lower_level(theta, 1e-10), 1e-10
    ).z

    rel = float((zb - zs).norm() / max(float(zs.norm()), 1e-30))
    assert rel < 1e-6, f"batched and sequential disagree by {rel:.3e} relative"


def test_construction_refuses_a_physics_that_couples_samples():
    """A batched Hessian is block diagonal only if A acts per sample."""
    physics, y, x_star = _denoising(n=3, size=8)

    class CouplingPhysics:
        def A(self, x):
            # Mixes samples, so one sample's measurement depends on another's.
            return x + x.flip(0)

        def A_adjoint(self, v):
            return v + v.flip(0)

    with pytest.raises(ValueError, match="(?i)sample|batch|coupl"):
        BatchedCRR(y=y, x_star=x_star, cfg=_tiny_cfg(), physics=CouplingPhysics())


def test_auto_batch_size_is_positive_and_clamped():
    n = auto_batch_size(16, per_sample_bytes=1024, device="cpu")
    assert 1 <= n <= 16
    capped = auto_batch_size(16, per_sample_bytes=1024, device="cpu", max_batch=4)
    assert capped <= 4
    # A sample that cannot fit must still leave a usable batch of one.
    huge = auto_batch_size(16, per_sample_bytes=10**15, device="cpu")
    assert huge >= 1


def test_auto_initial_accuracy_and_step_are_finite_and_positive():
    physics, y, x_star = _denoising(n=3, size=8, seed=6)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=0)
    oracle = BatchedMinibatchOracle([(physics, y, x_star)], cfg, batch_size=3)

    eps0 = auto_initial_accuracy(oracle, theta)
    assert 0.0 < eps0 <= 1.0 and torch.isfinite(torch.tensor(eps0))

    z0 = oracle.hypergradient(theta, oracle.solve_lower_level(theta, eps0), eps0).z
    alpha0 = auto_initial_step(oracle, theta, z0)
    assert alpha0 > 0.0 and torch.isfinite(torch.tensor(alpha0))
    # The first trial step should move theta by a bounded relative amount.
    move = alpha0 * float(z0.norm()) / max(float(theta.norm()), 1.0)
    assert move < 1.0, f"first step moves theta by {move:.3e} relative"


def test_initial_residual_rms_is_available_for_deriving_eps0():
    physics, y, x_star = _denoising(n=3, size=8, seed=7)
    cfg = _tiny_cfg()
    theta = pack_init_theta(cfg, seed=0)
    oracle = BatchedMinibatchOracle([(physics, y, x_star)], cfg, batch_size=3)
    rms = oracle.initial_residual_rms(theta)
    assert rms > 0.0 and torch.isfinite(torch.tensor(rms))


def test_float32_agrees_with_float64_in_direction():
    """Inexactness shortens the hypergradient; bias would rotate it."""
    physics, y, x_star = _denoising(n=3, size=8, seed=8)
    cfg = _tiny_cfg()

    theta64 = pack_init_theta(cfg, seed=1, dtype=torch.float64)
    o64 = BatchedMinibatchOracle([(physics, y, x_star)], cfg, batch_size=3)
    z64 = o64.hypergradient(theta64, o64.solve_lower_level(theta64, 1e-9), 1e-9).z

    y32, x32 = y.float(), x_star.float()
    theta32 = theta64.float()
    o32 = BatchedMinibatchOracle([(physics, y32, x32)], cfg, batch_size=3)
    z32 = o32.hypergradient(theta32, o32.solve_lower_level(theta32, 1e-6), 1e-6).z

    cos = float(torch.dot(z64, z32.double()) / (z64.norm() * z32.double().norm()))
    assert cos > 0.999, f"float32 hypergradient is rotated: cosine {cos:.8f}"
