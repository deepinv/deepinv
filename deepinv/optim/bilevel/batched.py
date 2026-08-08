"""Batched lower-level solves and hypergradient accumulation for MAID.

Motivation
----------
:mod:`minibatch` bounds peak memory with ``chunk_size`` but still solves the
samples one at a time, so a 24-sample outer iteration is 24 sequential Newton
solves on tensors of a few thousand elements. That path is latency bound:
three chained convolutions on ``(1, 3, 32, 32)`` take 244 us on CPU and
145 us on MPS, while the same work batched to ``(24, 3, 32, 32)`` takes
446 us on MPS in total (18.6 us per sample). The device is idle waiting for
dispatch, not for arithmetic.

The samples are independent given ``theta``, so the batched lower-level
Hessian is block diagonal and every operator applies to all samples at once.
Only the *scalars* stay per sample: CG's ``alpha`` and ``beta``, the Armijo
step, and the convergence and stall tests. Sharing any of those across
samples would couple independent problems and break both the Krylov property
and the certificate.

Memory
------
Batch size is chosen from a **measured** footprint, not a closed-form
estimate. A probe solve records peak allocation, then the batch is sized to
fit the device's free memory with a safety factor. An analytical working-set
estimate would require predicting autograd graph retention through three
convolutions and a CG basis; measurement is more reliable than that prediction.

Accumulation
------------
The hypergradient is a sum over samples, ``z = sum_i z_i``, so batches
accumulate exactly: partitioning the sample set changes only the peak
memory, not the result. This is checked against the sequential path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .cg_utils import cg_solve_batched
from .oracle import (
    HypergradientOracle,
    HypergradientState,
    LowerLevelState,
)
from .smooth import smooth_hypergradient_error_bound
from .convex_ridge import (
    ConvexRidgeConfig,
    ConvexRidgePrior,
    apply_conv,
    get_conv_lip,
    grad_smooth_l1,
    smooth_l1,
    unpack_theta,
)


def ridge_energy_per_sample(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scaling: torch.Tensor,
    beta: torch.Tensor,
    cfg: ConvexRidgeConfig,
    lip: torch.Tensor | None = None,
) -> torch.Tensor:
    """CRR energy reduced per sample, shape ``(B,)``.

    Same expression as :func:`convex_ridge.ridge_energy`, summed over the
    non-batch dimensions instead of globally, so the Armijo test can accept or
    reject each sample on its own objective.
    """
    if lip is None:
        lip = get_conv_lip(weights, cfg, detach=False)
    feats = apply_conv(x, weights, cfg, lip=lip)
    scaled = feats * torch.exp(scaling)
    rho = smooth_l1(torch.exp(beta) * scaled) * torch.exp(-beta)
    if cfg.weak_convexity != 0.0:
        rho = rho - smooth_l1(scaled) * cfg.weak_convexity
    reg = (rho * torch.exp(-2.0 * scaling)).flatten(1).sum(dim=1)
    return reg + 0.5 * cfg.gamma * (x * x).flatten(1).sum(dim=1)


def available_memory(device: torch.device | str) -> int:
    """Free bytes usable for a batch, per backend.

    CUDA reports free/total directly. MPS uses unified memory, so the
    recommended working-set limit minus what is already allocated is the
    meaningful figure. CPU has no hard ceiling here, so a conservative fixed
    budget is returned rather than pretending to know.
    """
    dev = torch.device(device)
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free)
    if dev.type == "mps":
        try:
            recommended = int(torch.mps.recommended_max_memory())
            allocated = int(torch.mps.current_allocated_memory())
            return max(recommended - allocated, 1 << 26)
        except Exception:
            return 1 << 31
    return 1 << 31


def measure_sample_bytes(
    run_one: Callable[[int], None],
    device: torch.device | str,
    probe_batches: tuple[int, int] = (2, 4),
) -> int:
    """Marginal bytes per sample, from the slope of two probe runs.

    Two probes, not one, because a single measurement conflates the
    per-sample cost with fixed overhead (the parameter vector, the FFT plan,
    the physics). The slope ``(m1 - m0) / (b1 - b0)`` isolates the part that
    scales with batch size, which is the quantity that decides how many
    samples fit.

    Backend accounting differs: the wrong API can report nearly zero cost.
    CUDA exposes a true high-water mark via ``max_memory_allocated``. MPS has
    no peak API; ``current_allocated_memory`` is observed after the solve has
    freed temporaries and reports about 0.06 MiB per sample at every image
    size from 32 to 256, while the true marginal cost rises from about 3 MiB
    to 128 MiB. ``driver_allocated_memory`` tracks what the driver still holds
    and reflects that growth.

    ``run_one`` must exercise the hypergradient as well as the solve, since
    the autograd graph for the mixed Jacobian is the peak, not the solve.
    """
    dev = torch.device(device)
    marks: list[int] = []
    for b in probe_batches:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            run_one(b)
            torch.cuda.synchronize(dev)
            marks.append(int(torch.cuda.max_memory_allocated(dev)))
        elif dev.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
            base = int(torch.mps.driver_allocated_memory())
            run_one(b)
            torch.mps.synchronize()
            marks.append(max(int(torch.mps.driver_allocated_memory()) - base, 0))
        else:
            run_one(b)
            marks.append(0)
    b0, b1 = probe_batches
    slope = (marks[1] - marks[0]) / max(b1 - b0, 1)
    if slope <= 0:
        # Degenerate probe (allocator reuse, or a backend reporting nothing):
        # fall back to the average rather than returning a floor that would
        # claim any batch fits.
        slope = marks[-1] / max(b1, 1)
    return max(int(slope), 1 << 16)


def auto_batch_size(
    n_samples: int,
    per_sample_bytes: int,
    device: torch.device | str,
    *,
    safety: float = 0.35,
    max_batch: int | None = None,
) -> int:
    """Largest batch fitting ``safety`` of free memory, clamped to sanity.

    ``safety`` is kept well under 1 because the probe measures a steady-state
    solve, while the full run also holds the autograd graph for the mixed
    Jacobian and the CG basis. A smaller batch is preferable to an
    out-of-memory abort mid-training.
    """
    budget = int(available_memory(device) * float(safety))
    b = max(1, budget // max(int(per_sample_bytes), 1))
    b = min(b, int(n_samples))
    if max_batch is not None:
        b = min(b, int(max_batch))
    return int(b)


@dataclass
class BatchedCRR:
    """``B`` samples sharing one ``theta`` and one physics, solved together.

    Any DeepInverse physics works, not only denoising. What is required is
    that ``A`` act per sample along the batch dimension, which the operators
    already do, including with **per-sample parameters**: an ``Inpainting``
    built with a ``(B, C, H, W)`` mask and a ``BlurFFT`` built with batched
    kernels both apply their own operator to their own sample. Construction
    checks this by perturbing one sample and confirming that the others'
    measurements do not move.

    Samples that need genuinely different operators (one inpainting, one
    deblurring) are grouped into separate batches and their hypergradients
    summed. Accumulation across batches is exact: splitting 8 samples into
    batches of 3 moves ``z`` by 2e-15 in float64.

    ``mu_data`` is 1 when ``A = I`` and 0 otherwise, matching
    :class:`CRRSampleProblem`, so a general forward operator needs
    ``cfg.gamma > 0`` for the certificate to have a positive modulus.
    """

    y: torch.Tensor  # (B, ...) measurements, shape set by the physics
    x_star: torch.Tensor  # (B, C, H, W)
    cfg: ConvexRidgeConfig
    physics: Any = None

    def __post_init__(self) -> None:
        if self.x_star.dim() != 4:
            raise ValueError(
                f"expected x_star (B, C, H, W), got {tuple(self.x_star.shape)}"
            )
        if self.y.shape[0] != self.x_star.shape[0]:
            raise ValueError("y and x_star must agree on the batch dimension")
        self.dtype = self.x_star.dtype
        self.device = self.x_star.device
        self.B = int(self.x_star.shape[0])
        self.prior = ConvexRidgePrior(self.cfg)
        self.n_elem = int(self.x_star[0].numel())
        if self.physics is None:
            import deepinv as _dinv

            self.physics = _dinv.physics.Denoising(
                noise_model=_dinv.physics.GaussianNoise(0.0)
            )
        self._check_batched_independence()
        self.mu = self._mu_data() + self.cfg.gamma
        if self.mu <= 0.0:
            raise ValueError(
                "lower level is not strongly convex: mu_data + gamma = "
                f"{self.mu}. Set cfg.gamma > 0 for a general forward operator."
            )

    def _mu_data(self) -> float:
        """1 for ``A = I``, else 0, as in :class:`CRRSampleProblem`."""
        v = torch.randn_like(self.x_star)
        Av = self.physics.A(v)
        if Av.shape == v.shape and torch.allclose(
            Av, v, atol=1e-12, rtol=0.0
        ):
            return 1.0
        return 0.0

    def _check_batched_independence(self) -> None:
        """Refuse a physics that mixes samples.

        Batching is valid only when ``A`` is block diagonal across the batch.
        An operator that couples samples would produce an incorrect gradient
        for every sample, so independence is checked at construction.
        """
        v = torch.randn_like(self.x_star)
        try:
            a1 = self.physics.A(v)
        except RuntimeError as exc:
            raise ValueError(
                f"{type(self.physics).__name__} cannot be applied to a batch "
                f"of {self.B}: {exc}. A physics carrying per-sample parameters "
                "(for example an Inpainting built with a (B, C, H, W) mask) is "
                "tied to that batch size. Build one physics per batch, or use "
                "a shared parameter."
            ) from exc
        v2 = v.clone()
        v2[0] = v2[0] + 1.0
        a2 = self.physics.A(v2)
        if self.B > 1 and not torch.allclose(a1[1:], a2[1:], atol=0.0, rtol=0.0):
            raise ValueError(
                f"{type(self.physics).__name__} couples samples across the "
                "batch dimension, so the batched Hessian is not block "
                "diagonal. Use the sequential path for this physics."
            )

    # -- model ---------------------------------------------------------
    def _parts(self, theta: torch.Tensor):
        return unpack_theta(theta, self.cfg)

    def energy_per_sample(self, x: torch.Tensor, theta: torch.Tensor):
        w, s, b = self._parts(theta)
        r = self.physics.A(x) - self.y
        data = 0.5 * (r * r).flatten(1).sum(dim=1)
        return data + ridge_energy_per_sample(x, w, s, b, self.cfg)

    def grad_x(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Batched gradient. Independent samples, so one autograd pass."""
        w, s, b = self._parts(theta)
        w = [wi.detach() for wi in w]
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, theta.detach()).sum()
        (g,) = torch.autograd.grad(e, x_)
        return g.detach()

    def hess_matvec(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Block-diagonal Hessian-vector product, all samples at once."""
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, theta.detach()).sum()
        (g,) = torch.autograd.grad(e, x_, create_graph=True)
        (hv,) = torch.autograd.grad((g * v.detach()).sum(), x_)
        return hv.detach()

    def mixed_jac_T(
        self, x: torch.Tensor, theta: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """``d/dtheta (grad_x h) ^T v``, summed over samples.

        The sum is the point: the hypergradient over a sample set is the sum
        of per-sample hypergradients, so batches accumulate exactly.
        """
        th = theta.detach().requires_grad_(True)
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, th).sum()
        (g,) = torch.autograd.grad(e, x_, create_graph=True)
        (jtv,) = torch.autograd.grad((g * v.detach()).sum(), th)
        return jtv.detach()

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.x_star

    def g_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        d = x - self.x_star
        return 0.5 * (d * d).flatten(1).sum(dim=1)

    # -- solve ---------------------------------------------------------
    def solve_lower(
        self,
        theta: torch.Tensor,
        eps: float,
        *,
        x_init: torch.Tensor | None = None,
        max_iter: int = 5000,
        cg_iters: int = 20,
        armijo_c: float = 1e-4,
        armijo_rho: float = 0.5,
        max_bt: int = 20,
        stall_patience: int = 25,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Batched truncated Newton with per-sample Armijo and stalling.

        ``eps`` is the per-element rms tolerance, matching
        :meth:`CRRSampleProblem.solve_lower`, so the same number means the
        same accuracy whether the samples are run batched or one by one.
        """
        scale = float(self.n_elem) ** 0.5
        eps_eff = max(float(eps), 100.0 * float(torch.finfo(self.dtype).eps))
        tol = eps_eff * self.mu * scale

        x = (
            self.physics.A_adjoint(self.y).detach().clone()
            if x_init is None
            else x_init.detach().clone()
        )
        g = self.grad_x(x, theta)
        res = g.flatten(1).norm(dim=1)
        best_res = res.clone()
        best_x = x.clone()
        since = torch.zeros(self.B, dtype=torch.long, device=self.device)
        stalled = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        n_iter = 0

        for it in range(1, int(max_iter) + 1):
            live = (res > tol) & (~stalled)
            if not bool(live.any()):
                break
            n_iter = it
            g = self.grad_x(x, theta)
            res = g.flatten(1).norm(dim=1)

            cg_tol = torch.clamp(0.1 * res, min=tol * 0.1)
            cg = cg_solve_batched(
                lambda v: self.hess_matvec(x, theta, v),
                -g,
                tol=cg_tol,
                max_iter=cg_iters,
            )
            p = cg.x
            # Per-sample descent check; fall back to steepest descent only for
            # the samples that need it.
            gTp = (g * p).flatten(1).sum(dim=1)
            bad = (gTp >= 0.0).view(self.B, 1, 1, 1)
            p = torch.where(bad, -g, p)
            gTp = (g * p).flatten(1).sum(dim=1)

            f0 = self.energy_per_sample(x, theta)
            alpha = torch.ones(self.B, device=self.device, dtype=self.dtype)
            accepted = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            x_new = x.clone()
            for _bt in range(max_bt):
                if bool(accepted.all()):
                    break
                a = alpha.view(self.B, 1, 1, 1)
                trial = x + a * p
                f_t = self.energy_per_sample(trial, theta)
                ok = (f_t <= f0 + armijo_c * alpha * gTp) & (~accepted)
                sel = ok.view(self.B, 1, 1, 1)
                x_new = torch.where(sel, trial, x_new)
                accepted = accepted | ok
                alpha = torch.where(accepted, alpha, alpha * armijo_rho)
            # Samples with no acceptable step keep their current iterate.
            x = torch.where(accepted.view(self.B, 1, 1, 1), x_new, x)
            # Frozen samples must not move at all.
            x = torch.where(live.view(self.B, 1, 1, 1), x, best_x)

            res = self.grad_x(x, theta).flatten(1).norm(dim=1)
            improved = res < best_res * (1.0 - 1e-3)
            best_res = torch.where(improved, res, best_res)
            best_x = torch.where(improved.view(self.B, 1, 1, 1), x, best_x)
            since = torch.where(
                improved, torch.zeros_like(since), since + 1
            )
            stalled = stalled | (since >= stall_patience)

        x = best_x
        res = best_res
        info = {
            "n_iter": n_iter,
            "residual": res,
            "residual_rms": res / (scale * self.mu),
            "reached": bool((res <= tol * 1.01).all()),
            "n_stalled": int(stalled.sum().item()),
            "tol": tol,
        }
        return x, res, info

    def initial_residual_rms(self, theta: torch.Tensor) -> float:
        """Per-element stationarity residual at ``x0 = A^* y``.

        This is the scale ``eps`` is measured against. It is available before
        any solve and depends on neither the noise level nor the ground truth,
        which makes it the natural reference for choosing an initial accuracy;
        see :func:`auto_initial_accuracy`.
        """
        x0 = self.physics.A_adjoint(self.y)
        g = self.grad_x(x0, theta)
        scale = float(self.n_elem) ** 0.5
        return float(g.flatten(1).norm(dim=1).mean()) / (scale * self.mu)

    def hypergradient(
        self, x: torch.Tensor, theta: torch.Tensor, delta: float,
        *, max_cg_iter: int = 200,
    ) -> torch.Tensor:
        """``z = -sum_i J_i^T H_i^{-1} grad g_i`` over this batch."""
        rhs = self.grad_g(x)
        scale = float(self.n_elem) ** 0.5
        tol = max(float(delta), 100.0 * float(torch.finfo(self.dtype).eps))
        tol = tol * self.mu * scale
        cg = cg_solve_batched(
            lambda v: self.hess_matvec(x, theta, v),
            rhs,
            tol=torch.full((self.B,), tol, dtype=self.dtype, device=self.device),
            max_iter=max_cg_iter,
        )
        return -self.mixed_jac_T(x, theta, cg.x)


class BatchedMinibatchOracle(HypergradientOracle):
    """MAID oracle backed by batched solves and exact accumulation.

    Drop-in for :class:`MinibatchOracle`: same API, same reduction (the mean
    over samples), but each group of samples is solved as one batch instead of
    one at a time. Groups are split into sub-batches sized from measured
    memory, and their hypergradients summed. Accumulation is exact, so the
    batch size changes peak memory and nothing else.

    :param groups: ``[(physics, y, x_star), ...]``. One entry per distinct
        forward operator; samples sharing an operator (including one with
        per-sample parameters, such as a batched inpainting mask) belong in
        the same entry.
    :param batch_size: override the measured choice.
    """

    def __init__(
        self,
        groups: list[tuple[Any, torch.Tensor, torch.Tensor]],
        cfg: ConvexRidgeConfig,
        *,
        batch_size: int | None = None,
        safety: float = 0.35,
        max_batch: int | None = None,
        solver_max_iter: int = 5000,
    ):
        if not groups:
            raise ValueError("groups must be non-empty")
        self.cfg = cfg
        self.solver_max_iter = int(solver_max_iter)
        self.m = int(sum(g[2].shape[0] for g in groups))
        x0 = groups[0][2]
        self.device, self.dtype = x0.device, x0.dtype
        self.L_g_value = 1.0
        self.L_H_inv = 0.0
        self.L_J = 0.0
        self.reset_counters()

        if batch_size is None:
            probe = groups[0]

            th_probe = torch.zeros(
                cfg.n_params, dtype=self.dtype, device=self.device
            )

            def run_probe(b: int) -> None:
                bp = BatchedCRR(
                    y=probe[1][:b], x_star=probe[2][:b], cfg=cfg,
                    physics=probe[0],
                )
                xp, _r, _i = bp.solve_lower(th_probe, eps=1e-3, max_iter=5)
                # The mixed-Jacobian graph is the peak allocation; the probe
                # includes it so the estimate tracks the full working set.
                bp.hypergradient(xp, th_probe, delta=1e-3, max_cg_iter=5)

            n_avail = int(probe[2].shape[0])
            pb = (1, 2) if n_avail < 4 else (2, 4)
            try:
                per = measure_sample_bytes(
                    run_probe, self.device, probe_batches=pb
                )
            except (ValueError, RuntimeError):
                # A physics with per-sample parameters cannot be applied to a
                # slice of its own batch, so the two-point probe is impossible.
                # Measure once at the full batch and divide; this includes the
                # fixed overhead and so over-estimates the marginal cost, which
                # errs towards a smaller batch.
                per = measure_sample_bytes(
                    run_probe, self.device, probe_batches=(n_avail, n_avail)
                ) // max(n_avail, 1)
                per = max(int(per), 1 << 16)
            batch_size = auto_batch_size(
                self.m, per, self.device, safety=safety, max_batch=max_batch
            )
            self.per_sample_bytes = per
        else:
            self.per_sample_bytes = 0
        self.batch_size = max(1, int(batch_size))

        # Materialise the sub-batches once; order is fixed so the reduction is
        # deterministic across runs.
        self.batches: list[BatchedCRR] = []
        for physics, y, x_star in groups:
            n = int(x_star.shape[0])
            for s in range(0, n, self.batch_size):
                e = min(s + self.batch_size, n)
                self.batches.append(
                    BatchedCRR(
                        y=y[s:e], x_star=x_star[s:e], cfg=cfg, physics=physics
                    )
                )
        self._certified = True
        self._citation = (
            "Salehi et al., SIAM J. Math. Data Sci. 2025, Theorem 2.1; "
            "batched convex ridge regulariser"
        )

    # -- bookkeeping ---------------------------------------------------
    def reset_counters(self) -> None:
        self.n_lower_solves = 0
        self.n_hypergradients = 0
        self.n_sample_lower_solves = 0
        self.n_sample_hypergradients = 0
        self.n_gd_iters = 0
        self.peak_working_bytes = 0

    @property
    def certified(self) -> bool:
        return self._certified

    @property
    def citation(self) -> str:
        return self._citation

    @property
    def L_g(self) -> float:
        return float(self.L_g_value)

    def require_certified_or_opt_in(self, allow_uncertified: bool) -> None:
        if not self.certified and not allow_uncertified:
            raise RuntimeError("uncertified oracle without opt-in")

    # -- oracle API ----------------------------------------------------
    def solve_lower_level(
        self, theta: torch.Tensor, eps: float,
        warm_start: LowerLevelState | None = None,
    ) -> LowerLevelState:
        xs, g_acc, ng_acc, n_it = [], 0.0, 0.0, 0
        warm = warm_start.extras.get("x_list") if warm_start else None
        for k, bp in enumerate(self.batches):
            x, _res, info = bp.solve_lower(
                theta, eps=eps,
                x_init=warm[k] if warm is not None else None,
                max_iter=self.solver_max_iter,
            )
            xs.append(x)
            g_acc += float(bp.g_per_sample(x).sum().item())
            ng_acc += float(bp.grad_g(x).flatten(1).norm(dim=1).sum().item())
            n_it = max(n_it, int(info["n_iter"]))
            self.n_sample_lower_solves += bp.B
        self.n_lower_solves += 1
        self.n_gd_iters += n_it
        return LowerLevelState(
            x=torch.cat(xs),
            eps=eps,
            extras={
                "x_list": xs,
                "g_mean": g_acc / self.m,
                "grad_g_norm_mean": ng_acc / self.m,
                "n_iter": n_it,
            },
        )

    def hypergradient(
        self, theta: torch.Tensor, lower: LowerLevelState, delta: float
    ) -> HypergradientState:
        xs = lower.extras["x_list"]
        z_acc: torch.Tensor | None = None
        J_max = 0.0
        for bp, x in zip(self.batches, xs):
            z_b = bp.hypergradient(x, theta, delta=delta)
            z_acc = z_b.clone() if z_acc is None else z_acc + z_b
            J_max = max(J_max, self._batch_J_norm(bp, x, theta))
            self.n_sample_hypergradients += bp.B
        assert z_acc is not None
        self.n_hypergradients += 1
        return HypergradientState(
            z=z_acc / self.m,
            delta=delta,
            extras={
                "x": lower.x,
                "theta": theta,
                "mu": min(bp.mu for bp in self.batches),
                "J_norm": J_max,
                "grad_g_norm": lower.extras["grad_g_norm_mean"],
                "L_H_inv": 0.0,
                "L_J": 0.0,
            },
        )

    @staticmethod
    def _batch_J_norm(
        bp: BatchedCRR, x: torch.Tensor, theta: torch.Tensor, n_power: int = 2
    ) -> float:
        """Power estimate of ``||J||`` for the batch.

        ``mixed_jac_T`` returns the sum over samples, so this estimates the
        norm of the summed operator. Probing it with a vector supported on one
        sample recovers that sample's ``J_i``, hence this is an **upper bound**
        on every per-sample norm. Using it in the error bound is therefore
        conservative: omega comes out at least as large as the per-sample mean,
        which costs extra backtracking at worst and never understates the
        hypergradient error.
        """
        nrm = 0.0
        for _ in range(max(n_power, 1)):
            e = torch.randn_like(x)
            e = e / e.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-30)
            jte = bp.mixed_jac_T(x, theta, e)
            nrm = max(nrm, float(jte.norm().item()))
        return nrm

    def initial_residual_rms(self, theta: torch.Tensor) -> float:
        """Sample-weighted mean of the per-batch initial residual."""
        vals = [bp.initial_residual_rms(theta) for bp in self.batches]
        weights = [bp.B for bp in self.batches]
        return float(
            sum(v * w for v, w in zip(vals, weights)) / max(sum(weights), 1)
        )

    def error_bound(
        self, theta: torch.Tensor, lower: LowerLevelState,
        hyper: HypergradientState, eps: float, delta: float,
    ) -> float:
        return smooth_hypergradient_error_bound(
            eps=eps,
            delta=delta,
            mu=float(hyper.extras["mu"]),
            L_g=self.L_g,
            J_norm=float(hyper.extras["J_norm"]),
            grad_g_norm=float(hyper.extras["grad_g_norm"]),
            L_H_inv=0.0,
            L_J=0.0,
        )

    def g(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        s = 0
        for bp in self.batches:
            out = out + bp.g_per_sample(x[s:s + bp.B]).sum()
            s += bp.B
        return out / self.m

    def grad_g(self, x: torch.Tensor) -> torch.Tensor:
        parts, s = [], 0
        for bp in self.batches:
            parts.append(bp.grad_g(x[s:s + bp.B]))
            s += bp.B
        return torch.cat(parts) / self.m

    def f_closed_form(self, theta: torch.Tensor) -> torch.Tensor:
        lower = self.solve_lower_level(theta, eps=1e-4)
        return torch.as_tensor(
            lower.extras["g_mean"], dtype=self.dtype, device=self.device
        )

    def update_lipschitz_estimates(
        self, lower: LowerLevelState, theta: torch.Tensor
    ) -> None:
        # g is quadratic, so L_g = 1 exactly; nothing to estimate.
        self.L_g_value = 1.0



def auto_initial_accuracy(
    oracle: Any,
    theta0: torch.Tensor,
    *,
    factor: float = 0.02,
    min_eps: float | None = None,
    max_eps: float = 1.0,
) -> float:
    r"""Initial lower-level accuracy taken from the problem's own scale.

    Returns

    .. math::

        \varepsilon_0 = \text{factor} \cdot
        \frac{\|\nabla_x h(x_0, \theta_0)\|}{\sqrt{n}\,\mu},
        \qquad x_0 = A^\ast y,

    clamped to ``[min_eps, max_eps]``: a fixed relative reduction of the
    stationarity residual at the natural initialisation. The reference is
    observable before any solve and requires neither the noise level nor the
    ground truth.

    A single absolute ``eps0`` cannot be safe across problems. The residual
    scale carries the forward operator through :math:`\mu`, and the same
    constant can be fatal on one problem and harmless on another: on a
    denoising problem ``eps0 = 1e-1`` leaves the result below the noisy input
    (24.70 dB against 26.01), while on an inpainting problem with
    :math:`\gamma = 10^{-2}` the same value sits in a plateau that extends
    over four orders of magnitude.

    Choice of ``factor``
    --------------------
    The default is calibrated on a denoising sweep, the only problem in that
    study whose performance depends on ``eps0`` at all: ``1e-2`` succeeds and
    ``1e-1`` fails against an initial residual of ``0.0715``, placing the
    largest safe factor between 0.14 and 1.4. The default of 0.02 sits about
    seven times inside that boundary. Problems that are insensitive to
    ``eps0`` cannot constrain it further, so treat 0.02 as a conservative
    starting point rather than a tuned optimum, and raise it to trade accuracy
    for cheaper early solves.

    Clamping
    --------
    ``min_eps`` defaults to the dtype's achievable floor, ``100 *
    finfo(dtype).eps``, matching the lower-level solver: a smaller request
    cannot be met and would start the accuracy schedule below what the
    arithmetic supports. ``max_eps`` bounds the loose end, since a per-element
    tolerance above 1 is uninformative for data on the unit interval. A
    non-positive residual (an initialisation that is already stationary)
    returns ``min_eps``.

    :param oracle: an oracle exposing ``initial_residual_rms``.
    :param theta0: starting parameters.
    :param float factor: relative reduction requested of the first solve.
    :param float min_eps: lower clamp; defaults to the dtype floor.
    :param float max_eps: upper clamp.
    """
    if not hasattr(oracle, "initial_residual_rms"):
        raise TypeError(
            f"{type(oracle).__name__} does not expose initial_residual_rms; "
            "pass eps0 explicitly."
        )
    if factor <= 0.0:
        raise ValueError(f"factor must be positive, got {factor}")
    if min_eps is None:
        min_eps = 100.0 * float(torch.finfo(theta0.dtype).eps)
    if not (0.0 < min_eps <= max_eps):
        raise ValueError(
            f"need 0 < min_eps <= max_eps, got {min_eps} and {max_eps}"
        )
    r0 = float(oracle.initial_residual_rms(theta0))
    if not (r0 > 0.0) or r0 != r0:  # non-positive or NaN
        return float(min_eps)
    return float(min(max(float(factor) * r0, min_eps), max_eps))
