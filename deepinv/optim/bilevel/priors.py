"""Parametric priors for bilevel learning, and a prior-agnostic problem.

A prior supplies one thing: a scalar energy per sample, differentiable in the
image and in its parameters. Everything the bilevel machinery needs is
obtained from that single function by automatic differentiation, so the
forward model and the differentiated model cannot disagree:

* the lower-level gradient is ``autograd.grad(energy, x)``
* the Hessian-vector product is a second backward pass in ``x``
* the mixed Jacobian is a backward pass in ``x`` followed by one in ``theta``

Adding a prior therefore means implementing :meth:`ParametricPrior.energy`
and an initialisation. No derivative is written by hand.

Requirements on the energy
--------------------------
Convex in ``x``, so the lower level has a unique minimiser and the
strong-convexity certificate applies. Twice differentiable in ``x``, so the
Hessian-vector product exists. Differentiable in ``theta``, so the
hypergradient exists. Priors that are convex only for certain parameter values
must enforce that by construction, for instance through a positive
reparameterisation, rather than relying on the optimiser to stay in the
feasible region.

Included priors
---------------
:class:`ConvexRidgePrior2` multi-convolution ridge regulariser.
:class:`LearnedTVPrior` smoothed total variation with learned channel weights.
:class:`ICNNPrior` input-convex neural network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from .convex_ridge import (
    ConvexRidgeConfig,
    apply_conv,
    get_conv_lip,
    smooth_l1,
    unpack_theta,
)


class ParametricPrior(ABC):
    """A regulariser with a flat parameter vector.

    Subclasses define the energy and an initialisation; the bilevel machinery
    derives every derivative from the energy by autograd.
    """

    #: Number of entries in the flat parameter vector.
    n_params: int

    @abstractmethod
    def energy(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Energy per sample, shape ``(B,)``, for images ``x`` of ``(B, C, H, W)``."""

    @abstractmethod
    def init_theta(
        self,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> torch.Tensor:
        """A starting parameter vector."""

    def curvature_bound(self, theta: torch.Tensor) -> float:
        """Upper bound on the top eigenvalue of ``grad^2_x`` of the energy.

        Used only for the gradient and FISTA step sizes. Truncated Newton does
        not need it. The default is deliberately absent rather than guessed:
        a bound that is too small makes those solvers unstable.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide a curvature bound; "
            "use the NEWTON lower-level solver, which does not require one."
        )


class ConvexRidgePrior2(ParametricPrior):
    """Multi-convolution ridge regulariser, as :mod:`convex_ridge`.

    Present so the reference prior and the new ones share exactly one code
    path, which is what makes the comparison between them meaningful.
    """

    def __init__(self, cfg: ConvexRidgeConfig):
        self.cfg = cfg
        self.n_params = cfg.n_params

    def energy(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        weights, scaling, beta = unpack_theta(theta, self.cfg)
        lip = get_conv_lip(weights, self.cfg, detach=False)
        feats = apply_conv(x, weights, self.cfg, lip=lip)
        scaled = feats * torch.exp(scaling)
        rho = smooth_l1(torch.exp(beta) * scaled) * torch.exp(-beta)
        if self.cfg.weak_convexity != 0.0:
            rho = rho - smooth_l1(scaled) * self.cfg.weak_convexity
        return (rho * torch.exp(-2.0 * scaling)).flatten(1).sum(dim=1)

    def init_theta(self, *, dtype=torch.float64, device="cpu", seed=0):
        from .convex_ridge import pack_init_theta

        return pack_init_theta(
            self.cfg, device=device, dtype=dtype, seed=seed, weight_scale=1.0
        )

    def curvature_bound(self, theta: torch.Tensor) -> float:
        _w, _s, beta = unpack_theta(theta, self.cfg)
        return float(2.0 * torch.exp(beta).item())


def _forward_differences(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal and vertical forward differences, zero at the far edge."""
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


class LearnedTVPrior(ParametricPrior):
    r"""Smoothed isotropic total variation with learned per-channel weights.

    .. math::

        R_\theta(x) = \sum_{c} w_c \sum_{ij}
        \sqrt{(D_h x)_{cij}^2 + (D_v x)_{cij}^2 + \eta^2} - \eta

    with :math:`w_c = \exp(\theta_c)`. The exponential keeps the weights
    positive, which is what makes the energy convex in ``x`` for every value
    of ``theta``; a free weight could turn negative and destroy convexity
    mid-training.

    The Huber-style smoothing :math:`\eta` makes the energy twice
    differentiable, as the Hessian-vector product requires. Plain TV is not.
    Subtracting :math:`\eta` sets the energy of a constant image to zero.

    ``n_params = C``, so this is a very small model: one weight per channel.
    It is included as the simplest non-trivial learnable prior, and as a
    like-for-like comparison against the grid-tuned TV baseline, which is the
    same energy with a single shared weight.
    """

    def __init__(self, channels: int = 3, eta: float = 1e-3):
        if eta <= 0:
            raise ValueError(f"eta must be positive for smoothness, got {eta}")
        self.channels = int(channels)
        self.eta = float(eta)
        self.n_params = int(channels)

    def energy(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.numel() != self.n_params:
            raise ValueError(
                f"theta has {theta.numel()} entries, expected {self.n_params}"
            )
        gx, gy = _forward_differences(x)
        mag = torch.sqrt(gx * gx + gy * gy + self.eta**2) - self.eta
        w = torch.exp(theta).view(1, -1, 1, 1)
        return (w * mag).flatten(1).sum(dim=1)

    def init_theta(self, *, dtype=torch.float64, device="cpu", seed=0):
        # exp(theta) = 0.03, the scale at which grid-tuned TV lands on natural
        # images at sigma = 0.05.
        import math

        return torch.full((self.n_params,), math.log(0.03), dtype=dtype, device=device)

    def curvature_bound(self, theta: torch.Tensor) -> float:
        # |grad^2 sqrt(t^2 + eta^2)| <= 1/eta, and ||D||^2 <= 8 for forward
        # differences in two dimensions.
        w_max = float(torch.exp(theta).max().item())
        return float(8.0 * w_max / self.eta)


class ICNNPrior(ParametricPrior):
    r"""Input-convex neural network regulariser.

    .. math::

        z_1 = s(W_0 x + b_0), \quad
        z_{k+1} = s(W_k^+ z_k + U_k x + b_k), \quad
        R_\theta(x) = \mathbf{1}^\top v^+ z_L

    where :math:`s` is softplus and :math:`W^+ = \mathrm{softplus}(W)`.

    Convexity in ``x`` holds by construction, following Amos, Xu and Kolter
    (2017): a convex non-decreasing function of a convex function is convex,
    so the hidden-to-hidden weights must be non-negative while the direct
    "skip" weights :math:`U_k` from the input are unconstrained. Softplus is
    convex, non-decreasing and smooth, so the energy is twice differentiable
    as the Hessian-vector product requires; ReLU would give convexity but not
    the second derivative.

    Convolutional layers keep the parameter count independent of image size
    and make the prior translation equivariant, so a prior learned on small
    patches applies to whole images.
    """

    def __init__(
        self,
        channels: int = 3,
        hidden: int = 16,
        depth: int = 2,
        kernel: int = 3,
    ):
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.channels, self.hidden = int(channels), int(hidden)
        self.depth, self.kernel = int(depth), int(kernel)
        k = self.kernel
        self._shapes: list[tuple[str, tuple[int, ...]]] = [
            ("W0", (self.hidden, self.channels, k, k)),
            ("b0", (self.hidden,)),
        ]
        for i in range(self.depth):
            self._shapes += [
                (f"W{i+1}", (self.hidden, self.hidden, k, k)),
                (f"U{i+1}", (self.hidden, self.channels, k, k)),
                (f"b{i+1}", (self.hidden,)),
            ]
        self._shapes.append(("v", (self.hidden,)))
        self.n_params = sum(int(torch.tensor(s).prod()) for _n, s in self._shapes)

    def _unpack(self, theta: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for name, shape in self._shapes:
            n = int(torch.tensor(shape).prod())
            out[name] = theta[i : i + n].reshape(shape)
            i += n
        return out

    def energy(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.numel() != self.n_params:
            raise ValueError(
                f"theta has {theta.numel()} entries, expected {self.n_params}"
            )
        p = self._unpack(theta)
        pad = self.kernel // 2
        z = F.softplus(F.conv2d(x, p["W0"], p["b0"], padding=pad))
        for i in range(self.depth):
            wz = F.conv2d(z, F.softplus(p[f"W{i+1}"]), padding=pad)
            ux = F.conv2d(x, p[f"U{i+1}"], p[f"b{i+1}"], padding=pad)
            z = F.softplus(wz + ux)
        v = F.softplus(p["v"]).view(1, -1, 1, 1)
        return (v * z).flatten(1).sum(dim=1)

    def init_theta(self, *, dtype=torch.float64, device="cpu", seed=0):
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        parts = []
        for name, shape in self._shapes:
            n = int(torch.tensor(shape).prod())
            if name.startswith("b"):
                parts.append(torch.zeros(n, dtype=dtype))
            elif name == "v":
                # softplus(-2) ~ 0.13, a small positive output weight.
                parts.append(torch.full((n,), -2.0, dtype=dtype))
            else:
                fan = int(torch.tensor(shape[1:]).prod())
                scale = (1.0 / max(fan, 1)) ** 0.5
                parts.append(torch.randn(n, generator=gen, dtype=dtype) * scale)
        return torch.cat(parts).to(device=device)


class BatchedPriorProblem:
    """A batch of samples sharing one parametric prior and one physics.

    The prior-agnostic counterpart of
    :class:`~deepinv.optim.bilevel.BatchedCRR`: the lower level is

    .. math::

        h(x, \\theta) = \\tfrac12\\|A x - y\\|^2 + R_\\theta(x)
                      + \\tfrac{\\gamma}{2}\\|x\\|^2

    and every derivative comes from that one expression by autograd, so any
    :class:`ParametricPrior` can be substituted without touching the solver,
    the hypergradient or the certificate.

    ``mu = mu_data + gamma`` with ``mu_data = 1`` when ``A = I`` and 0
    otherwise, so a general forward operator needs ``gamma > 0``.
    """

    def __init__(
        self, y, x_star, prior: ParametricPrior, physics=None, gamma: float = 0.0
    ):
        if x_star.dim() != 4:
            raise ValueError(f"expected x_star (B, C, H, W), got {tuple(x_star.shape)}")
        self.y, self.x_star, self.prior = y, x_star, prior
        self.gamma = float(gamma)
        self.dtype, self.device = x_star.dtype, x_star.device
        self.B = int(x_star.shape[0])
        self.n_elem = int(x_star[0].numel())
        if physics is None:
            import deepinv as _dinv

            physics = _dinv.physics.Denoising(
                noise_model=_dinv.physics.GaussianNoise(0.0)
            )
        self.physics = physics
        v = torch.randn_like(x_star)
        Av = self.physics.A(v)
        mu_data = (
            1.0
            if Av.shape == v.shape and torch.allclose(Av, v, atol=1e-12, rtol=0.0)
            else 0.0
        )
        self.mu = mu_data + self.gamma
        if self.mu <= 0.0:
            raise ValueError(
                "lower level is not strongly convex: mu_data + gamma = "
                f"{self.mu}. Set gamma > 0 for a general forward operator."
            )

    def energy_per_sample(self, x, theta):
        r = self.physics.A(x) - self.y
        data = 0.5 * (r * r).flatten(1).sum(dim=1)
        out = data + self.prior.energy(x, theta)
        if self.gamma:
            out = out + 0.5 * self.gamma * (x * x).flatten(1).sum(dim=1)
        return out

    def grad_x(self, x, theta):
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, theta.detach()).sum()
        (g,) = torch.autograd.grad(e, x_)
        return g.detach()

    def hess_matvec(self, x, theta, v):
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, theta.detach()).sum()
        (g,) = torch.autograd.grad(e, x_, create_graph=True)
        (hv,) = torch.autograd.grad((g * v.detach()).sum(), x_)
        return hv.detach()

    def mixed_jac_T(self, x, theta, v):
        th = theta.detach().requires_grad_(True)
        x_ = x.detach().requires_grad_(True)
        e = self.energy_per_sample(x_, th).sum()
        (g,) = torch.autograd.grad(e, x_, create_graph=True)
        (jtv,) = torch.autograd.grad((g * v.detach()).sum(), th)
        return jtv.detach()

    def grad_g(self, x):
        return x - self.x_star

    def g_per_sample(self, x):
        d = x - self.x_star
        return 0.5 * (d * d).flatten(1).sum(dim=1)

    def solve_lower(
        self,
        theta,
        eps,
        *,
        x_init=None,
        max_iter=5000,
        cg_iters=20,
        armijo_c=1e-4,
        armijo_rho=0.5,
        max_bt=20,
        stall_patience=25,
    ):
        """Batched truncated Newton, per-sample Armijo and stall detection."""
        from .cg_utils import cg_solve_batched

        scale = float(self.n_elem) ** 0.5
        eps_eff = max(float(eps), 100.0 * float(torch.finfo(self.dtype).eps))
        tol = eps_eff * self.mu * scale
        x = (
            self.physics.A_adjoint(self.y).detach().clone()
            if x_init is None
            else x_init.detach().clone()
        )
        res = self.grad_x(x, theta).flatten(1).norm(dim=1)
        best_res, best_x = res.clone(), x.clone()
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
            cg = cg_solve_batched(
                lambda v: self.hess_matvec(x, theta, v),
                -g,
                tol=torch.clamp(0.1 * res, min=tol * 0.1),
                max_iter=cg_iters,
            )
            p = cg.x
            gTp = (g * p).flatten(1).sum(dim=1)
            p = torch.where((gTp >= 0.0).view(self.B, 1, 1, 1), -g, p)
            gTp = (g * p).flatten(1).sum(dim=1)
            f0 = self.energy_per_sample(x, theta)
            alpha = torch.ones(self.B, device=self.device, dtype=self.dtype)
            acc = torch.zeros(self.B, dtype=torch.bool, device=self.device)
            x_new = x.clone()
            for _bt in range(max_bt):
                if bool(acc.all()):
                    break
                trial = x + alpha.view(self.B, 1, 1, 1) * p
                ok = (
                    self.energy_per_sample(trial, theta) <= f0 + armijo_c * alpha * gTp
                ) & (~acc)
                x_new = torch.where(ok.view(self.B, 1, 1, 1), trial, x_new)
                acc = acc | ok
                alpha = torch.where(acc, alpha, alpha * armijo_rho)
            x = torch.where(acc.view(self.B, 1, 1, 1), x_new, x)
            x = torch.where(live.view(self.B, 1, 1, 1), x, best_x)
            res = self.grad_x(x, theta).flatten(1).norm(dim=1)
            improved = res < best_res * (1.0 - 1e-3)
            best_res = torch.where(improved, res, best_res)
            best_x = torch.where(improved.view(self.B, 1, 1, 1), x, best_x)
            since = torch.where(improved, torch.zeros_like(since), since + 1)
            stalled = stalled | (since >= stall_patience)
        return (
            best_x,
            best_res,
            {
                "n_iter": n_iter,
                "residual_rms": best_res / (scale * self.mu),
                "reached": bool((best_res <= tol * 1.01).all()),
                "n_stalled": int(stalled.sum().item()),
            },
        )

    def hypergradient(self, x, theta, delta, *, max_cg_iter=200):
        from .cg_utils import cg_solve_batched

        scale = float(self.n_elem) ** 0.5
        tol = max(float(delta), 100.0 * float(torch.finfo(self.dtype).eps))
        cg = cg_solve_batched(
            lambda v: self.hess_matvec(x, theta, v),
            self.grad_g(x),
            tol=torch.full(
                (self.B,), tol * self.mu * scale, dtype=self.dtype, device=self.device
            ),
            max_iter=max_cg_iter,
        )
        return -self.mixed_jac_T(x, theta, cg.x)

    def initial_residual_rms(self, theta) -> float:
        x0 = self.physics.A_adjoint(self.y)
        g = self.grad_x(x0, theta)
        return float(g.flatten(1).norm(dim=1).mean()) / (
            float(self.n_elem) ** 0.5 * self.mu
        )
