"""Convex ridge regulariser for bilevel learning (Goujon-Unser / WCRR structure).

Architecture follows the reference CRR/WCRR implementation
(LearnedRegularizers, priors/wcrr.py): multi-layer convolution with no
nonlinearity between layers, Lipschitz normalisation, log-domain scaling,
and a smoothed-L1 potential.

Parameters are packed in a flat theta vector for MAID. The lower level also
carries a ridge floor (gamma/2)||x||^2 so mu >= gamma.

Default weak_convexity=0 yields CRR (convex).

Log-scale ``s`` and the smooth_l1 knee
--------------------------------------
The energy piece for one filter response is

    exp(-2 s) * smooth_l1( exp(s) * u )

with ``smooth_l1(t) = t^2/2`` for ``|t| < 1`` and ``|t| - 1/2`` for
``|t| >= 1``. In the **quadratic region** this cancels ``s`` identically:

    exp(-2 s) * sum smooth_l1( exp(s) * W x )
        = exp(-2 s) * exp(2 s) * sum (W x)^2 / 2
        = sum (W x)^2 / 2.

That identity is exact inside the knee. Whether a configuration sits there
depends on ``sigma`` (which sets the initial ``s`` via
``s0 = log(2 / sigma)``) relative to the filter-response scale after Lip
normalisation, and on the operating point of the image. It is not an
inherent limitation of the prior: with the reference ``sigma = 0.1``
responses often reach the linear tails and ``s`` becomes identifiable.
Pushing the knee out (large ``sigma``) keeps responses quadratic and
turns the ridge into a generalised Tikhonov penalty on filter responses.

Measured spread of the regulariser energy over ``s in {-1, 0, +1}`` as the
filter response scale varies relative to the smooth_l1 unit knee (mu = 1
in the table):

    filter response / knee | spread of reg over s in {-1, 0, +1}
    ---------------------- | -----------------------------------
    1e-3                   | 0
    1e-2                   | ~2e-16
    1e-1                   | ~3e-16
    1                      | ~0.25
    10                     | ~0.83

Lipschitz normalisation of the multiconv pins the linear map and removes
the filter-amplitude / scale degeneracy of a free convolution bank. The
example reports initial and final ``exp(s)``. Motion of ``s`` means
responses reached the knee; flat ``s`` means they stayed quadratic at that
configuration.

Lipschitz normalisation
-----------------------
``apply_conv`` scales the input by ``1/sqrt(lip)``. By default
:meth:`ConvexRidgePrior.load_theta` refreshes ``lip`` from the loaded
weights, so ``W / sqrt(lip(W))`` stays unit-norm for every ``theta`` and the
model is invariant under ``W -> c W``. The prior step-size chart is then

    L_prior = 2 * exp(beta) + gamma

independent of the weight scale.

The differentiable energy recomputes ``lip`` with the autograd graph
attached (see :func:`ridge_energy`), so the hypergradient carries
``d lip / d w`` rather than treating the normalisation as a constant.
Detaching a recomputed ``lip`` would make the forward model and the
differentiated model disagree: the solve would use ``lip(theta)`` while the
hypergradient dropped that dependence.

:meth:`ConvexRidgePrior.refresh_lip` recomputes and stores ``lip``
explicitly. Call it only between outer iterations, never inside a
lower-level solve or a hypergradient evaluation.

DeepInverse adaptations
-----------------------
Flat theta, per-``theta`` Lip normalisation, gamma ridge floor, colour
default channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


def _as_tuple(x: Sequence[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(v) for v in x)


def n_crr_params(
    nb_channels: Sequence[int] = (3, 4, 8, 64),
    filter_sizes: Sequence[int] = (5, 5, 5),
) -> int:
    """Number of free parameters in the flat theta vector."""
    ch = _as_tuple(nb_channels)
    fs = _as_tuple(filter_sizes)
    if len(ch) != len(fs) + 1:
        raise ValueError(
            f"len(nb_channels)={len(ch)} must be len(filter_sizes)+1={len(fs)+1}"
        )
    n_w = 0
    for i, k in enumerate(fs):
        n_w += ch[i + 1] * ch[i] * k * k
    return n_w + ch[-1] + 1


def smooth_l1(x: torch.Tensor) -> torch.Tensor:
    """Smoothed L1: t^2/2 for |t|<1, |t|-1/2 otherwise."""
    return torch.clamp(x * x, 0.0, 1.0) / 2.0 + torch.clamp(x.abs(), min=1.0) - 1.0


def grad_smooth_l1(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, -1.0, 1.0)


@dataclass
class ConvexRidgeConfig:
    """Architecture and fixed constants for the multiconv ridge regulariser."""

    nb_channels: tuple[int, ...] = (3, 4, 8, 64)
    filter_sizes: tuple[int, ...] = (5, 5, 5)
    gamma: float = 1e-2
    weak_convexity: float = 0.0
    sigma_init: float = 0.1
    beta_init: float = 4.0
    lip_fft_size: int = 128

    def __post_init__(self) -> None:
        self.nb_channels = _as_tuple(self.nb_channels)
        self.filter_sizes = _as_tuple(self.filter_sizes)
        if len(self.nb_channels) != len(self.filter_sizes) + 1:
            raise ValueError(
                "nb_channels must have length len(filter_sizes)+1, "
                f"got {len(self.nb_channels)} and {len(self.filter_sizes)}"
            )
        if not (0.0 <= self.weak_convexity <= 1.0):
            raise ValueError("weak_convexity must lie in [0, 1]")

    @property
    def n_params(self) -> int:
        return n_crr_params(self.nb_channels, self.filter_sizes)

    @property
    def n_filters(self) -> int:
        return int(self.nb_channels[-1])

    @property
    def effective_filter_size(self) -> int:
        return int(sum(self.filter_sizes) - len(self.filter_sizes) + 1)

    @property
    def in_channels(self) -> int:
        return int(self.nb_channels[0])


def _weight_layout(cfg: ConvexRidgeConfig) -> list[tuple[int, int, int, int, int]]:
    layout = []
    n = 0
    for i, k in enumerate(cfg.filter_sizes):
        cout = cfg.nb_channels[i + 1]
        cin = cfg.nb_channels[i]
        n_w = cout * cin * k * k
        layout.append((n, n + n_w, cout, cin, k))
        n += n_w
    return layout


def unpack_weights(theta: torch.Tensor, cfg: ConvexRidgeConfig) -> list[torch.Tensor]:
    weights = []
    for i, (start, end, cout, cin, k) in enumerate(_weight_layout(cfg)):
        w = theta[start:end].reshape(cout, cin, k, k)
        if i == 0:
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)
        weights.append(w)
    return weights


def unpack_scaling_beta(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    n_w = sum(end - start for start, end, *_ in _weight_layout(cfg))
    n_s = cfg.n_filters
    scaling = theta[n_w : n_w + n_s].view(1, n_s, 1, 1)
    beta = theta[n_w + n_s : n_w + n_s + 1].view(())
    return scaling, beta


def unpack_theta(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D, got shape {tuple(theta.shape)}")
    if theta.numel() != cfg.n_params:
        raise ValueError(f"theta has {theta.numel()} entries, expected {cfg.n_params}")
    return unpack_weights(theta, cfg), *unpack_scaling_beta(theta, cfg)


def pack_init_theta(
    cfg: ConvexRidgeConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    weight_scale: float = 0.05,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    parts: list[torch.Tensor] = []
    for start, end, cout, cin, k in _weight_layout(cfg):
        parts.append(
            weight_scale * torch.randn(end - start, generator=gen, dtype=dtype)
        )
    s0 = float(torch.log(torch.tensor(2.0 / cfg.sigma_init)))
    parts.append(torch.full((cfg.n_filters,), s0, dtype=dtype))
    parts.append(torch.tensor([cfg.beta_init], dtype=dtype))
    return torch.cat(parts).to(device=device, dtype=dtype)


def get_conv_lip(
    weights: list[torch.Tensor],
    cfg: ConvexRidgeConfig,
    detach: bool = True,
) -> torch.Tensor:
    fs = cfg.effective_filter_size
    device = weights[0].device
    dtype = weights[0].dtype
    cin = cfg.in_channels
    dirac = torch.zeros(1, cin, 2 * fs - 1, 2 * fs - 1, device=device, dtype=dtype)
    dirac[0, 0, fs - 1, fs - 1] = 1.0
    impulse = dirac
    for w in weights:
        pad = w.shape[-1] // 2
        impulse = F.conv2d(impulse, w, padding=pad)
    for w in reversed(weights):
        pad = w.shape[-1] // 2
        impulse = F.conv_transpose2d(impulse, w, padding=pad)
    if detach:
        impulse = impulse.detach()
    n = max(int(cfg.lip_fft_size), 2 * fs)
    return torch.fft.fft2(impulse, s=[n, n]).abs().amax().clamp_min(1e-12)


def apply_conv(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    cfg: ConvexRidgeConfig,
    lip: torch.Tensor | None = None,
) -> torch.Tensor:
    if lip is None:
        lip = get_conv_lip(weights, cfg, detach=True)
    x = x / torch.sqrt(lip)
    for w in weights:
        pad = w.shape[-1] // 2
        x = F.conv2d(x, w, padding=pad)
    return x


def apply_conv_transpose(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    cfg: ConvexRidgeConfig,
    lip: torch.Tensor | None = None,
) -> torch.Tensor:
    if lip is None:
        lip = get_conv_lip(weights, cfg, detach=True)
    x = x / torch.sqrt(lip)
    for w in reversed(weights):
        pad = w.shape[-1] // 2
        x = F.conv_transpose2d(x, w, padding=pad)
    return x


def ridge_energy(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scaling: torch.Tensor,
    beta: torch.Tensor,
    cfg: ConvexRidgeConfig,
    lip: torch.Tensor | None = None,
) -> torch.Tensor:
    if lip is None:
        # detach=False so that when the weights carry a graph (the mixed
        # Jacobian path) the normalisation is differentiated rather than
        # treated as a constant. When the weights are detached, this is free.
        lip = get_conv_lip(weights, cfg, detach=False)
    feats = apply_conv(x, weights, cfg, lip=lip)
    scaled = feats * torch.exp(scaling)
    rho = smooth_l1(torch.exp(beta) * scaled) * torch.exp(-beta)
    if cfg.weak_convexity != 0.0:
        rho = rho - smooth_l1(scaled) * cfg.weak_convexity
    reg = (rho * torch.exp(-2.0 * scaling)).sum()
    return reg + 0.5 * cfg.gamma * (x * x).sum()


def ridge_grad_x(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scaling: torch.Tensor,
    beta: torch.Tensor,
    cfg: ConvexRidgeConfig,
    lip: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit gradient matching the reference WCRR ``grad`` path.

    Avoids autograd on the lower-level residual loop (which would re-run the
    multiconv graph thousands of times).
    """
    w_det = [w.detach() for w in weights]
    s = scaling.detach()
    b = beta.detach()
    if lip is None:
        lip = get_conv_lip(w_det, cfg, detach=True)
    feats = apply_conv(x.detach(), w_det, cfg, lip=lip)
    scaled = feats * torch.exp(s)
    g = grad_smooth_l1(torch.exp(b) * scaled)
    if cfg.weak_convexity != 0.0:
        g = g - grad_smooth_l1(scaled) * cfg.weak_convexity
    g = g * torch.exp(-s)
    g = apply_conv_transpose(g, w_det, cfg, lip=lip)
    g = g + cfg.gamma * x.detach()
    return g


class ConvexRidgePrior:
    """Thin energy/grad wrapper driven by a flat theta vector.

    Lipschitz normalisation stores a scalar ``_lip`` used by energy and
    gradient evaluation (see module docstring). By default
    :meth:`load_theta` refreshes it from the loaded weights so
    ``W / sqrt(lip(W))`` stays unit-norm. Differentiable paths recompute
    ``lip`` with the graph attached so the hypergradient includes
    ``d lip / d w``.
    """

    def __init__(self, cfg: ConvexRidgeConfig | None = None):
        self.cfg = cfg if cfg is not None else ConvexRidgeConfig()
        self._weights: list[torch.Tensor] | None = None
        self._scaling: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None
        # Multiconv Lip used by apply_conv / apply_conv_transpose for the
        # non-differentiated energy and gradient paths.
        self._lip: torch.Tensor | None = None

    def refresh_lip(self, weights: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Recompute and store ``lip`` from the given (or loaded) weights.

        Call only between outer iterations, never inside a lower-level solve
        or a hypergradient evaluation.
        """
        if weights is None:
            if self._weights is None:
                raise RuntimeError("call load_theta before refresh_lip")
            weights = self._weights
        self._lip = get_conv_lip(weights, self.cfg, detach=True)
        return self._lip

    def load_theta(self, theta: torch.Tensor, *, refresh_lip: bool = True) -> None:
        """Unpack theta into weights, scaling and beta.

        By default ``lip`` is recomputed from the loaded weights, so the
        normalised operator ``W / sqrt(lip(W))`` is unit-norm for every
        ``theta`` and the model is invariant under ``W -> c W``. The
        differentiable energy recomputes ``lip`` with the graph attached, so
        the hypergradient carries ``d lip / d w``.

        Keeping a stale ``lip_0`` (``refresh_lip=False`` after the first
        load) makes the forward map exact for that fixed normalisation, but
        removes the scale invariance: prior curvature then grows as
        ``lip(W) / lip_0``, and a line-search trial that inflates the
        weights can drive the step size to zero.
        """
        w, s, b = unpack_theta(theta, self.cfg)
        self._weights = w
        self._scaling = s
        self._beta = b
        if self._lip is None or refresh_lip:
            self._lip = get_conv_lip(w, self.cfg, detach=True)

    @property
    def lip(self) -> torch.Tensor:
        """Stored Lip normalisation constant (scalar tensor)."""
        if self._lip is None:
            raise RuntimeError("call load_theta before reading lip")
        return self._lip

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if self._weights is None or self._scaling is None or self._beta is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_energy(
            x, self._weights, self._scaling, self._beta, self.cfg, lip=self._lip
        )

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        if self._weights is None or self._scaling is None or self._beta is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_grad_x(
            x, self._weights, self._scaling, self._beta, self.cfg, lip=self._lip
        )

    def current_weight_lip(self) -> float:
        """Composed multiconv Lip of the loaded weights (no gradient)."""
        if self._weights is None:
            raise RuntimeError("call load_theta before current_weight_lip")
        return float(get_conv_lip(self._weights, self.cfg, detach=True).item())

    def lipschitz_bound(self) -> float:
        """Prior step-size chart under per-``theta`` Lip normalisation.

        When ``lip`` is refreshed with the weights, ``W / sqrt(lip(W))`` is
        unit-norm and the bound is independent of the weight scale:

            L_prior = 2 * exp(beta) + gamma
        """
        if self._beta is None or self._weights is None or self._lip is None:
            raise RuntimeError("call load_theta before lipschitz_bound")
        return float(2.0 * torch.exp(self._beta).item() + self.cfg.gamma)


def scaling_vector(theta: torch.Tensor, cfg: ConvexRidgeConfig) -> torch.Tensor:
    s, _ = unpack_scaling_beta(theta, cfg)
    return s.view(-1)


def exp_scaling(theta: torch.Tensor, cfg: ConvexRidgeConfig) -> torch.Tensor:
    return torch.exp(scaling_vector(theta, cfg))
