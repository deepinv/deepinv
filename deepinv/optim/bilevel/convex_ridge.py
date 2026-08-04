"""Convex ridge regulariser for bilevel learning (Goujon-Unser / WCRR structure).

Architecture follows the reference CRR/WCRR implementation
(``LearnedRegularizers``, ``priors/wcrr.py``): a composition of convolutions
with no nonlinearity between layers (so the map stays linear and the
regulariser stays convex), Lipschitz normalisation of that map, a log-domain
scaling, and a smoothed-L1 potential.

.. math::

    u = \mathrm{LipNorm}(W_L \cdots W_1 x)

    R_\theta(x)
    = \sum_j e^{-2 s_{c(j)}}
      \Bigl(
        e^{-\beta}\,\rho\bigl(e^{\beta} e^{s_{c(j)}} u_j\bigr)
        - \kappa\,\rho\bigl(e^{s_{c(j)}} u_j\bigr)
      \Bigr)

    \rho(t) = \mathrm{smooth\_l1}(t)

with free parameters ``theta`` packing the multiconv weights, the log-scale
vector ``s`` and the log-temperature ``beta``. The lower level also carries a
ridge floor ``(gamma / 2) ||x||^2`` so the strong-convexity modulus satisfies
``mu >= gamma`` independent of the learned weights (required by MAID's
certified residual).

Default ``kappa = weak_convexity = 0`` yields a convex ridge regulariser
(CRR). Setting ``kappa = 1`` recovers the weakly convex variant (WCRR); that
path is not used by the bilevel demos because MAID's theory needs a convex
lower level.

Why this structure (and how it differs from a single-layer FoE)
--------------------------------------------------------------
1. **Multi-layer convolution.** ``nb_channels`` with ``filter_sizes`` builds a
   deeper linear feature map than one bank of kernels, at the cost of more
   free parameters. Convexity in ``x`` is preserved because there is no
   pointwise nonlinearity between layers.
2. **Lipschitz normalisation.** ``conv`` and ``conv_transpose`` divide by
   ``sqrt(get_conv_lip())``, where the constant is the spectral norm of the
   multiconv estimated by an impulse response and an FFT. This removes the
   scale degeneracy between filter amplitude and the learned scale: the
   operator is pinned, so scale lives in ``s``.
3. **Log scaling.** ``s`` is free; the feature map is multiplied by
   ``exp(s)`` and the energy by ``exp(-2 s)``. Same positivity geometry as
   ``lambda = exp(theta)`` on the scalar Tikhonov problem.
4. **``smooth_l1``** rather than log-cosh: matches the reference profile
   (quadratic near zero, linear in the tails, ``C^1``).
5. **Zero-mean first layer.** Constant shifts of the image sit in the null
   space of the first convolution, so they are not penalised.

DeepInverse adaptations (deliberate departures from the reference module)
-------------------------------------------------------------------------
* Parameters are a **flat** ``theta`` vector, not an ``nn.Module`` state dict,
  so MAID and the IFT hypergradient see one Euclidean parameter space.
* The Lip constant is **detached** when normalising. The reference lets the
  gradient flow through the FFT estimate; that path is poorly conditioned
  under ``create_graph=True`` HVP and is unnecessary once the operator is
  treated as a constant chart for the current weights.
* A **gamma ridge floor** is added so residual-stopped lower levels have a
  known ``mu`` (the reference prior alone is only convex, not strongly
  convex).
* Default channel stack starts at 3 for colour images; pass
  ``nb_channels=(1, ...)`` for greyscale.

Scale degeneracy
----------------
Without Lip normalisation, scaling every filter by ``c`` and adjusting
``s`` by ``-log c`` leaves the energy unchanged, so the hypergradient along
the scale coordinate is flat and the optimiser moves only the filters. Lip
normalisation pins the operator; the acceptance test in the example reports
initial and final ``exp(s)`` and flags if they barely move.
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
    n_scale = ch[-1]
    n_beta = 1
    return n_w + n_scale + n_beta


def smooth_l1(x: torch.Tensor) -> torch.Tensor:
    """Smoothed L1 potential: ``t^2/2`` for ``|t|<1``, ``|t|-1/2`` otherwise."""
    return torch.clamp(x * x, 0.0, 1.0) / 2.0 + torch.clamp(x.abs(), min=1.0) - 1.0


def grad_smooth_l1(x: torch.Tensor) -> torch.Tensor:
    """Gradient of :func:`smooth_l1` (clip to [-1, 1])."""
    return torch.clamp(x, -1.0, 1.0)


@dataclass
class ConvexRidgeConfig:
    """Architecture and fixed constants for the multiconv ridge regulariser.

    :param tuple nb_channels: channel counts through the multiconv, length
        ``len(filter_sizes)+1``. Default ``(3, 4, 8, 64)`` for colour.
    :param tuple filter_sizes: spatial size of each convolution.
    :param float gamma: strong-convexity floor in the lower level.
    :param float weak_convexity: ``0`` for CRR, ``1`` for WCRR. Bilevel
        demos use ``0``.
    :param float sigma_init: reference scale for the initial log-scaling
        ``s0 = log(2 / sigma_init)``.
    :param float beta_init: initial log-temperature of the smoothed L1.
    :param int lip_fft_size: FFT size for the spectral-norm estimate.
    """

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
        if self.weak_convexity < 0.0 or self.weak_convexity > 1.0:
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


def _weight_layout(
    cfg: ConvexRidgeConfig,
) -> list[tuple[int, int, int, int, int]]:
    """List of (start, end, cout, cin, k) for each multiconv layer."""
    layout = []
    n = 0
    for i, k in enumerate(cfg.filter_sizes):
        cout = cfg.nb_channels[i + 1]
        cin = cfg.nb_channels[i]
        n_w = cout * cin * k * k
        layout.append((n, n + n_w, cout, cin, k))
        n += n_w
    return layout


def unpack_weights(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> list[torch.Tensor]:
    """Unpack multiconv weights; zero-mean the first layer (per out-channel)."""
    weights = []
    for i, (start, end, cout, cin, k) in enumerate(_weight_layout(cfg)):
        w = theta[start:end].reshape(cout, cin, k, k)
        if i == 0:
            # ZeroMean on the first layer: mean over (cin, h, w).
            w = w - w.mean(dim=(1, 2, 3), keepdim=True)
        weights.append(w)
    return weights


def unpack_scaling_beta(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(scaling, beta)`` views from the flat theta tail."""
    n_w = sum(end - start for start, end, *_ in _weight_layout(cfg))
    n_s = cfg.n_filters
    scaling = theta[n_w : n_w + n_s].view(1, n_s, 1, 1)
    beta = theta[n_w + n_s : n_w + n_s + 1].view(())
    return scaling, beta


def unpack_theta(
    theta: torch.Tensor, cfg: ConvexRidgeConfig
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Map flat theta to ``(weights, scaling, beta)``."""
    if theta.ndim != 1:
        raise ValueError(f"theta must be 1-D, got shape {tuple(theta.shape)}")
    if theta.numel() != cfg.n_params:
        raise ValueError(
            f"theta has {theta.numel()} entries, expected {cfg.n_params}"
        )
    weights = unpack_weights(theta, cfg)
    scaling, beta = unpack_scaling_beta(theta, cfg)
    return weights, scaling, beta


def pack_init_theta(
    cfg: ConvexRidgeConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    weight_scale: float = 0.05,
) -> torch.Tensor:
    """Random multiconv weights; scaling and beta at the reference defaults.

    Scaling is initialised to ``log(2 / sigma_init)`` on every filter channel
    (matches the reference ``WCRR``). Beta starts at ``beta_init``.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    parts: list[torch.Tensor] = []
    for start, end, cout, cin, k in _weight_layout(cfg):
        n_w = end - start
        parts.append(weight_scale * torch.randn(n_w, generator=gen, dtype=dtype))
    s0 = float(torch.log(torch.tensor(2.0 / cfg.sigma_init)))
    parts.append(torch.full((cfg.n_filters,), s0, dtype=dtype))
    parts.append(torch.tensor([cfg.beta_init], dtype=dtype))
    return torch.cat(parts).to(device=device, dtype=dtype)


def get_conv_lip(
    weights: list[torch.Tensor],
    cfg: ConvexRidgeConfig,
    detach: bool = True,
) -> torch.Tensor:
    """Spectral-norm estimate of the multiconv via impulse response + FFT."""
    fs = cfg.effective_filter_size
    device = weights[0].device
    dtype = weights[0].dtype
    cin = cfg.in_channels
    # Impulse on the first input channel; other channels zero. For colour
    # this underestimates the joint operator slightly; a full block estimate
    # is more expensive and the detached chart only needs a stable scale.
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
    spec = torch.fft.fft2(impulse, s=[n, n]).abs().amax()
    return spec.clamp_min(1e-12)


def apply_conv(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    cfg: ConvexRidgeConfig,
) -> torch.Tensor:
    """Lipschitz-normalised multiconv (forward)."""
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
) -> torch.Tensor:
    """Lipschitz-normalised multiconv adjoint."""
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
) -> torch.Tensor:
    """Scalar energy ``R(x) + (gamma/2)||x||^2`` (batch summed)."""
    feats = apply_conv(x, weights, cfg)
    scaled = feats * torch.exp(scaling)
    rho = smooth_l1(torch.exp(beta) * scaled) * torch.exp(-beta)
    if cfg.weak_convexity != 0.0:
        rho = rho - smooth_l1(scaled) * cfg.weak_convexity
    reg = (rho * torch.exp(-2.0 * scaling)).sum()
    ridge = 0.5 * cfg.gamma * (x * x).sum()
    return reg + ridge


def ridge_grad_x(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scaling: torch.Tensor,
    beta: torch.Tensor,
    cfg: ConvexRidgeConfig,
) -> torch.Tensor:
    """Gradient of the ridge energy in ``x`` (detached weights/scale)."""
    x_ = x.detach().requires_grad_(True)
    w_det = [w.detach() for w in weights]
    e = ridge_energy(x_, w_det, scaling.detach(), beta.detach(), cfg)
    (g,) = torch.autograd.grad(e, x_)
    return g.detach()


class ConvexRidgePrior:
    """Thin energy/grad wrapper driven by a flat theta vector.

    Not a full :class:`~deepinv.optim.Prior` subclass: the bilevel path uses
    hand-rolled residual-stopped GD that only needs ``grad`` and the energy,
    which avoids fighting BaseOptim's ``lambda * prior.grad`` scaling.
    """

    def __init__(self, cfg: ConvexRidgeConfig | None = None):
        self.cfg = cfg if cfg is not None else ConvexRidgeConfig()
        self._weights: list[torch.Tensor] | None = None
        self._scaling: torch.Tensor | None = None
        self._beta: torch.Tensor | None = None

    def load_theta(self, theta: torch.Tensor) -> None:
        w, s, b = unpack_theta(theta, self.cfg)
        self._weights = w
        self._scaling = s
        self._beta = b

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        if self._weights is None or self._scaling is None or self._beta is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_energy(
            x, self._weights, self._scaling, self._beta, self.cfg
        )

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        if self._weights is None or self._scaling is None or self._beta is None:
            raise RuntimeError("call load_theta before energy/grad")
        return ridge_grad_x(
            x, self._weights, self._scaling, self._beta, self.cfg
        )

    def lipschitz_bound(self) -> float:
        """Conservative Lip of ``grad R`` for GD stepsize selection.

        After Lip normalisation of the multiconv, the map has operator norm
        at most 1. The second derivative of the scaled smooth-L1 profile is
        bounded by ``exp(beta)``, and the gamma ridge adds ``gamma``. A
        safety factor of 2 covers multi-channel accumulation and the
        detached spectral estimate.
        """
        if self._beta is None:
            raise RuntimeError("call load_theta before lipschitz_bound")
        return float(2.0 * torch.exp(self._beta).item() + self.cfg.gamma)


def scaling_vector(theta: torch.Tensor, cfg: ConvexRidgeConfig) -> torch.Tensor:
    """Return the length-``n_filters`` log-scaling vector from theta."""
    s, _ = unpack_scaling_beta(theta, cfg)
    return s.view(-1)


def exp_scaling(theta: torch.Tensor, cfg: ConvexRidgeConfig) -> torch.Tensor:
    """Return ``exp(s)`` for reporting."""
    return torch.exp(scaling_vector(theta, cfg))
