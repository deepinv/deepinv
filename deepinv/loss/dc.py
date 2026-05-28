from __future__ import annotations

import math

import torch
from torch import Tensor

from deepinv.loss.loss import Loss


class DCLoss(Loss):
    r"""
    Distributional consistency (DC) loss for known measurement noise models,
    as proposed by :footcite:t:`webber2026distributional`.

    Let :math:`q = A(\hat{x})` be the re-measured reconstruction and let
    :math:`y` denote the noisy measurements. The loss computes the
    probability-integral-transform (PIT) of the measurements under the predicted
    measurement distribution and compares the resulting empirical distribution to
    the uniform law on :math:`[0, 1]` through a quantile approximation of the
    Wasserstein-1 distance in logit coordinates.

    The supported distributions are:

    - ``"gaussian"`` for additive Gaussian noise,
    - ``"poisson"`` for Poisson noise,
    - ``"clipped_gaussian"`` for Gaussian noise observed after clipping to
      :math:`[0, 1]`.

    For the Gaussian case,

    .. math::

        y_i \sim \mathcal{N}(q_i, \sigma^2),

    the probability integral transform

    .. math::

        u_i = \Phi\left(\frac{y_i - q_i}{\sigma}\right)

    should be uniformly distributed on :math:`[0, 1]` when the predicted
    measurements are distributionally consistent with the observation model.

    By default, the quantile locations are the deterministic midpoints of a
    regular partition of :math:`[0, 1]`, which makes the loss deterministic.
    For clipped Gaussian noise, measurements exactly at the clipping boundaries
    are mapped to the midpoint of the corresponding boundary ramp.

    :param str distribution: Noise model used to define the PIT. Supported values
        are ``"gaussian"``, ``"poisson"``, and ``"clipped_gaussian"``. The alias
        ``"gaussian_clipped"`` is also accepted.
    :param float, torch.Tensor sigma: Standard deviation of the Gaussian noise for
        the Gaussian-based variants. If ``None``, try to infer it from
        ``physics.noise_model.sigma`` at forward time.
    :param int n_points: Number of PIT quantile points used to cover :math:`[0, 1]`.
        If ``None``, use one point per measurement entry.
    :param float eps: Numerical clamp for PIT values before the logit transform.
    :param float boundary_eps: Width of the clipped boundary ramps used for
        ``distribution="clipped_gaussian"``.

    |sep|

    :Examples:

        >>> import torch
        >>> import deepinv as dinv
        >>> x = torch.randn(1, 2, 16, 16)
        >>> physics = dinv.physics.MRI(
        ...     img_size=x.shape[1:],
        ...     noise_model=dinv.physics.GaussianNoise(0.05),
        ... )
        >>> y = physics(x)
        >>> x_net = physics.A_dagger(y)
        >>> loss = dinv.loss.DCLoss(
        ...     distribution="gaussian", sigma=0.05,
        ... )
        >>> l = loss(y=y, x_net=x_net, physics=physics, model=None)
    """

    def __init__(
        self,
        distribution: str = "gaussian",
        sigma: float | Tensor | None = None,
        n_points: int | None = None,
        eps: float = 1e-4,
        boundary_eps: float = 1e-3,
    ):
        super().__init__()
        self.distribution = self._canonicalize_distribution(distribution)
        self.register_buffer(
            "sigma",
            torch.as_tensor(sigma, dtype=torch.float32) if sigma is not None else None,
        )
        self.n_points = n_points
        self.eps = eps
        self.boundary_eps = boundary_eps

        if self.n_points is not None and self.n_points <= 0:
            raise ValueError("n_points must be strictly positive when provided.")
        if not 0 < self.eps < 0.5:
            raise ValueError("eps must lie strictly between 0 and 0.5.")
        if not 0 < self.boundary_eps < 0.5:
            raise ValueError("boundary_eps must lie strictly between 0 and 0.5.")

    @staticmethod
    def _canonicalize_distribution(distribution: str) -> str:
        aliases = {
            "gaussian": "gaussian",
            "poisson": "poisson",
            "clipped_gaussian": "clipped_gaussian",
            "gaussian_clipped": "clipped_gaussian",
        }
        if distribution not in aliases:
            raise ValueError(
                "distribution must be one of 'gaussian', 'poisson', 'clipped_gaussian', or 'gaussian_clipped'."
            )
        return aliases[distribution]

    def _resolve_sigma(
        self,
        y: Tensor,
        physics=None,
        sigma: float | Tensor | None = None,
    ) -> Tensor | None:
        if self.distribution == "poisson":
            return None

        sigma = (
            sigma
            if sigma is not None
            else (
                self.sigma
                if self.sigma is not None
                else getattr(getattr(physics, "noise_model", None), "sigma", None)
            )
        )
        if sigma is None:
            raise ValueError(
                "DCLoss requires a set Gaussian noise level. Pass sigma explicitly or set physics.noise_model.sigma."
            )

        sigma = torch.as_tensor(sigma, dtype=y.dtype, device=y.device)
        if torch.any(sigma <= 0):
            raise ValueError("sigma must be strictly positive.")
        return sigma

    def _pit_grid(self, n_points: int, device, dtype) -> Tensor:
        intervals = torch.arange(n_points, device=device, dtype=dtype)
        offsets = torch.full((n_points,), 0.5, device=device, dtype=dtype)
        return torch.clamp((intervals + offsets) / n_points, self.eps, 1.0 - self.eps)

    def _gaussian_logit_pit(
        self,
        q: Tensor,
        y: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        sigma = sigma[(...,) + (None,) * (y.dim() - sigma.dim())]
        z = (y - q) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        cdf = torch.clamp(cdf, min=self.eps, max=1.0 - self.eps)

        logit_pit = torch.empty_like(cdf)
        low_mask = cdf <= (self.eps + 1e-12)
        high_mask = cdf >= (1.0 - self.eps - 1e-12)
        mid_mask = ~(low_mask | high_mask)

        logit_pit[mid_mask] = torch.logit(cdf[mid_mask])

        log_norm = 0.5 * math.log(2.0 * math.pi)
        tail = 0.5 * z.square() + torch.log(z.abs().clamp_min(1e-12)) + log_norm
        logit_pit[low_mask] = -tail[low_mask]
        logit_pit[high_mask] = tail[high_mask]
        return logit_pit

    def _poisson_logit_pit(self, q: Tensor, y: Tensor) -> Tensor:
        q = torch.nn.functional.relu(q)
        y = torch.nn.functional.relu(y)
        q_safe = q.clamp_min(1e-12)

        cdf = 1.0 - torch.distributions.Gamma(y + 1.0, torch.ones_like(y)).cdf(q)
        cdf = torch.clamp(cdf, min=self.eps, max=1.0 - self.eps)

        logit_pit = torch.zeros_like(cdf)
        low_mask = cdf <= (self.eps + 1e-12)
        high_mask = cdf >= (1.0 - self.eps - 1e-12)
        mid_mask = ~(low_mask | high_mask)

        logit_pit[mid_mask] = torch.logit(cdf[mid_mask])
        logit_pit[low_mask] = (
            y[low_mask] * torch.log(q_safe[low_mask])
            - q[low_mask]
            - torch.lgamma(y[low_mask] + 1.0)
        )
        logit_pit[high_mask] = (
            q[high_mask]
            - y[high_mask] * torch.log(q_safe[high_mask])
            + torch.lgamma(y[high_mask] + 2.0)
        )
        return logit_pit

    def _prepare_clipped_measurements(self, y: Tensor) -> Tensor:
        y = y.clamp(0.0, 1.0).clone()

        zero_mask = y == 0
        if zero_mask.any():
            y[zero_mask] = 0.5 * self.boundary_eps

        one_mask = y == 1
        if one_mask.any():
            y[one_mask] = 1.0 - 0.5 * self.boundary_eps

        return y

    def _clipped_gaussian_logit_pit(
        self,
        q: Tensor,
        y: Tensor,
        sigma: Tensor,
    ) -> Tensor:
        sigma = sigma[(...,) + (None,) * (y.dim() - sigma.dim())]
        y = self._prepare_clipped_measurements(y)
        normal = torch.distributions.Normal(q, sigma)

        cdf_y = normal.cdf(y)
        cdf_left = normal.cdf(torch.full_like(y, self.boundary_eps))
        cdf_right = normal.cdf(torch.full_like(y, 1.0 - self.boundary_eps))

        s = torch.empty_like(y)
        left_mask = y < self.boundary_eps
        right_mask = y > 1.0 - self.boundary_eps
        center_mask = ~(left_mask | right_mask)

        s[left_mask] = (y[left_mask] / self.boundary_eps) * cdf_left[left_mask]
        s[right_mask] = cdf_right[right_mask] + (
            (y[right_mask] - (1.0 - self.boundary_eps)) / self.boundary_eps
        ) * (1.0 - cdf_right[right_mask])
        s[center_mask] = cdf_y[center_mask]
        s = torch.clamp(s, min=self.eps, max=1.0 - self.eps)

        z = (y - q) / sigma
        logit_pit = torch.empty_like(s)
        low_mask = s <= (self.eps + 1e-12)
        high_mask = s >= (1.0 - self.eps - 1e-12)
        mid_mask = ~(low_mask | high_mask)
        logit_pit[mid_mask] = torch.logit(s[mid_mask])

        log_norm = 0.5 * math.log(2.0 * math.pi)
        tail = 0.5 * z.square() + torch.log(z.abs().clamp_min(1e-12)) + log_norm
        logit_pit[low_mask] = -tail[low_mask]
        logit_pit[high_mask] = tail[high_mask]
        return logit_pit

    def _logit_pit(self, q: Tensor, y: Tensor, sigma: Tensor | None) -> Tensor:
        if self.distribution == "gaussian":
            return self._gaussian_logit_pit(q, y, sigma)
        if self.distribution == "poisson":
            return self._poisson_logit_pit(q, y)
        return self._clipped_gaussian_logit_pit(q, y, sigma)

    def forward(
        self,
        y: Tensor,
        x_net: Tensor,
        physics,
        model=None,
        sigma: float | Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        r"""
        Compute the distributional consistency loss.

        :param torch.Tensor y: noisy measurements.
        :param torch.Tensor x_net: reconstructed image.
        :param deepinv.physics.Physics physics: forward operator.
        :param torch.nn.Module model: unused, present for compatibility with
            :class:`deepinv.loss.Loss`.
        :param float, torch.Tensor sigma: optional Gaussian noise level overriding
            the loss initialization for the Gaussian-based variants.
        :return: torch.Tensor loss of size ``(batch_size,)``.
        """
        sigma = self._resolve_sigma(y=y, physics=physics, sigma=sigma)
        logit_pit = self._logit_pit(physics.A(x_net), y, sigma)

        batch_size = logit_pit.shape[0]
        logit_pit = logit_pit.reshape(batch_size, -1)
        sorted_logit_pit = torch.sort(logit_pit, dim=-1).values

        n_measurements = sorted_logit_pit.shape[-1]
        n_points = self.n_points if self.n_points is not None else n_measurements

        pit_grid = self._pit_grid(
            n_points=n_points,
            device=sorted_logit_pit.device,
            dtype=sorted_logit_pit.dtype,
        )
        positions = pit_grid * max(n_measurements - 1, 0)
        lower = positions.floor().long()
        upper = positions.ceil().long()
        weight = (positions - lower).unsqueeze(0)

        lower = lower.unsqueeze(0).expand(batch_size, -1)
        upper = upper.unsqueeze(0).expand(batch_size, -1)

        quantiles_low = torch.gather(sorted_logit_pit, -1, lower)
        quantiles_high = torch.gather(sorted_logit_pit, -1, upper)
        empirical_quantiles = quantiles_low + weight * (quantiles_high - quantiles_low)

        reference_quantiles = torch.logit(pit_grid).unsqueeze(0)
        return (empirical_quantiles - reference_quantiles).abs().mean(dim=-1)
