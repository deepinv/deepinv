import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from deepinv.optim import Prior
import torchcde


class LinearSpline(nn.Module):
    """
    Learn N functions alpha_i(σ) = exp( s_i(σ) ) / (σ + eps),
    where s_i is a natural cubic spline over σ.

    Args
    ----
    N : int
        Number of alphas.
    K : int
        Number of spline knots (>=2).
    sigma_min, sigma_max : float
        Range of σ where knots are placed.
    eps : float
        Small constant in denominator for stability (e.g., 1e-5).
    init : float
        Initial value for s_i at all knots.
    """

    def __init__(
        self,
        N: int = 32,
        K: int = 12,
        sigma_min: float = 0.01,
        sigma_max: float = 0.1,
        eps: float = 1e-5,
        init: float = 0.0,
    ):
        super().__init__()
        self.N = N
        self.K = K
        self.eps = float(eps)
        # --- fixed knot locations (registered buffers, no grads) ---
        t = torch.linspace(sigma_min, sigma_max, K)
        self.register_buffer("t_knots", t)  # shape [K]
        # --- learnable spline values at knots: s_i(t_k) ---
        # Shape [1, K, N]: batch=1, time=K, channels=N
        self.s_at_knots = nn.Parameter(torch.full((1, K, N), float(init)))

    def _build_spline(self):
        """
        Build a LinearSpline object from current knot values.
        torchcde expects data of shape [B, T, C] with strictly increasing T.
        """
        # coefficients are computed with gradients flowing to s_at_knots
        coeffs = torchcde.linear_interpolation_coeffs(self.s_at_knots, t=self.t_knots)
        return torchcde.LinearInterpolation(coeffs)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all N alphas at a batch of noise levels.

        Parameters
        ----------
        sigma : Tensor, shape [B] or [B,1]
            Per-sample noise levels (must be > 0).

        Returns
        -------
        alphas : Tensor, shape [B, N]
        """
        if type(sigma) is not torch.Tensor and type(sigma) is not float:
            raise TypeError(
                f"Expected sigma to be a 1D torch.Tensor or float, got {type(sigma)}"
            )
        elif type(sigma) is float:
            sigma = torch.tensor(
                [sigma], dtype=torch.float32, device=self.s_at_knots.device
            )
        sigma = sigma.view(-1)  # [B]
        spline = self._build_spline()  # linear spline s(t)
        s_vals = spline.evaluate(sigma)  # shape [1, B, N]
        s_vals = s_vals.squeeze(0)  # -> [B, N]
        alphas = torch.exp(s_vals) / (sigma.view(-1, 1) + self.eps)
        return alphas.view(len(sigma), self.N, 1, 1, 1)


class ZeroMean3D(nn.Module):
    """Enforcing zero mean on 3D filters improves performance"""

    def forward(self, x):
        return x - torch.mean(x, dim=(1, 2, 3, 4), keepdim=True)
