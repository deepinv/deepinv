from __future__ import annotations

"""
DEAL-based reconstructor for DeepInverse.

This module wraps the official DEAL implementation

    Pourya et al., "DEALing with Image Reconstruction:
    Deep Attentive Least Squares", arXiv:2502.04079.

The DEAL code and pretrained weights are provided by the authors at:
    https://mehrsapo.github.io/DEAL

This wrapper assumes that the user has installed the DEAL package
and downloaded a compatible checkpoint file (e.g. ``deal_gray.pth``
for single-channel images or ``deal_color.pth`` for RGB images).
"""

from typing import Optional, Any

import torch
import deal




class DEALReconstructor:
    """
    Wrapper around the official DEAL solver.

    Parameters
    ----------
    checkpoint_path : str
        Path to a pretrained DEAL checkpoint file, for example
        ``deal_gray.pth`` (single-channel) or ``deal_color.pth`` (RGB).
    color : bool, default False
        If ``True``, instantiate the color version of DEAL (three channels).
        If ``False``, instantiate the grayscale version (one channel).
        This must match the checkpoint and the image / sinogram shape.
    sigma : float, default 25.0
        Noise level parameter expected by the DEAL model.
    lam : float, default 10.0
        Regularisation strength (λ) used by the DEAL solver.
    max_iters : int, default 50
        Maximum number of outer iterations in the inverse solver.
    device : {"cuda", "cpu"} or None, default None
        Device used for computations. If ``None``, "cuda" is used when
        available, otherwise "cpu".
    clamp_output : bool, default True
        If ``True``, the reconstruction is clamped to the range [0, 1]
        before it is returned.
    auto_scale : bool, default False
        If ``True``, the measurements ``y`` are rescaled so that their
        standard deviation is close to ``target_y_std``. This is useful
        when the forward model produces measurements on a very different
        scale from the data used to train DEAL. By default it is disabled
        to avoid surprising behaviour.
    target_y_std : float, default 25.0
        Target standard deviation for ``y`` when ``auto_scale`` is enabled.
    scale_print : bool, default False
        If ``True``, print the scale factor that is applied to ``y`` when
        ``auto_scale`` is enabled.

    Notes
    -----
    * The ``physics`` object passed to :meth:`reconstruct` must implement
      a call operator ``physics(x)`` (forward model :math:`Hx`) and an
      adjoint method ``physics.A_adjoint(y)`` (:math:`H^T y`), as in
      :class:`deepinv.physics.Tomography`.
    * The small tolerances ``eps_in`` and ``eps_out`` are set here to
      values that worked well in our experiments, but they can be changed
      in the future if needed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        color: bool = False,
        sigma: float = 25.0,
        lam: float = 10.0,
        max_iters: int = 50,
        device: Optional[str] = None,
        clamp_output: bool = True,
        auto_scale: bool = False,
        target_y_std: float = 25.0,
        scale_print: bool = False,
    ) -> None:
        # Device selection
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Instantiate DEAL model (grayscale or color) and load weights.
        self.model = deal.DEAL(color=color).to(self.device).eval()

        # Older checkpoints might not support weights_only=True.
        try:
            state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state["state_dict"])

        # Solver parameters
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.max_iters = int(max_iters)
        self.clamp_output = bool(clamp_output)

        # Tolerances used inside the DEAL solver
        self.eps_in = 5e-3
        self.eps_out = 5e-3

        # A few internal iteration limits which are exposed by the DEAL class.
        for name, val in [
            ("inner_iter", 1),
            ("cg_max_iter", 5),
            ("max_cg_iter", 5),
            ("lbfgs_max_iter", 0),
        ]:
            if hasattr(self.model, name):
                try:
                    setattr(self.model, name, int(val))
                except Exception:
                    # Fail silently if a particular attribute is not present
                    pass

        # Optional measurement scaling
        self.auto_scale = bool(auto_scale)
        self.target_y_std = float(target_y_std)
        self.scale_print = bool(scale_print)

    @torch.no_grad()
    def reconstruct(
        self,
        y: torch.Tensor,
        physics: Any,
        x_init: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reconstruct an image :math:`x` from measurements :math:`y`.

        Parameters
        ----------
        y : torch.Tensor
            Measurements (e.g. sinogram) with shape
            ``(N, C, det_bins, angles)``.
        physics : object
            Forward operator with signature ``physics(x) -> Hx`` and an
            adjoint method ``physics.A_adjoint(y) -> H^T y``.
        x_init : torch.Tensor, optional
            Optional initial guess for the reconstruction. If ``None``,
            a zero tensor with the shape of ``H^T y`` is used.

        Returns
        -------
        torch.Tensor
            Reconstructed image with the same shape as ``x_init`` (or
            ``H^T y`` when ``x_init`` is not provided). If
            ``clamp_output=True`` the result is clamped to [0, 1].
        """
        y = y.to(self.device)

        # Define H and H^T as simple callables expected by DEAL.
        H = lambda z: physics(z)
        Ht = physics.A_adjoint

        # Optional measurement rescaling.
        if self.auto_scale:
            y_std = float(y.std().detach().cpu())
            if 0.0 < y_std < 5.0:
                scale = self.target_y_std / (y_std + 1e-12)
                y = y * scale
                if self.scale_print:
                    print(
                        f"[DEALReconstructor] auto_scale applied: "
                        f"y.std={y_std:.3f} -> scale={scale:.1f} "
                        f"(target {self.target_y_std})"
                    )

        # Zero initialisation if no starting point is provided.
        if x_init is None:
            x_init = torch.zeros_like(Ht(y))

        # Number of outer iterations for the DEAL solver.
        self.model.max_iter = max(self.max_iters, 1)

        x_hat = self.model.solve_inverse_problem(
            y,
            H=H,
            Ht=Ht,
            sigma=self.sigma,
            lmbda=self.lam,
            eps_in=self.eps_in,
            eps_out=self.eps_out,
            path=False,
            x_init=x_init,
            verbose=False,
        )

        return x_hat.clamp(0.0, 1.0) if self.clamp_output else x_hat


@torch.no_grad()
def deal_reconstruct(
    y: torch.Tensor,
    physics: Any,
    checkpoint_path: str,
    *,
    color: bool = False,
    sigma: float = 25.0,
    lam: float = 10.0,
    max_iters: int = 50,
    device: Optional[str] = None,
    clamp_output: bool = True,
    auto_scale: bool = False,
    target_y_std: float = 25.0,
    scale_print: bool = False,
) -> torch.Tensor:
    """
    Convenience function for running DEAL in one line.

    Example
    -------
    >>> x_hat = deal_reconstruct(y, H, checkpoint_path)

    All keyword arguments are forwarded to :class:`DEALReconstructor`.
    """
    reconstructor = DEALReconstructor(
        checkpoint_path=checkpoint_path,
        color=color,
        sigma=sigma,
        lam=lam,
        max_iters=max_iters,
        device=device,
        clamp_output=clamp_output,
        auto_scale=auto_scale,
        target_y_std=target_y_std,
        scale_print=scale_print,
    )
    return reconstructor.reconstruct(y, physics)
