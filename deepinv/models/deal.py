import torch
from .base import Reconstructor
from . import deal_impl as deal_lib
from typing import Optional


class DEAL(Reconstructor):
    """
    DeepInverse-compatible wrapper around the official DEAL solver.

    Parameters
    ----------
    checkpoint_path : str
        Path to a pretrained DEAL checkpoint file (e.g. ``deal_gray.pth``).
    sigma : float, optional
        Noise level parameter expected by the DEAL model (default: 25.0).
    lam : float, optional
        Regularisation strength (lambda) used by the DEAL solver (default: 10.0).
    max_iter : int, optional
        Maximum number of outer iterations in the inverse solver (default: 50).
    auto_scale : bool, optional
        If True, rescale the measurements ``y`` so that their standard
        deviation is close to ``target_y_std`` (default: False).
    target_y_std : float, optional
        Target standard deviation used for automatic scaling (default: 25.0).
    color : bool, optional
        If True, use the color version of DEAL (three channels). If False,
        use the grayscale version (one channel). This must match the
        checkpoint and the data (default: False).
    device : {"cuda", "cpu"} or None, optional
        Device used for computations. If None, "cuda" is used when available,
        otherwise "cpu".
    clamp_output : bool, optional
        If True, clamp the reconstructed image to the range [0, 1]
        before returning (default: True).
    """

    def __init__(
        self,
        checkpoint_path: str,
        sigma: float = 25.0,
        lam: float = 10.0,
        max_iter: int = 50,
        auto_scale: bool = False,
        target_y_std: float = 25.0,
        color: bool = False,
        device: Optional[str] = None,
        clamp_output: bool = True,
    ) -> None:

        super().__init__()

        # Device selection
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Solver parameters
        self.sigma = float(sigma)
        self.lam = float(lam)
        self.max_iter = int(max_iter)
        self.auto_scale = bool(auto_scale)
        self.target_y_std = float(target_y_std)
        self.clamp_output = bool(clamp_output)

        # Underlying DEAL model from the official package
        self.model = deal_lib.DEAL(color=color).to(self.device).eval()

        # Load checkpoint (support both newer and older torch.load signatures)
        try:
            state = torch.load(
                checkpoint_path, map_location=self.device, weights_only=True
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(state["state_dict"])

    @torch.no_grad()
    def forward(self, y, physics):
        """
        Run the DEAL reconstruction.

        Parameters
        ----------
        y : torch.Tensor
            Measurements (e.g. sinogram).
        physics : object
            DeepInverse physics operator with ``__call__`` and ``A_adjoint``.

        Returns
        -------
        torch.Tensor
            Reconstructed image with the same spatial shape as ``H^T y``.
        """
        # Move data to the correct device
        y = y.to(self.device)

        # Forward and adjoint operators as callables
        H = lambda z: physics(z)
        Ht = physics.A_adjoint

        # Optional automatic scaling of y
        if self.auto_scale:
            y_std = float(y.std().detach().cpu())
            if 0.0 < y_std < 5.0:
                scale = self.target_y_std / (y_std + 1e-12)
                y = y * scale

        # Zero initialisation
        x_init = torch.zeros_like(Ht(y))

        # Set number of outer iterations on the underlying DEAL model
        if hasattr(self.model, "max_iter"):
            self.model.max_iter = max(int(self.max_iter), 1)

        # Call the official DEAL solver
        x_hat = self.model.solve_inverse_problem(
            y,
            H=H,
            Ht=Ht,
            sigma=self.sigma,
            lmbda=self.lam,
            x_init=x_init,
            verbose=False,
            path=False,
        )

        return x_hat.clamp(0.0, 1.0) if self.clamp_output else x_hat
