import torch
from deepinv.physics import Physics
from math import comb


class SpikePhysics(Physics):
    """
    Forward operator for a collection of blurred spikes on [0,1]^2.

    Parameters
    ----------
    Nx : int
        Number of pixels along the x-axis.
    Ny : int
        Number of pixels along the y-axis.
    sigma_psf : float
        Standard deviation of the Gaussian PSF.

    Input
    -----
    x : torch.Tensor or list[torch.Tensor]
        Either spike positions of shape (K, 2), in which case all amplitudes
        are set to one, or a list [positions, amplitudes] with amplitudes
        of shape (K,).

    Output
    ------
    torch.Tensor
        Blurred image of shape (1, 1, Ny, Nx).
    """

    def __init__(
        self,
        Nx: int,
        Ny: int,
        sigma_psf: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.Nx = Nx
        self.Ny = Ny
        self.sigma_psf = sigma_psf

    

    def A(self, x, **kwargs) -> torch.Tensor:

        # --------------------------------------------------
        # Positions and amplitudes
        # --------------------------------------------------
        if isinstance(x, torch.Tensor):
            positions = x

            amplitudes = torch.ones(
                positions.shape[0],
                device=positions.device,
                dtype=positions.dtype,
            )
        else:
            positions, amplitudes = x

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "positions must have shape (K, 2)."
            )

        if amplitudes.ndim != 1:
            raise ValueError(
                "amplitudes must have shape (K,)."
            )

        if amplitudes.shape[0] != positions.shape[0]:
            raise ValueError(
                "positions and amplitudes must contain "
                "the same number of spikes."
            )

        amplitudes = amplitudes.to(
            device=positions.device,
            dtype=positions.dtype,
        )

        device = positions.device
        dtype = positions.dtype

        # --------------------------------------------------
        # Pixel centers
        # --------------------------------------------------
        xc = (
            torch.arange(self.Nx, device=device, dtype=dtype)
            + 0.5
        ) / self.Nx

        yc = (
            torch.arange(self.Ny, device=device, dtype=dtype)
            + 0.5
        ) / self.Ny

        Y, X = torch.meshgrid(
            yc,
            xc,
            indexing="ij",
        )

        # --------------------------------------------------
        # Fixed normalization
        # --------------------------------------------------
        x_ref = 0.5 * (X.min() + X.max())
        y_ref = 0.5 * (Y.min() + Y.max())

        kernel_ref = torch.exp(
            -(
                (X - x_ref) ** 2
                + (Y - y_ref) ** 2
            )
            / (2 * self.sigma_psf**2)
        )

        C_sigma = kernel_ref.sum()

        # --------------------------------------------------
        # Blurred spikes
        # --------------------------------------------------
        dx = X[None, :, :] - positions[:, 0, None, None]
        dy = Y[None, :, :] - positions[:, 1, None, None]

        kernels = torch.exp(
            -(dx**2 + dy**2)
            / (2 * self.sigma_psf**2)
        ) / C_sigma

        # --------------------------------------------------
        # Weighted sum
        # --------------------------------------------------
        image = torch.sum(
            amplitudes[:, None, None] * kernels,
            dim=0,
        )
        return image[None, None, :, :]




# ============================================================
# Bézier curve Class
# ============================================================

class BezierCurve:
    """
    Represent a Bézier curve in R^d defined by its control points.

    Parameters
    ----------
    control_points : torch.Tensor
        Control points of shape ``(n+1, d)``, where ``n`` is the
        degree of the curve.
    dev : torch.device or str, optional
        Device used for tensor computations.
    """

    def __init__(self, control_points, dev=None):

        if not isinstance(control_points, torch.Tensor):
            raise TypeError("control_points must be a torch.Tensor.")

        if control_points.ndim != 2:
            raise ValueError(
                "control_points must have shape (n+1, d)."
            )

        if dev is None:
            dev = control_points.device

        self.dev = torch.device(dev)
        self.control_points = control_points.to(self.dev)

        self.n = self.control_points.shape[0] - 1
        self.d = self.control_points.shape[1]


    
    @staticmethod
    def bernstein_poly(i, n, t):
        """Evaluate the Bernstein basis polynomial B_i^n at t."""
        return comb(n, i) * t**i * (1.0 - t)**(n - i)


    def _prepare_t(self, t):
        """Convert parameter values to the curve dtype and device."""
        return torch.as_tensor(
            t,
            dtype=self.control_points.dtype,
            device=self.control_points.device,
        ).reshape(-1)


    def evaluate(self, t):
        """
        Evaluate the Bézier curve at parameter values t.

        Returns
        -------
        torch.Tensor
            Curve points of shape ``(len(t), d)``.
        """
        t = self._prepare_t(t)

        B = torch.stack(
            [
                self.bernstein_poly(i, self.n, t)
                for i in range(self.n + 1)
            ],
            dim=1,
        )

        return B @ self.control_points

    
    def derivative(self, t):

        """Evaluate the first derivative of the Bézier curve."""

        t = self._prepare_t(t)

        if self.n == 0:
            return torch.zeros(
                (len(t), self.d),
                dtype=self.control_points.dtype,
                device=self.dev,
            )

        derivative_ctrl = self.n * (
            self.control_points[1:]
            - self.control_points[:-1]
        )

        B = torch.stack(
            [
                self.bernstein_poly(i, self.n - 1, t)
                for i in range(self.n)
            ],
            dim=1,
        )

        return B @ derivative_ctrl

    

    def second_derivative(self, t):

        """Evaluate the second derivative of the Bézier curve."""

        t = self._prepare_t(t)

        if self.n < 2:
            return torch.zeros(
                (len(t), self.d),
                dtype=self.control_points.dtype,
                device=self.dev,
            )
        second_ctrl = self.n * (self.n - 1) * (
            self.control_points[2:]
            - 2.0 * self.control_points[1:-1]
            + self.control_points[:-2]
        )
        B = torch.stack(
            [
                self.bernstein_poly(i, self.n - 2, t)
                for i in range(self.n - 1)
            ],
            dim=1,
        )
        return B @ second_ctrl


    
    def length(self, N_points=200):

        """Approximate the curve length using the trapezoidal rule."""

        if N_points < 2:
            raise ValueError("N_points must be at least 2.")

        t_vals = torch.linspace(
            0.0,
            1.0,
            N_points,
            dtype=self.control_points.dtype,
            device=self.dev,
        )

        speed = torch.linalg.vector_norm(
            self.derivative(t_vals),
            dim=1,
        )

        return torch.trapezoid(speed, t_vals)

    

# ============================================================
# physics
# ============================================================

class BezierCurvePhysics(Physics):

    def __init__(
        self,
        Nx: int,
        Ny: int,
        sigma_psf: float,
        N_points: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.Nx = Nx
        self.Ny = Ny
        self.sigma_psf = sigma_psf
        self.N_points = N_points


    def A(self, x: list[torch.Tensor], **kwargs) -> torch.Tensor:

        if len(x) == 0:
            raise ValueError("x must contain at least one curve.")

        device = x[0].device
        dtype = x[0].dtype

        # --------------------------------------------------
        # Pixel centers on [0,1]^2
        # --------------------------------------------------
        xc = (torch.arange(self.Nx, device=device, dtype=dtype) + 0.5) / self.Nx
        yc = (torch.arange(self.Ny, device=device, dtype=dtype) + 0.5) / self.Ny

        Y, X = torch.meshgrid(yc, xc, indexing="ij")  # (Ny, Nx)

        # --------------------------------------------------
        # Fixed discrete normalization of the Gaussian PSF
        # --------------------------------------------------
        x_ref = 0.5 * (X.min() + X.max())
        y_ref = 0.5 * (Y.min() + Y.max())

        kernel_ref = torch.exp(
            -(
                (X - x_ref) ** 2
                + (Y - y_ref) ** 2
            )
            / (2 * self.sigma_psf**2)
        )

        C_sigma = kernel_ref.sum()

        # --------------------------------------------------
        # Curve discretization
        # --------------------------------------------------
        t = torch.linspace(
            0.0,
            1.0,
            self.N_points,
            device=device,
            dtype=dtype,
        )

        image = torch.zeros(
            (self.Ny, self.Nx),
            device=device,
            dtype=dtype,
        )

        # --------------------------------------------------
        # Contribution of each Bézier curve
        # --------------------------------------------------
        for control_points in x:

            curve = BezierCurve(control_points)

            gamma = curve.evaluate(t)
            gamma_dot = curve.derivative(t)

            speed = torch.linalg.vector_norm(gamma_dot, dim=1)

            dx = X[None, :, :] - gamma[:, 0, None, None]
            dy = Y[None, :, :] - gamma[:, 1, None, None]

            kernel = torch.exp(
                -(dx**2 + dy**2)
                / (2 * self.sigma_psf**2)
            ) / C_sigma

            image += torch.trapezoid(
                kernel * speed[:, None, None],
                t,
                dim=0,
            )

        return image[None, None, :, :]








