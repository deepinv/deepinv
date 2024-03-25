import torch


class ThinPlateSpline:
    r"""Solve the Thin Plate Spline interpolation

    Given a set of control points X_c \\ in R^{n_c \\times d_s} and target points X_t \\in R^{n_c \\times d_t}
    it learns a transformation f that maps X_c on X_t with some regularization.

    More formally:
    f = min_f E_{ext}(f) + \\alpha E_{int}(f)      (1)

    with E_{ext}(f) = \\sum_{i=1}^n ||X_{ti} - f(X_{ci})||_2^2
    and E_{int}(f) = \\iint \\left[\\left({\\frac{\\partial^2 f}{\\partial x_1^2}}\\right)^2
                                + 2\\left({\\frac{\\partial^2 f}{\\partial x_1\\partial x_2}}\\right)^2
                                +  \\left({\\frac{\\partial^2 f}{\\partial x_2^2}}\\right)^2 \\right]{dx_1\\,dx_2


    Let X \\in R^{n \\times d_s} be n point from the source space. Then \\Phi(X) is the radial distance of those point
    to the control points \\in R^{n \\times n_c}:
    with d_{ij} = ||X_i - X_{cj}||_2, \\Phi(X)_{ij} = d_{ij}^2 \\log d_ij

    Then f(X) = A + X.B + \\Phi(X).C
    with A \\ in R^{d_t}, B \\in R^{d_s \\times d_t}, C \\ in R^{n_c \\times d_t} the parameters to learn.

    Learning A, B, C is done by solving a linear system so that f minimizes the energy (1) to transform X_c in X_t.

    The equation to solve is:

           A      .   P   =   Y
                         <=>
    |  K   , X'_c|  | C |   |X_t|
    |            |  |   | = |   |
    |X'_c^T,   0 |  | B'|   | 0 |

    with X'_c = |1_{n_c}, X_c|  \\in R^{n_c \\times 1+d_s}, B'.T = |A, B.T|  \\in R^{d_t \\times 1+d_s}
    and K = \\Phi(X_c) + \\alpha I_{n_c}

    A \\in R^{(n_c + d_s + 1)\\times(n_c + d_s + 1)},
    P \\in R^{(n_c + d_s + 1)\\times d_t},
    Y \\in R^{(n_c + d_s + 1)\\times d_t},

    Attrs:
        alpha (float): Regularization parameter
        parameters (Tensor): All the parameters (P). Shape: (n_c + d_s + 1, d_t)
        control_points (Tensor): Control points fitted (X_c). Shape: (n_c, d_s)
    """

    def __init__(self, alpha=0.0, device="cpu") -> None:
        self._fitted = False
        self.alpha = alpha
        self.device = torch.device(device)

        self.parameters = torch.tensor([], dtype=torch.float32)
        self.control_points = torch.tensor([], dtype=torch.float32)

    def fit(self, X: torch.Tensor, Y: torch.Tensor) :
        """Learn f that matches Y given X

        Args:
            X (Tensor): Control point at source space (X_c)
                Shape: (n_c, d_s)
            Y (Tensor): Control point in the target space (X_t)
                Shape: (n_c, d_t)

        Returns:
            ThinPlateSpline: self
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        X = _ensure_2d(X)
        Y = _ensure_2d(Y)
        assert X.shape[0] == Y.shape[0]

        n_c, d_s = X.shape
        self.control_points = X

        phi = self._radial_distance(X)

        # Build the linear system AP = Y
        X_p = torch.hstack([torch.ones((n_c, 1), device=self.device), X])

        A = torch.vstack(
            [
                torch.hstack([phi + self.alpha * torch.eye(n_c, device=self.device), X_p]),
                torch.hstack([X_p.T, torch.zeros((d_s + 1, d_s + 1), device=self.device)]),
            ]
        )

        Y = torch.vstack([Y, torch.zeros((d_s + 1, Y.shape[1]), device=self.device)])

        self.parameters = torch.linalg.solve(A, Y)  # pylint: disable=not-callable
        self._fitted = True

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Map source space to target space

        Args:
            X (Tensor): Points in the source space
                Shape: (n, d_s)

        Returns:
            Tensor: Mapped points in the target space
                Shape: (n, d_t)
        """
        assert self._fitted, "Please call fit first."

        X = X.to(self.device)
        X = _ensure_2d(X)
        assert X.shape[1] == self.control_points.shape[1]

        phi = self._radial_distance(X)  # n x n_c

        X = torch.hstack([phi, torch.ones((X.shape[0], 1), device=self.device), X])  # n x (n_c + 1 + d_s)
        return X @ self.parameters

    def _radial_distance(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the pairwise radial distances of the given points to the control points

        Input dimensions are not checked.

        Args:
            X (Tensor): N points in the source space
                Shape: (n, d_s)

        Returns:
            Tensor: The radial distance for each point to a control point (\\Phi(X))
                Shape: (n, n_c)
        """
        # Don't use mm for euclid dist, lots of imprecision comes from it (Will be a bit slower)
        dist = torch.cdist(X, self.control_points, compute_mode="donot_use_mm_for_euclid_dist")
        dist[dist == 0] = 1  # phi(r) = r^2 log(r) ->  (phi(0) = 0)
        return dist**2 * torch.log(dist)


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure that tensor is a 2d tensor

    In case of 1d tensor, let's expand the last dim
    """
    assert tensor.ndim in (1, 2)

    # Expand last dim in order to interpret this as (n, 1) points
    if tensor.ndim == 1:
        tensor = tensor[:, None]

    return tensor