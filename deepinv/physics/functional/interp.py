import torch


class ThinPlateSpline:
    r"""Solve the Thin Plate Spline interpolation problem

    Given a set of control points :math:`X_c` in :math:`\mathbb{R}^{n_c \times d_s}` and target points
    :math:`X_t` in :math:`\mathbb{R}^{n_c \times d_t}`,
    it learns a transformation :math:`f` that maps :math:`X_c` to :math:`X_t` with some regularization.

    The mapping is defined by:

    .. math:: 
        :label: formula

        f = \min_f E_{\text{ext}}(f) + \alpha E_{\text{int}}(f)


    with

    .. math::

        E_{\text{ext}}(f) = \frac{1}{2}\sum_{i=1}^n \|X_{t_i} - f(X_{c_i})\|_2^2

    .. math::

        E_{\text{int}}(f) = \iint \left[\left({\frac{\partial^2 f}{\partial x_1^2}}\right)^2
                                + 2\left({\frac{\partial^2 f}{\partial x_1\partial x_2}}\right)^2
                                +  \left({\frac{\partial^2 f}{\partial x_2^2}}\right)^2 \right]{dx_1\,dx_2}

    Let :math:`X \in \mathbb{R}^{n \times d_s}` be :math:`n` point from the source space. Then :math:`\Phi(X)`
    is the radial distance of those points
    to the control points in :math:`\mathbb{R}^{n \times n_c}`:
    with :math:`d_{ij} = ||X_{t_i} - X_{c_j}||_2, \Phi(X)_{ij} = d_{ij}^2 \log d_{ij}`

    Then :math:`f(X) = A + X \cdot B + \Phi(X) \cdot C`
    with :math:`A \in \mathbb{R}^{d_t}`, :math:`B \in \mathbb{R}^{d_s \times d_t}`,
    :math:`C \in \mathbb{R}^{n_c \times d_t}` the parameters to learn.

    Learning :math:`A`, :math:`B`, :math:`C` is done by solving a linear system so that :math:`f` minimizes
    the energy :eq:`formula` to transform :math:`X_c` in :math:`X_t`.

    The equation to solve is:

    .. math::

        \begin{equation*}
            A      \cdot   P =   Y
        \end{equation*}

    .. math::

        \begin{align*}
            \begin{bmatrix}
                K   & X'_c \\
                X_{c}^{'T} &   0
            \end{bmatrix} 
            \begin{bmatrix}
                C \\
                B'
            \end{bmatrix}   
            = 
            \begin{bmatrix}
                X_t \\
                0
            \end{bmatrix}
        \end{align*}

    with :math:`X'_c = [1_{n_c}, X_c]  \in \mathbb{R}^{n_c \times (1+d_s)}`, :math:`B'` = :math:`[A, B^{\top}]`
    in :math:`\mathbb{R}^{d_t \times (1+d_s)}`
    and :math:`K = \Phi(X_c) + \alpha I_{n_c}`

    :math:`A \in \mathbb{R}^{(n_c + d_s + 1)\times(n_c + d_s + 1)}`,
    :math:`P \in \mathbb{R}^{(n_c + d_s + 1)\times d_t}`,
    :math:`Y \in \mathbb{R}^{(n_c + d_s + 1)\times d_t}`,

    Attrs:
        alpha (float): Regularization parameter
        parameters (Tensor): All the parameters (P). Shape: :math:`(n_c + d_s + 1, d_t)`
        control_points (Tensor): Control points fitted (X_c). Shape: :math:`(n_c, d_s)`
    """

    def __init__(self, alpha=0.0, device="cpu", dtype=torch.float32) -> None:
        self._fitted = False
        self.alpha = alpha
        self.device = torch.device(device)
        self.dtype = dtype

        self.parameters = torch.tensor([], dtype=torch.float32)
        self.control_points = torch.tensor([], dtype=torch.float32)

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        r"""Learn f that matches Y given X

        Args:
            X (Tensor): Control point at source space (X_c)
                Shape: (n_c, d_s)
            Y (Tensor): Control point in the target space (X_t)
                Shape: (B, C, n_c, d_t)

        Returns:
            ThinPlateSpline: self
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        X = _ensure_batched_2d(X)
        Y = _ensure_batched_2d(Y)

        assert X.shape[2] == Y.shape[2]

        n_c, d_s = X.shape[-2:]
        self.control_points = X

        phi = self._radial_distance(X)

        # Build the linear system AP = Y
        one = torch.ones((1, 1, n_c, 1), device=self.device).expand(
            X.size(0), X.size(1), -1, -1
        )
        zeros = torch.zeros((1, 1, d_s + 1, d_s + 1), device=self.device).expand(
            X.size(0), X.size(1), -1, -1
        )

        X_p = torch.cat([one, X], dim=-1)

        A = torch.cat(
            [
                torch.cat(
                    [
                        phi
                        + self.alpha
                        * torch.eye(n_c, device=self.device)[None, None].expand(
                            X.size(0), X.size(1), -1, -1
                        ),
                        X_p,
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        X_p.transpose(-1, -2),
                        zeros,
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        Y = torch.cat(
            [
                Y,
                torch.zeros((1, 1, d_s + 1, Y.size(-1)), device=self.device).expand(
                    Y.size(0), Y.size(1), -1, -1
                ),
            ],
            dim=-2,
        )

        self.parameters = torch.linalg.solve(A, Y)  # pylint: disable=not-callable
        self._fitted = True

        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        r"""Map source space to target space

        Args:
            X (Tensor): Points in the source space
                Shape: (n, d_s)

        Returns:
            Tensor: Mapped points in the target space
                Shape: (n, d_t)
        """
        assert self._fitted, "Please call fit first."

        X = X.to(self.device)
        X = _ensure_batched_2d(X)
        assert X.shape[-1] == self.control_points.shape[-1]

        phi = self._radial_distance(X)

        one = torch.ones((1, 1, X.shape[-2], 1), device=self.device).expand(
            X.size(0), X.size(1), -1, -1
        )
        X = torch.cat([phi, one, X], dim=-1)
        return torch.matmul(X, self.parameters)

    def _radial_distance(self, X: torch.Tensor) -> torch.Tensor:
        r"""Compute the pairwise radial distances of the given points to the control points

        Input dimensions are not checked.

        Args:
            X (Tensor): N points in the source space
                Shape: (n, d_s)

        Returns:
            Tensor: The radial distance for each point to a control point (\\Phi(X))
                Shape: (n, n_c)
        """
        # Don't use mm for euclid dist, lots of imprecision comes from it (Will be a bit slower)
        dist = torch.cdist(
            X, self.control_points, compute_mode="donot_use_mm_for_euclid_dist"
        )
        dist[dist == 0] = 1  # phi(r) = r^2 log(r) ->  (phi(0) = 0)
        return dist**2 * torch.log(dist)


def _ensure_batched_2d(tensor: torch.Tensor) -> torch.Tensor:
    r"""Ensure that tensor is a 2d tensor

    In case of 1d tensor, let's expand the last dim
    """
    assert tensor.ndim in (2, 4)

    # Expand first 2 dim in order to interpret this as (B, C, ..., ...) points
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    return tensor
