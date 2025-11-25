import torch
from .base import Denoiser


class TVDenoiser(Denoiser):
    r"""
    Proximal operator of the isotropic Total Variation operator.

    This algorithm converges to the unique image :math:`x` that is the solution of

    .. math::

        \underset{x}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \gamma \|Dx\|_{1,2},

    where :math:`D` maps an image to its gradient field.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm, see :footcite:t:`condat2013primal`.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    This algorithm is implemented with warm restart, i.e. the primary and dual variables are kept in memory
    between calls to the forward method. This speeds up the computation when using this class in an iterative algorithm.

    :param bool verbose: Whether to print computation details or not. Default: False.
    :param float tau: Stepsize for the primal update. Default: 0.01.
    :param float rho: Over-relaxation parameter. Default: 1.99.
    :param int n_it_max: Maximum number of iterations. Default: 1000.
    :param float crit: Convergence criterion. Default: 1e-5.
    :param torch.Tensor, None x2: Primary variable for warm restart. Default: None.
    :param torch.Tensor, None u2: Dual variable for warm restart. Default: None.

    .. note::
        The regularization term :math:`\|Dx\|_{1,2}` is implicitly normalized by its Lipschitz constant, i.e.
        :math:`\sqrt{8}`, see e.g. A. Beck and M. Teboulle, "Fast gradient-based algorithms for constrained total
        variation image denoising and deblurring problems", IEEE T. on Image Processing. 18(11), 2419-2434, 2009.

    .. warning::
        For using TV as a prior for Plug and Play algorithms, it is recommended to use the class
        :class:`~deepinv.optim.prior.TVPrior` instead. In particular, it allows to evaluate TV.
    """

    def __init__(
        self,
        verbose=False,
        tau=0.01,
        rho=1.99,
        n_it_max=1000,
        crit=1e-5,
        x2=None,
        u2=None,
    ):
        super(TVDenoiser, self).__init__()

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = tau
        self.rho = rho

        self.x2 = x2
        self.u2 = u2

        self.has_converged = False

    def prox_tau_fx(self, x, y):
        r"""
        Proximal operator of the function :math:`\frac{1}{2}\|x-y\|_2^2`.
        """
        return (x + self.tau * y) / (1 + self.tau)

    def prox_sigma_g_conj(self, u, lambda2):
        return u / (
            torch.clamp(
                torch.linalg.vector_norm(u, dim=-1, ord=2, keepdim=True) / lambda2,
                min=1.0,
            )
        )

    def forward(self, y, ths=None, **kwargs):
        r"""
        Computes the proximity operator of the TV norm.

        :param torch.Tensor y: Noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
        :param float, torch.Tensor ths: Regularization parameter :math:`\gamma`.
        :return: Denoised image.
        """

        restart = (
            True
            if (
                self.restart
                or self.x2 is None
                or self.u2 is None
                or self.x2.shape != y.shape
            )
            else False
        )

        if restart:
            x2 = y.clone()
            u2 = torch.zeros((*y.shape, y.ndim - 2), device=y.device).type(y.dtype)
            self.restart = False
        else:
            x2 = self.x2.clone()
            u2 = self.u2.clone()

        sigma = (
            1 / self.tau / 2 ** (y.ndim - 1)
        )  # square of the Lipschitz constant of nabla

        if ths is not None:
            lambd = self._handle_sigma(
                ths,
                batch_size=y.size(0),
                ndim=y.ndim + 1,
                device=y.device,
                dtype=y.dtype,
            )

        for _ in range(self.n_it_max):
            x_prev = x2

            x = self.prox_tau_fx(x2 - self.tau * self.nabla_adjoint(u2), y)
            u = self.prox_sigma_g_conj(u2 + sigma * self.nabla(2 * x - x2), lambd)
            x2 = x2 + self.rho * (x - x2)
            u2 = u2 + self.rho * (u - u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - x2.flatten()
            ) / torch.linalg.norm(x2.flatten() + 1e-12)

            if _ > 1 and rel_err < self.crit:
                if self.verbose:
                    print("TV prox reached convergence")
                break

        self.x2 = x2.detach()
        self.u2 = u2.detach()

        return x2

    @staticmethod
    def nabla(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x, in either 2D or 3D.
        """
        if x.ndim not in [4, 5]:
            raise ValueError(f"Input tensor must be 4D or 5D, got {x.ndim}D")

        n_spatial = (
            x.ndim - 2
        )  # Number of spatial dimensions (2 for 4D tensor, 3 for 5D tensor)
        u = torch.zeros(
            (*x.shape, n_spatial), device=x.device, dtype=x.dtype
        )  # Create output tensor with extra dimension for gradients

        # Compute gradient along each spatial dimension
        for i in range(n_spatial):
            # Dimension index in the original tensor (skip batch and channel dims)
            dim = i + 2
            # Create slice objects for indexing
            slice_from = [slice(None)] * x.ndim
            slice_to = [slice(None)] * x.ndim
            slice_from[dim] = slice(None, -1)
            slice_to[dim] = slice(1, None)

            # Compute finite difference and assign to output
            u[(*slice_from, i)] = x[tuple(slice_to)] - x[tuple(slice_from)]

        return u

    @staticmethod
    def nabla_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        if x.ndim not in [5, 6]:
            raise ValueError(f"Input tensor must be 5D or 6D, got {x.ndim}D")

        n_spatial = (
            x.ndim - 3
        )  # Number of spatial dimensions (2 for 5D tensor, 3 for 6D tensor)
        u = torch.zeros(
            x.shape[:-1], device=x.device, dtype=x.dtype
        )  # Output without gradient dimension

        # Apply adjoint along each spatial dimension
        for i in range(n_spatial):
            dim = (
                i + 2
            )  # Dimension index in the output tensor (skip batch and channel dims)

            # Create slice objects for indexing
            slice_from = [slice(None)] * u.ndim
            slice_to = [slice(None)] * u.ndim
            slice_from[dim] = slice(None, -1)
            slice_to[dim] = slice(1, None)

            slice_grad = [slice(None)] * x.ndim
            slice_grad[-1] = i
            slice_grad[dim] = slice(None, -1)

            # Apply adjoint operator (reversed finite difference)
            u[tuple(slice_from)] -= x[tuple(slice_grad)]
            u[tuple(slice_to)] += x[tuple(slice_grad)]

        return u
