import torch
from .base import Denoiser
from .tv import TVDenoiser


class TGVDenoiser(Denoiser):
    r"""
    Proximal operator of (2nd order) Total Generalized Variation operator.

    Adapted from :footcite:t:`bredies2010total`.

    This algorithm converges to the unique image :math:`x` (and the auxiliary vector field :math:`r`) minimizing

    .. math::

        \underset{x, r}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda_1 \|r\|_{1,2} + \lambda_2 \|J(Dx-r)\|_{1,F}

    where :math:`D` maps an image to its gradient field and :math:`J` maps a vector field to its Jacobian.
    For a large value of :math:`\lambda_2`, the TGV behaves like the TV.
    For a small value, it behaves like the :math:`\ell_1`-Frobenius norm of the Hessian.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm, see :footcite:t:`condat2013primal`.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.


    .. note::
        The regularization term :math:`\|r\|_{1,2} + \|J(Dx-r)\|_{1,F}` is implicitly normalized by its Lipschitz
        constant, i.e. :math:`\sqrt{72}`, see e.g. K. Bredies et al., "Total generalized variation," SIAM J. Imaging
        Sci., 3(3), 492-526, 2010.

    :param bool verbose: Whether to print computation details or not. Default: False.
    :param int n_it_max: Maximum number of iterations. Default: 1000.
    :param float crit: Convergence criterion. Default: 1e-5.
    :param torch.Tensor, None x2: Primary variable. Default: None.
    :param torch.Tensor, None u2: Dual variable. Default: None.
    :param torch.Tensor, None r2: Auxiliary variable. Default: None.


    """

    def __init__(
        self, verbose=False, n_it_max=1000, crit=1e-5, x2=None, u2=None, r2=None
    ):
        super(TGVDenoiser, self).__init__()

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = 0.01  # > 0

        self.rho = 1.99  # in 1,2

        self.x2 = x2
        self.r2 = r2
        self.u2 = u2

        self.has_converged = False

    def prox_tau_fx(self, x, y):
        return (x + self.tau * y) / (1 + self.tau)

    def prox_tau_fr(self, r, lambda1):
        left = torch.sqrt(torch.sum(r**2, axis=-1)) / (self.tau * lambda1)
        tmp = r - r / (
            torch.maximum(
                left, torch.tensor([1], device=left.device).type(left.dtype)
            ).unsqueeze(-1)
        )
        return tmp

    def prox_sigma_g_conj(self, u, lambda2):
        return u / (
            torch.maximum(
                torch.sqrt(torch.sum(u**2, axis=-1)) / lambda2,
                torch.tensor([1], device=u.device).type(u.dtype),
            ).unsqueeze(-1)
        )

    def forward(self, y, ths=None, **kwargs):
        r"""
        Computes the proximity operator of the TGV norm.

        :param torch.Tensor y: Noisy image. Assumes a tensor of shape (B, C, H, W) (2D data) or (B, C, D, H, W) (3D data).
        :param float, torch.Tensor ths: Regularization parameter.
        :return: Denoised image.
        """
        restart = (
            True
            if (self.restart or self.x2 is None or self.x2.shape != y.shape)
            else False
        )

        if restart:
            self.x2 = y.clone()
            self.r2 = torch.zeros(
                (*self.x2.shape, y.ndim - 2), device=self.x2.device, dtype=self.x2.dtype
            )
            self.u2 = torch.zeros(
                (*self.x2.shape, (y.ndim - 2) ** 2),
                device=self.x2.device,
                dtype=self.x2.dtype,
            )
            self.restart = False

        f = 3 if (y.ndim - 2) == 3 else 1
        sigma = 1 / self.tau / (72 * f)

        if ths is not None:
            lambda1 = (
                self._handle_sigma(
                    ths,
                    batch_size=y.size(0),
                    ndim=y.ndim,
                    device=y.device,
                    dtype=y.dtype,
                )
                * 0.1
            )
            lambda2 = (
                self._handle_sigma(
                    ths,
                    batch_size=y.size(0),
                    ndim=y.ndim,
                    device=y.device,
                    dtype=y.dtype,
                )
                * 0.15
            )

        cy = (y**2).sum() / 2
        primalcostlowerbound = 0

        for _ in range(self.n_it_max):
            x_prev = self.x2.clone()
            tmp = self.tau * self.epsilon_adjoint(self.u2)
            x = self.prox_tau_fx(self.x2 - self.nabla_adjoint(tmp), y)
            r = self.prox_tau_fr(self.r2 + tmp, lambda1)
            u = self.prox_sigma_g_conj(
                self.u2
                + sigma * self.epsilon(self.nabla(2 * x - self.x2) - (2 * r - self.r2)),
                lambda2,
            )
            self.x2 = self.x2 + self.rho * (x - self.x2)
            self.r2 = self.r2 + self.rho * (r - self.r2)
            self.u2 = self.u2 + self.rho * (u - self.u2)

            rel_err = torch.linalg.norm(x_prev.flatten() - self.x2.flatten()) / (
                torch.linalg.norm(self.x2.flatten()) + 1e-12
            )

            if _ > 1 and rel_err < self.crit:
                self.has_converged = True
                if self.verbose:
                    print("TGV prox reached convergence")
                break

            if self.verbose and _ % 100 == 0:
                primalcost = (
                    torch.linalg.norm(x.flatten() - y.flatten()) ** 2
                    + lambda1 * torch.sum(torch.sqrt(torch.sum(r**2, axis=-1)))
                    + lambda2
                    * torch.sum(
                        torch.sqrt(
                            torch.sum(self.epsilon(self.nabla(x) - r) ** 2, axis=-1)
                        )
                    )
                )
                # dualcost = cy - ((y - nablaT(epsilonT(u))) ** 2).sum() / 2.0
                tmp = torch.max(
                    torch.sqrt(torch.sum(self.epsilon_adjoint(u) ** 2, axis=-1))
                )  # to check feasibility: the value will be  <= lambda1 only at convergence. Since u is not feasible, the dual cost is not reliable: the gap=primalcost-dualcost can be <0 and cannot be used as stopping criterion.
                u3 = u / torch.maximum(
                    tmp / lambda1, torch.ones_like(tmp)
                )  # u3 is a scaled version of u, which is feasible. so, its dual cost is a valid, but very rough lower bound of the primal cost.
                dualcost2 = (
                    cy
                    - torch.sum((y - self.nabla_adjoint(self.epsilon_adjoint(u3))) ** 2)
                    / 2.0
                )  # we display the best value of dualcost2 computed so far.
                primalcostlowerbound = max(primalcostlowerbound, dualcost2.item())
                print(
                    "Iter: ",
                    _,
                    " Primal cost: ",
                    primalcost.item(),
                    " Rel err:",
                    rel_err,
                )

            if _ == self.n_it_max - 1:
                if self.verbose:
                    print(
                        "The algorithm did not converge, stopped after "
                        + str(_ + 1)
                        + " iterations."
                    )

        return self.x2

    @staticmethod
    def nabla(x):
        r"""
        Applies the finite differences operator associated with tensors of the same shape as x.
        """
        return TVDenoiser.nabla(x)

    @staticmethod
    def nabla_adjoint(x):
        r"""
        Applies the adjoint of the finite difference operator.
        """
        return TVDenoiser.nabla_adjoint(x)

    @staticmethod
    def epsilon(I):
        r"""
        Applies the jacobian of a vector field.
        """
        if I.ndim not in [5, 6]:
            raise ValueError(f"Input tensor must be 5D or 6D, got {I.ndim}D")

        n_spatial = (
            I.ndim - 3
        )  # Number of spatial dimensions (2 for 5D tensor, 3 for 6D tensor)
        n_components = n_spatial**2  # Jacobian components (4 for 2D, 9 for 3D)

        G = torch.zeros((*I.shape[:-1], n_components), device=I.device, dtype=I.dtype)

        # Compute all Jacobian components: d(component_i)/d(spatial_j)
        comp_idx = 0
        for i in range(n_spatial):  # Vector component index
            for j in range(n_spatial):  # Spatial derivative index
                dim = (
                    j + 2
                )  # Dimension to differentiate along (skip batch and channel dims)

                # Create slice objects for spatial differencing
                slice_from = [slice(None)] * (I.ndim - 1)
                slice_to = [slice(None)] * (I.ndim - 1)
                slice_from[dim] = slice(None, -1)
                slice_to[dim] = slice(1, None)

                # Slice for input component
                slice_input_from = (*slice_from, i)
                slice_input_to = (*slice_to, i)

                # Slice for output component
                slice_output_from = (*slice_from, comp_idx)
                slice_output_to = (*slice_to, comp_idx)

                # Apply finite difference
                G[slice_output_to] = I[slice_input_to] - I[slice_input_from]

                comp_idx += 1

        return G

    @staticmethod
    def epsilon_adjoint(G):
        r"""
        Applies the adjoint of the jacobian of a vector field.
        """
        if G.ndim not in [5, 6]:
            raise ValueError(f"Input tensor must be 5D or 6D, got {G.ndim}D")

        n_spatial = (
            G.ndim - 3
        )  # Number of spatial dimensions (2 for 5D tensor, 3 for 6D tensor)
        I = torch.zeros((*G.shape[:-1], n_spatial), device=G.device, dtype=G.dtype)

        # Apply adjoint for all Jacobian components: d(component_i)/d(spatial_j)
        comp_idx = 0
        for i in range(n_spatial):  # Vector component index
            for j in range(n_spatial):  # Spatial derivative index
                dim = (
                    j + 2
                )  # Dimension to differentiate along (skip batch and channel dims)

                # Create slice objects for spatial differencing
                slice_from = [slice(None)] * (G.ndim - 1)
                slice_to = [slice(None)] * (G.ndim - 1)
                slice_from[dim] = slice(None, -1)
                slice_to[dim] = slice(1, None)

                # Slice for output component
                slice_output_from = (*slice_from, i)
                slice_output_to = (*slice_to, i)

                # Slice for input component
                slice_input_to = (*slice_to, comp_idx)

                # Apply adjoint operator (reversed finite difference)
                I[slice_output_from] -= G[slice_input_to]
                I[slice_output_to] += G[slice_input_to]

                comp_idx += 1

        return I
