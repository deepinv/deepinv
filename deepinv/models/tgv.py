import warnings
import torch
import torch.nn as nn


class TGV(nn.Module):
    r"""
    Proximal operator of (2nd order) Total Generalised Variation operator.

    (see K. Bredies, K. Kunisch, and T. Pock, "Total generalized variation," SIAM J. Imaging Sci., 3(3), 492-526, 2010.)

    This algorithm converges to the unique image :math:`x` (and the auxiliary vector field :math:`r`) minimizing

    .. math::

        \underset{x, r}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda_1 \|r\|_{1,2} + \lambda_2 \|J(Dx-r)\|_{1,F}

    where :math:`D` maps an image to its gradient field and :math:`J` maps a vector field to its Jacobian.
    For a large value of :math:`\lambda_2`, the TGV behaves like the TV.
    For a small value, it behaves like the :math:`\ell_1`-Frobenius norm of the Hessian.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
    Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov's `code <https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb>`_.

    :param bool verbose: Whether to print computation details or not. Default: False.
    :param int n_it_max: Maximum number of iterations. Default: 1000.
    :param float crit: Convergence criterion. Default: 1e-5.
    :param torch.tensor, None x2: Primary variable. Default: None.
    :param torch.tensor, None u2: Dual variable. Default: None.
    :param torch.tensor, None r2: Auxiliary variable. Default: None.
    """

    def __init__(
        self, verbose=False, n_it_max=1000, crit=1e-5, x2=None, u2=None, r2=None
    ):
        super(TGV, self).__init__()

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = 0.01  # > 0

        self.rho = 1.99  # in 1,2
        self.sigma = 1 / self.tau / 72

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

    def forward(self, y, ths=None):
        restart = (
            True
            if (self.restart or self.x2 is None or self.x2.shape != y.shape)
            else False
        )

        if restart:
            self.x2 = y.clone()
            self.r2 = torch.zeros((*self.x2.shape, 2), device=self.x2.device).type(
                self.x2.dtype
            )
            self.u2 = torch.zeros((*self.x2.shape, 4), device=self.x2.device).type(
                self.x2.dtype
            )
            self.restart = False

        if ths is not None:
            lambda1 = ths * 0.1
            lambda2 = ths * 0.15

        cy = (y**2).sum() / 2
        primalcostlowerbound = 0

        for _ in range(self.n_it_max):
            x_prev = self.x2.clone()
            tmp = self.tau * epsilonT(self.u2)
            x = self.prox_tau_fx(self.x2 - nablaT(tmp), y)
            r = self.prox_tau_fr(self.r2 + tmp, lambda1)
            u = self.prox_sigma_g_conj(
                self.u2
                + self.sigma * epsilon(nabla(2 * x - self.x2) - (2 * r - self.r2)),
                lambda2,
            )
            self.x2 = self.x2 + self.rho * (x - self.x2)
            self.r2 = self.r2 + self.rho * (r - self.r2)
            self.u2 = self.u2 + self.rho * (u - self.u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - self.x2.flatten()
            ) / torch.linalg.norm(self.x2.flatten() + 1e-12)

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
                        torch.sqrt(torch.sum(epsilon(nabla(x) - r) ** 2, axis=-1))
                    )
                )
                # dualcost = cy - ((y - nablaT(epsilonT(u))) ** 2).sum() / 2.0
                tmp = torch.max(
                    torch.sqrt(torch.sum(epsilonT(u) ** 2, axis=-1))
                )  # to check feasibility: the value will be  <= lambda1 only at convergence. Since u is not feasible, the dual cost is not reliable: the gap=primalcost-dualcost can be <0 and cannot be used as stopping criterion.
                u3 = u / torch.maximum(
                    tmp / lambda1, torch.tensor([1], device=tmp.device).type(tmp.dtype)
                )  # u3 is a scaled version of u, which is feasible. so, its dual cost is a valid, but very rough lower bound of the primal cost.
                dualcost2 = (
                    cy - torch.sum((y - nablaT(epsilonT(u3))) ** 2) / 2.0
                )  # we display the best value of dualcost2 computed so far.
                primalcostlowerbound = max(primalcostlowerbound, dualcost2.item())
                if self.verbose:
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


def nabla(I):
    b, c, h, w = I.shape
    G = torch.zeros((b, c, h, w, 2), device=I.device).type(I.dtype)
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] - I[:, :, :-1]
    G[:, :, :-1, :, 0] = G[:, :, :-1, :, 0] + I[:, :, 1:]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] - I[..., :-1]
    G[:, :, :, :-1, 1] = G[:, :, :, :-1, 1] + I[..., 1:]
    return G


def nablaT(G):
    b, c, h, w = G.shape[:-1]
    I = torch.zeros((b, c, h, w), device=G.device).type(
        G.dtype
    )  # note that we just reversed left and right sides of each line to obtain the transposed operator
    I[:, :, :-1] = I[:, :, :-1] - G[:, :, :-1, :, 0]
    I[:, :, 1:] = I[:, :, 1:] + G[:, :, :-1, :, 0]
    I[..., :-1] = I[..., :-1] - G[..., :-1, 1]
    I[..., 1:] = I[..., 1:] + G[..., :-1, 1]
    return I


# # ADJOINTNESS TEST
# u = torch.randn((4, 3, 100,100)).type(torch.DoubleTensor)
# Au = nabla(u)
# v = torch.randn(*Au.shape).type(Au.dtype)
# Atv = nablaT(v)
# e = v.flatten()@Au.flatten()-Atv.flatten()@u.flatten()
# print('Adjointness test (should be small): ', e)


def epsilon(I):  # Simplified
    b, c, h, w, _ = I.shape
    G = torch.zeros((b, c, h, w, 4), device=I.device).type(I.dtype)
    G[:, :, 1:, :, 0] = G[:, :, 1:, :, 0] - I[:, :, :-1, :, 0]  # xdy
    G[..., 0] = G[..., 0] + I[..., 0]
    G[..., 1:, 1] = G[..., 1:, 1] - I[..., :-1, 0]  # xdx
    G[..., 1:, 1] = G[..., 1:, 1] + I[..., 1:, 0]
    G[..., 1:, 2] = G[..., 1:, 2] - I[..., :-1, 1]  # xdx
    G[..., 2] = G[..., 2] + I[..., 1]
    G[:, :, :-1, :, 3] = G[:, :, :-1, :, 3] - I[:, :, :-1, :, 1]  # xdy
    G[:, :, :-1, :, 3] = G[:, :, :-1, :, 3] + I[:, :, 1:, :, 1]
    return G


def epsilonT(G):
    b, c, h, w, _ = G.shape
    I = torch.zeros((b, c, h, w, 2), device=G.device).type(G.dtype)
    I[:, :, :-1, :, 0] = I[:, :, :-1, :, 0] - G[:, :, 1:, :, 0]
    I[..., 0] = I[..., 0] + G[..., 0]
    I[..., :-1, 0] = I[..., :-1, 0] - G[..., 1:, 1]
    I[..., 1:, 0] = I[..., 1:, 0] + G[..., 1:, 1]
    I[..., :-1, 1] = I[..., :-1, 1] - G[..., 1:, 2]
    I[..., 1] = I[..., 1] + G[..., 2]
    I[:, :, :-1, :, 1] = I[:, :, :-1, :, 1] - G[:, :, :-1, :, 3]
    I[:, :, 1:, :, 1] = I[:, :, 1:, :, 1] + G[:, :, :-1, :, 3]
    return I


# # ADJOINTNESS TEST
# u = torch.randn((2, 3,100,100,2)).type(torch.DoubleTensor)
# Au = epsilon(u)
# v = torch.randn(*Au.shape).type(Au.dtype)
# Atv = epsilonT(v)
# e = v.flatten()@Au.flatten()-Atv.flatten()@u.flatten()
# print('Adjointness test (should be small): ', e)
