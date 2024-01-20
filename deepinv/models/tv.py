import torch
import torch.nn as nn


class TV(nn.Module):
    r"""
    Proximal operator of the isotropic Total Variation operator.

    This algorithm converges to the unique image :math:`x` that is the solution of

    .. math::

        \underset{x}{\arg\min} \;  \frac{1}{2}\|x-y\|_2^2 + \lambda \|Dx\|_{1,2},

    where :math:`D` maps an image to its gradient field.

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
    """

    def __init__(
        self,
        verbose=False,
        n_it_max=1000,
        crit=1e-5,
        x2=None,
        u2=None,
    ):
        super(TV, self).__init__()

        self.verbose = verbose
        self.n_it_max = n_it_max
        self.crit = crit
        self.restart = True

        self.tau = 0.01  # > 0

        self.rho = 1.99  # in 1,2
        self.sigma = 1 / self.tau / 72

        self.x2 = x2
        self.u2 = u2

        self.has_converged = False

    def prox_tau_fx(self, x, y):
        return (x + self.tau * y) / (1 + self.tau)

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
            self.u2 = torch.zeros((*self.x2.shape, 2), device=self.x2.device).type(
                self.x2.dtype
            )
            self.restart = False

        if ths is not None:
            lambd = ths

        for _ in range(self.n_it_max):
            x_prev = self.x2.clone()

            x = self.prox_tau_fx(self.x2 - self.tau * nablaT(self.u2), y)
            u = self.prox_sigma_g_conj(
                self.u2 + self.sigma * nabla(2 * x - self.x2), lambd
            )
            self.x2 = self.x2 + self.rho * (x - self.x2)
            self.u2 = self.u2 + self.rho * (u - self.u2)

            rel_err = torch.linalg.norm(
                x_prev.flatten() - self.x2.flatten()
            ) / torch.linalg.norm(self.x2.flatten() + 1e-12)

            if _ > 1 and rel_err < self.crit:
                if self.verbose:
                    print("TV prox reached convergence")
                break

            if _ % 100 == 0 and self.verbose:
                primalcost = 0.5 * torch.linalg.norm(
                    self.x2.flatten() - y.flatten()
                ) ** 2 + lambd * torch.sum(
                    torch.sqrt(torch.sum(nabla(self.x2) ** 2, axis=-1))
                )
                dualcost = (y**2).sum() / 2 - torch.sum(
                    (y - nablaT(self.u2)) ** 2
                ) / 2.0
                primalcostlowerbound = max(primalcost, dualcost)
                print("Iter ", _, "primal cost :", primalcost.item())

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
