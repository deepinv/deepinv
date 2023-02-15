import torch
import torch.nn as nn

class TGVprox(nn.Module):
    '''
    Implements the proximal operator of the (second order) Total Generalised Variation (TGV) operator.(see K. Bredies,
    K. Kunisch, and T. Pock, "Total generalized variation," SIAM J. Imaging Sci., 3(3), 492-526, 2010.)

    This algorithm converges to the unique image x (and the auxiliary vector field r) minimizing

    ||x-y||_2^2/2 + lambda1.||r||_1,2 + lambda2.||J(Dx-r)||_1,Frobenius

    where D maps an image to its gradient field and J maps a vector field to its Jacobian. For a large value of lambda2,
    the TGV behaves like the TV. For a small value, it behaves like the l1-Frobenius norm of the Hessian.

    The problem is solved with an over-relaxed Chambolle-Pock algorithm (see L. Condat, "A primal-dual splitting method
    for convex optimization  involving Lipschitzian, proximable and linear composite terms", J. Optimization Theory and
   Applications, vol. 158, no. 2, pp. 460-479, 2013.

    Code (and description) adapted from Laurent Condat's matlab version (https://lcondat.github.io/software.html) and
    Daniil Smolyakov (https://github.com/RoundedGlint585/TGVDenoising/blob/master/TGV%20WithoutHist.ipynb)
    '''
    def __init__(self, reg=1., verbose=True):
        super(TGVprox, self).__init__()

        self.verbose = verbose

        self.lambda1 = reg * 0.1
        self.lambda2 = reg * 0.15
        self.tau = 0.01  # >0

        self.rho = 1.99  # in 1,2
        self.sigma = 1 / self.tau / 72

    def prox_tau_fx(self, x, y):
        return (x + self.tau * y) / (1 + self.tau)

    def prox_tau_fr(self, r):
        left = torch.sqrt(torch.sum(r ** 2, axis=-1)) / (self.tau * self.lambda1)
        tmp = r - r / (torch.maximum(left, torch.tensor([1])).unsqueeze(-1))
        return tmp

    def prox_sigma_g_conj(self, u):
        return u / (torch.maximum(torch.sqrt(torch.sum(u ** 2, axis=-1)) / self.lambda2, torch.tensor([1])).unsqueeze(-1))

    def forward(self, y, x2=None, u2=None, r2=None, n_it_max=1000, crit=1e-5):

        if x2 is None:
            x2 = y.clone()
        if r2 is None:
#             r2 = torch.zeros((x2.shape[-3], x2.shape[-2], x2.shape[-1], 2))
            r2 = torch.zeros((*x2.shape, 2))
        if u2 is None:
#             u2 = torch.zeros((x2.shape[-3], x2.shape[-2], x2.shape[-1], 4))
            u2 = torch.zeros((*x2.shape, 4))
        cy = (y ** 2).sum() / 2
        primalcostlowerbound = 0

        for _ in range(n_it_max):
            x_prev = x2.clone()
            tmp = self.tau * epsilonT(u2)
            x = self.prox_tau_fx(x2 - nablaT(tmp), y)
            r = self.prox_tau_fr(r2 + tmp)
            u = self.prox_sigma_g_conj(u2 + self.sigma * epsilon(nabla(2 * x - x2) - (2 * r - r2)))
            x2 = x2 + self.rho * (x - x2)
            r2 = r2 + self.rho * (r - r2)
            u2 = u2 + self.rho * (u - u2)

            rel_err = torch.linalg.norm(x_prev.flatten() - x2.flatten()) / torch.linalg.norm(x2.flatten() + 1e-12)

            if _ > 1 and rel_err < crit:
                print('TGV prox reached convergence')
                break

            if self.verbose and _ % 100 == 0:
                primalcost = torch.linalg.norm(x.flatten() - y.flatten()) ** 2 + self.lambda1 * torch.sum(
                    torch.sqrt(torch.sum(r ** 2, axis=-1))) + self.lambda2 * torch.sum(
                    torch.sqrt(torch.sum(epsilon(nabla(x) - r) ** 2, axis=-1)))
                dualcost = cy - ((y - nablaT(epsilonT(u))) ** 2).sum() / 2.
                tmp = torch.max(torch.sqrt(torch.sum(epsilonT(u) ** 2,
                                                     axis=-1)))  # to check feasibility: the value will be  <= lambda1 only at convergence. Since u is not feasible, the dual cost is not reliable: the gap=primalcost-dualcost can be <0 and cannot be used as stopping criterion.
                u3 = u / torch.maximum(tmp / self.lambda1, torch.tensor([
                                                                       1]))  # u3 is a scaled version of u, which is feasible. so, its dual cost is a valid, but very rough lower bound of the primal cost.
                dualcost2 = cy - torch.sum(
                    (y - nablaT(epsilonT(u3))) ** 2) / 2.  # we display the best value of dualcost2 computed so far.
                primalcostlowerbound = max(primalcostlowerbound, dualcost2.item())
                print('Iter: ', _, ' Primal cost: ', primalcost.item(), ' Rel err:', rel_err)

        # return x2, r2, u2
        return x2  # TODO: to allow warm restart, we would need to output r2 and u2

def nabla(I):
    b, c, h, w = I.shape
    G = torch.zeros((b, c, h, w, 2)).type(I.dtype)
    G[:, :, :-1, :, 0] -= I[:, :, :-1]
    G[:, :, :-1, :, 0] += I[:, :, 1:]
    G[:, :, :, :-1, 1] -= I[..., :-1]
    G[:, :, :, :-1, 1] += I[..., 1:]
    return G


def nablaT(G):
    b, c, h, w = G.shape[:-1]
    I = torch.zeros(b, c, h, w).type(G.dtype) # note that we just reversed left and right sides of each line to obtain the transposed operator
    I[:, :, :-1] -= G[:, :, :-1, :, 0]
    I[:, :, 1:] += G[:, :, :-1, :, 0]
    I[..., :-1] -= G[..., :-1, 1]
    I[..., 1:] += G[..., :-1, 1]
    return I

# # ADJOINTNESS TEST
# u = torch.randn((4, 3, 100,100)).type(torch.DoubleTensor)
# Au = nabla(u)
# v = torch.randn(*Au.shape).type(Au.dtype)
# Atv = nablaT(v)
# e = v.flatten()@Au.flatten()-Atv.flatten()@u.flatten()
# print('Adjointness test (should be small): ', e)


def epsilon(I): # Simplified
    b, c, h, w, _ = I.shape
    G = torch.zeros((b, c, h, w, 4)).type(I.dtype)
    G[:, :, 1:, :, 0] -= I[:, :, :-1, :, 0]  # xdy
    G[..., 0] += I[..., 0]
    G[..., 1:, 1] -= I[..., :-1, 0]  # xdx
    G[..., 1:, 1] += I[..., 1:, 0]
    G[..., 1:, 2] -= I[..., :-1, 1]  # xdx
    G[..., 2] += I[..., 1]
    G[:, :, :-1, :, 3] -= I[:, :, :-1, :, 1]  # xdy
    G[:, :, :-1, :, 3] += I[:, :, 1:, :, 1]
    return G


def epsilonT(G):
    b, c, h, w, _ = G.shape
    I = torch.zeros((b, c, h, w, 2)).type(G.dtype)
    I[:, :, :-1, :, 0] -= G[:, :, 1:, :, 0]
    I[..., 0] += G[..., 0]
    I[..., :-1, 0] -= G[..., 1:, 1]
    I[..., 1:, 0] += G[..., 1:, 1]
    I[..., :-1, 1] -= G[..., 1:, 2]
    I[..., 1] += G[..., 2]
    I[:, :, :-1, :, 1] -= G[:, :, :-1, :, 3]
    I[:, :, 1:, :, 1] += G[:, :, :-1, :, 3]
    return I

# # ADJOINTNESS TEST
# u = torch.randn((2, 3,100,100,2)).type(torch.DoubleTensor)
# Au = epsilon(u)
# v = torch.randn(*Au.shape).type(Au.dtype)
# Atv = epsilonT(v)
# e = v.flatten()@Au.flatten()-Atv.flatten()@u.flatten()
# print('Adjointness test (should be small): ', e)