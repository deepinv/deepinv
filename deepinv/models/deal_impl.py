import torch

from torch import nn

from .deal_linearspline import LinearSpline
from .deal_multi_conv import *


class DEAL(nn.Module):
    def __init__(self, color) -> None:
        super().__init__()

        self.kernel_size = 9
        self.conv_pad = self.kernel_size // 2
        self.color = color

        if self.color:
            self.n = 3
            channels = [3, 12, 24, 128]
        else:
            self.n = 1
            channels = [1, 4, 8, 128]

        self.last_c = channels[-1]
        self.W1 = MultiConv2d(
            channels, [self.kernel_size] * (len(channels) - 1), color=self.color
        )

        self.M1 = MultiConv2d(
            channels, [self.kernel_size] * (len(channels) - 1), color=self.color
        )
        self.M2 = nn.Conv2d(
            self.last_c, self.last_c, kernel_size=3, padding=1, bias=False, groups=1
        )
        self.M3 = nn.Conv2d(
            self.last_c, self.last_c, kernel_size=3, padding=1, bias=False, groups=1
        )

        self.spline1 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="identity",
            clamp=False,
            slope_min=0,
        )
        self.spline2 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="identity",
            clamp=False,
            slope_min=0,
        )
        self.spline3 = LinearSpline(
            num_activations=1,
            num_knots=31,
            x_min=0,
            x_max=3,
            init="gaussian",
            clamp=False,
        )

        self.spline_lambda = LinearSpline(
            num_activations=1,
            num_knots=53,
            x_min=-1,
            x_max=51,
            init="identity",
            clamp=False,
        )
        self.spline_scaling = LinearSpline(
            num_activations=self.last_c,
            num_knots=14,
            x_min=-1,
            x_max=51,
            init=3.0,
            clamp=False,
        )

        self.number_of_cgs = 0
        self.last_cg_iter = 0
        self.max_iter = 1000

        return

    def cal_lambda(self, sigma):
        self.lmbda = self.spline_lambda(sigma)

    def cal_scaling(self, sigma):
        sigma = torch.ones((sigma.size(0), self.last_c, 1, 1)).to(sigma.device) * sigma
        self.scaling = torch.exp(self.spline_scaling(sigma)) / (sigma + 1e-5)

    def last_act(self, x):
        x = torch.abs(x)
        x = self.spline3(self.scaling * x)
        return torch.clip(x, 1e-2, 1)

    def K(self, x, idx=None):
        return torch.sqrt(self.lmbda) * self.W1(x) * self.mask[idx]

    def Kt(self, y, idx=None):
        return torch.sqrt(self.lmbda) * self.W1.transpose(y * self.mask[idx])

    def KtK(self, x, idx=None):
        return self.Kt(self.K(x, idx), idx)

    def cal_mask(self, x):
        self.mask = self.last_act(
            self.M3(
                self.spline2(torch.abs(self.M2(self.spline1(torch.abs(self.M1(x))))))
            )
        )
        return self.mask

    def L(self, x, idx="None"):
        return self.W1(x) * self.mask[idx]

    def Lt(self, y, idx=None):
        return self.W1.transpose(y * self.mask[idx])

    def BtB(self, x, H, Ht, idx=None):
        BtBD = (Ht(H(x)) + self.lmbda[idx] * self.Lt(self.L(x, idx), idx)) / (
            1 + self.lmbda[idx]
        )
        return BtBD

    def cg_sample(self, b, x0, max_iter, eps=1e-5):
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        b = self.Kt(b, correct_idx_mask)
        r = b - self.KtK(x, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = list()
        idx_uniques_cont = list()
        idx_uniques_cont.append([i for i in range(x.size(0))])

        for i in range(max_iter):

            idx_cont = torch.where(r_norm.squeeze() > eps)[0].tolist()

            if i == max_iter - 1:
                idx_cont = []

            len_new = len(idx_cont)

            if len_new != len_old:
                idx_done = torch.where(r_norm.squeeze() <= eps)[0].tolist()

                if i == max_iter - 1:
                    idx_done = [h for h in range(x.size(0))]
                idx_uniques_done.append(idx_done)

                correct_idx = [idx_uniques_cont[-1][id] for id in idx_done]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx = [idx_uniques_cont[-j - 2][id] for id in correct_idx]

                correct_idx_mask = [idx_uniques_cont[-1][id] for id in idx_cont]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx_mask = [
                        idx_uniques_cont[-j - 2][id] for id in correct_idx_mask
                    ]

                output[correct_idx] = x[idx_done]
                idx_uniques_cont.append(idx_cont)
                r = r[idx_cont]
                p = p[idx_cont]
                x = x[idx_cont]
                r_norm = r_norm[idx_cont]
                len_old = len_new

            if len(idx_cont) == 0:
                break

            BTBp = self.KtK(p, correct_idx_mask)
            alpha = r_norm / ((p * BTBp).sum(dim=(1, 2, 3), keepdim=True))

            x = x + alpha * p
            r_norm_old = r_norm.clone()
            r = r - alpha * BTBp

            r_norm = (r**2).sum(dim=(1, 2, 3), keepdim=True)
            beta = r_norm / (r_norm_old)
            p = r + beta * p

        return output, i

    def cg(
        self, b, x0, max_iter, eps=1e-5, H=lambda x: x, Ht=lambda x: x, dims=(1, 2, 3)
    ):

        b = b / (1 + self.lmbda)
        x = x0.clone()
        correct_idx_mask = [i for i in range(x.size(0))]
        r = b - self.BtB(x, H, Ht, correct_idx_mask)
        p = r.clone()
        r_norm = r_norm_old = (r**2).sum(dim=(1, 2, 3), keepdim=True)
        output = torch.zeros_like(x)
        len_old = x.size(0)

        idx_uniques_done = list()
        idx_uniques_cont = list()
        idx_uniques_cont.append([i for i in range(x.size(0))])

        for i in range(max_iter):

            idx_cont = torch.where(r_norm.squeeze() > eps)[0].tolist()

            if i == max_iter - 1:
                idx_cont = []

            len_new = len(idx_cont)

            if len_new != len_old:
                idx_done = torch.where(r_norm.squeeze() <= eps)[0].tolist()

                if i == max_iter - 1:
                    idx_done = [h for h in range(x.size(0))]
                idx_uniques_done.append(idx_done)

                correct_idx = [idx_uniques_cont[-1][id] for id in idx_done]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx = [idx_uniques_cont[-j - 2][id] for id in correct_idx]

                correct_idx_mask = [idx_uniques_cont[-1][id] for id in idx_cont]
                for j in range(len(idx_uniques_cont) - 1):
                    correct_idx_mask = [
                        idx_uniques_cont[-j - 2][id] for id in correct_idx_mask
                    ]

                output[correct_idx] = x[idx_done]
                idx_uniques_cont.append(idx_cont)
                r = r[idx_cont]
                p = p[idx_cont]
                x = x[idx_cont]
                r_norm = r_norm[idx_cont]
                len_old = len_new

            if len(idx_cont) == 0:
                break

            BTBp = self.BtB(p, H, Ht, correct_idx_mask)
            alpha = r_norm / ((p * BTBp).sum(dim=(1, 2, 3), keepdim=True))

            x = x + alpha * p
            r_norm_old = r_norm.clone()
            r = r - alpha * BTBp

            r_norm = (r**2).sum(dim=(1, 2, 3), keepdim=True)
            beta = r_norm / (r_norm_old)
            p = r + beta * p

        return output, i

    def denoise(self, y, sigma):

        self.W1.spectral_norm()
        self.cal_lambda(sigma)
        self.cal_scaling(sigma)

        if self.training:
            self.c_k_list = list()
            grad_steps = 1
            n_out = torch.randint(14, 59, (1, 1))
            n_in = 50
            eps_in = 1e-4
            eps_out = 1e-4
            eps_bck = 1e-4

        else:
            grad_steps = 0
            n_out = 60
            n_in = 200
            eps_in = 1e-6
            eps_out = 1e-5

        c_k = torch.zeros_like(y)
        c_k_old = c_k.clone()

        with torch.no_grad():
            for id in range(n_out):
                self.cal_mask(c_k)
                c_k, self.last_cg_iter = self.cg(y, c_k_old, n_in, eps=eps_in)
                res = torch.norm(c_k - c_k_old) / (torch.norm(c_k_old))
                c_k_old = c_k.clone()
                if (res < eps_out).all():
                    break

        def backward_hook1(grad):
            self.cal_mask(d_k)
            g, _ = self.cg(grad, grad, n_in, eps=eps_bck)
            return g

        if self.training:
            d_k = c_k
            self.cal_mask(d_k)
            self.c_k_list.append(d_k)
            with torch.no_grad():
                c_k, self.last_cg_iter = self.cg(y, c_k, n_in, eps=eps_bck)
            idx = [i for i in range(y.size(0))]
            c_k1 = y - self.lmbda * self.Lt(self.L(c_k.detach(), idx), idx)
            c_k1.register_hook(backward_hook1)
            self.c_k_list.append(c_k1)

        else:
            c_k1 = c_k

        self.number_of_cgs = id + grad_steps

        return c_k1

    def solve_inverse_problem(
        self,
        y,
        H,
        Ht,
        sigma,
        lmbda,
        eps_in=1e-8,
        eps_out=1e-5,
        path=False,
        x_init=None,
        verbose=False,
    ):

        self.W1.spectral_norm()
        self.cal_scaling(torch.tensor([[sigma]]).to(y.device))
        self.lmbda = torch.tensor([[lmbda]]).to(y.device)
        if path:
            c_ks = list()

        # NEW: use self.max_iter to control iterations
        max_iters = getattr(self, "max_iter", 1000)
        max_cg_iters = getattr(self, "max_iter", 1000)

        with torch.no_grad():
            if x_init is not None:
                c_k = x_init
            else:
                c_k = Ht(y) * 0
            c_k_old = c_k.clone()

            for m in range(max_iters):  # instead of range(1000)
                if path:
                    c_ks.append(c_k)

                self.cal_mask(c_k)
                c_k, cg_iters = self.cg(
                    Ht(y),
                    c_k_old,
                    max_cg_iters,  # instead of 1000
                    eps=eps_in,
                    H=H,
                    Ht=Ht,
                )

                res = torch.norm(c_k - c_k_old) / (torch.norm(c_k_old))
                c_k_old = c_k.clone()

                if verbose:
                    print(
                        "CG Number:",
                        m,
                        "CG iterations:",
                        cg_iters,
                        "Outer residual:",
                        res,
                    )

                if (res < eps_out).all():
                    break

        if path:
            c_ks.append(c_k)
            return torch.clip(c_k, 0, 1), c_ks
        else:
            return torch.clip(c_k, 0, 1)
