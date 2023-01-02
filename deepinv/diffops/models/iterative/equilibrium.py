import torch
import torch.nn as nn
import deepinv.diffops.models as models


class ModelOptimization(nn.Module):
    def __init__(self, mode, step_size, model, device):
        super(ModelOptimization, self).__init__()
        assert mode in ['lfb', 'pgd', 'gd', 'admm']

        self.name = f'equilibrium-{mode}'
        self.mode = mode
        self.model = model

        # self.time_step = time_step # maximum number of iterations in Fixed-point calculation


        self.step_size = step_size # eta
        # self.register_parameter(name='step_size', param=torch.nn.Parameter(torch.tensor(step_size), requires_grad=True))

        # self.block_config = block_config
        # self.block =models.__dict__[block_arch](**self.block_config).to(device)

        # gradient of (least square) data consistency term
        # self.dc_grad = lambda x,y,physics: physics.A_adjoint(y - physics.A(x))
        self.dc_grad = lambda x, y, physics: physics.A_adjoint(
            y.to(physics.device) - physics.A(x.to(physics.device))).to(x.device)

    def forward(self, y, physics, z_init=None):
        z = physics.A_dagger(y).to(y.device) if z_init is None else z_init.clone()

        if self.mode == 'lfb':  # learned forward backward
            z = self.model(torch.cat([z, self.dc_grad(z, y, physics)], dim=1))
            z = z[:, :self.block_config['out_channels'], ...]
        if self.mode == 'pgd':  # proximal gradient descent
            # s = x + self.step_size * self.dc_grad(x, y, physics)
            # x = s + self.block(s)
            z = self.model(z + self.step_size * self.dc_grad(z, y, physics))

        if self.mode == 'gd':  # gradient descent
            z = z + self.step_size * (self.dc_grad(z, y, physics) + self.model(z))

        return z

# class ModelOptimization(nn.Module):
#     def __init__(self, mode, step_size, block_arch, block_config, device):
#         super(ModelOptimization, self).__init__()
#         assert mode in ['lfb', 'pgd', 'gd', 'admm']
#
#         self.name = f'equilibrium-{mode}'
#         self.mode = mode
#
#         # self.time_step = time_step # maximum number of iterations in Fixed-point calculation
#
#
#         self.step_size = step_size # eta
#         # self.register_parameter(name='step_size', param=torch.nn.Parameter(torch.tensor(step_size), requires_grad=True))
#
#         self.block_config = block_config
#         self.block =models.__dict__[block_arch](**self.block_config).to(device)
#
#         # gradient of (least square) data consistency term
#         # self.dc_grad = lambda x,y,physics: physics.A_adjoint(y - physics.A(x))
#         self.dc_grad = lambda x, y, physics: physics.A_adjoint(
#             y.to(physics.device) - physics.A(x.to(physics.device))).to(x.device)
#
#     def forward(self, y, physics, z_init=None):
#         z = physics.A_dagger(y).to(y.device) if z_init is None else z_init.clone()
#
#         if self.mode == 'lfb':  # learned forward backward
#             z = self.block(torch.cat([z, self.dc_grad(z, y, physics)], dim=1))
#             z = z[:, :self.block_config['out_channels'], ...]
#         if self.mode == 'pgd':  # proximal gradient descent
#             # s = x + self.step_size * self.dc_grad(x, y, physics)
#             # x = s + self.block(s)
#             z = self.block(z + self.step_size * self.dc_grad(z, y, physics))
#
#         if self.mode == 'gd':  # gradient descent
#             z = z + self.step_size * (self.dc_grad(z, y, physics) + self.block(z))
#
#         return z



class FixedPoint(nn.Module):
    # def __init__(self, f, fixed_point_solver=AndersonExp(), **kwargs): #todo: fixed_point_solver=AndersonExp (default), f=PGD (default), Done
    def __init__(self, f, **kwargs): #todo: fixed_point_solver=AndersonExp (default), f=PGD (default), Done
        super(FixedPoint, self).__init__()

        self.f = f # single step mobel-based optimization
                   # using DNN proximal operaetor: [GD, PGD, LFB] network
        # self.fixed_point_solver = fixed_point_solver # anderson

        self.fixed_point_solver = AndersonExp()

        # self.f = EquilibriumIteration()
        # self.fixed_point_solver = andersonexp

        self.kwargs = kwargs

    def forward(self, y, physics, initial_point = None): # todo: init_point=A^Ty (default)
        if initial_point is None:
            init_point = torch.zeros_like(y)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            # z, self.forward_res = self.fixed_point_solver(lambda z: self.f(z, physics, x), init_point, **self.kwargs)
            z, self.forward_res = self.fixed_point_solver(lambda z: self.f(z, physics, y), init_point)
        z = self.f(z, physics, y)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, physics, y)

        def backward_hook(grad):
            g, self.backward_res = self.fixed_point_solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

    # def forward(self, x, physics, initial_point = None):
    #     if initial_point is None:
    #         init_point = torch.zeros_like(x)
    #     else:
    #         init_point = initial_point
    #     # compute forward pass and re-engage autograd tape
    #     with torch.no_grad():
    #         # z, self.forward_res = self.fixed_point_solver(lambda z: self.f(z, physics, x), init_point, **self.kwargs)
    #         z, self.forward_res = self.fixed_point_solver(lambda z: self.f(z, physics, x), init_point)
    #     z = self.f(z, physics, x)
    #
    #     # set up Jacobian vector product (without additional forward calls)
    #     z0 = z.clone().detach().requires_grad_()
    #     f0 = self.f(z0, physics, x)
    #
    #     def backward_hook(grad):
    #         g, self.backward_res = self.fixed_point_solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
    #                                            grad, **self.kwargs)
    #         return g
    #
    #     z.register_hook(backward_hook)
    #     return z


class AndersonExp(nn.Module):
    """ Anderson acceleration for fixed point iteration. """
    def __init__(self, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
        super(AndersonExp, self).__init__()
        self.m = m
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol
        self.beta = beta

    def forward(self, f, x0):
        bsz, d, H, W = x0.shape
        X = torch.zeros(bsz, self.m, d * H * W, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, self.m, d * H * W, dtype=x0.dtype, device=x0.device)
        X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
        X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)

        H = torch.zeros(bsz, self.m + 1, self.m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, self.m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1

        current_k = 0
        for k in range(2, self.max_iter):
            current_k = k
            n = min(k, self.m)
            G = F[:, :n] - X[:, :n]
            H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + self.lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
                None]
            alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

            X[:, k % self.m] = self.beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - self.beta) * (alpha[:, None] @ X[:, :n])[:, 0]
            F[:, k % self.m] = f(X[:, k % self.m].reshape(x0.shape)).reshape(bsz, -1)
            res = (F[:, k % self.m] - X[:, k % self.m]).norm().item() / (1e-5 + F[:, k % m].norm().item())

            if (res < self.tol):
                break
        # tt += bsz
        return X[:, current_k % self.m].view_as(x0), res


# todo: a jupyter example