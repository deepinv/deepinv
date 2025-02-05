import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import deepinv.physics
from deepinv.models import Reconstructor
from deepinv.utils.plotting import plot


class PnPFlow(Reconstructor):

    def __init__(
        self,
        model,
        data_fidelity,
        max_iter=100,
        n_avg=2,
        lr=1e-3,
        device='cuda',
        verbose=False,
    ):
        super(PnPFlow, self).__init__()
        self.model = model
        self.max_iter = max_iter
        self.data_fidelity = data_fidelity
        self.n_avg = n_avg
        self.lr = lr
        self.verbose = verbose
        self.device = device

    def denoiser(self, x, t):
        return x + (1-t.view(-1, 1, 1, 1)) * self.model(x, t)

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def forward(self, y, physics: deepinv.physics.DecomposablePhysics, seed=None):
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            if hasattr(physics.noise_model, "sigma"):
                sigma_noise = physics.noise_model.sigma
            else:
                sigma_noise = 0.01

            if physics.__class__ == deepinv.physics.Denoising:
                mask = torch.ones_like(
                    y
                )  # TODO: fix for economic SVD decompositions (eg. Decolorize)
            else:
                mask = torch.cat([physics.mask.abs()] * y.shape[0], dim=0)

            y_bar = physics.U_adjoint(y)
            x = y_bar

            delta = 1 / self.max_iter

            for iter in tqdm(range(1, self.max_iter), disable=(not self.verbose)):
                t = torch.ones(
                    len(x), device=self.device) * delta * iter
                lr_t = self.lr * (1 - t.view(-1, 1, 1, 1))
                z = x - lr_t * self.data_fidelity.grad(x, y, physics)
                x_new = torch.zeros_like(x)
                for _ in range(self.n_avg):
                    z_tilde = self.interpolation_step(z, t.view(-1, 1, 1, 1))
                    x_new += self.denoiser(z_tilde, t)
                x_new /= self.n_avg
                x = x_new
        return x
