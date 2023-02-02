import torch
import numpy as np
from .radon import Radon, IRadon
from deepinv.diffops.physics.forward import Forward

class CT(Forward):
    def __init__(self, img_width, radon_view, uniform=True, circle = False, device='cuda:0', I0=1e5):
        if uniform:
            theta = np.linspace(0, 180, radon_view, endpoint=False)
        else:
            theta = torch.arange(radon_view)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)

        self.name='ct'
        self.I0 = I0

        # used for normalzation input
        self.MAX = 0.032 / 5
        self.MIN = 0

    def forward(self, x):
        m = self.I0 * torch.exp(-self.radon(x))  # clean GT measurement
        m = self.noise(m) # Mixed-Poisson-Gaussian Noise
        return m

    def A(self, x, add_noise=False):
        m = self.I0 * torch.exp(-self.radon(x)) # clean GT measurement

        return m

    def A_dagger(self, y):
        return self.iradon(y)