import torch
import numpy as np
from .radon import Radon, IRadon
from deepinv.diffops.physics.forward import Forward

class CT(Forward):
    '''
        TODO
    '''
    def __init__(self, img_width, views, uniform=True,
                 circle=False, device='cuda:0', I0=1e5):
        '''
        TODO
        :param img_width:
        :param views:
        :param uniform:
        :param circle:
        :param device:
        :param I0:
        '''
        super().__init__()
        if uniform:
            theta = np.linspace(0, 180, views, endpoint=False)
        else:
            theta = torch.arange(views)
        self.radon = Radon(img_width, theta, circle).to(device)
        self.iradon = IRadon(img_width, theta, circle).to(device)
        self.I0 = I0

        # used for normalzation input
        self.MAX = 0.032 / 5
        self.MIN = 0

    def forward(self, x):
        m = self.I0 * torch.exp(-self.radon(x))  # clean GT measurement
        m = self.noise(m) # Mixed-Poisson-Gaussian Noise
        return m

    def A(self, x):
        m = self.I0 * torch.exp(-self.radon(x))  # clean GT measurement

        return m

    def A_adjoint(self, y): # TODO: true adjoint
        return self.iradon(y)

    def A_dagger(self, y):
        return self.iradon(y)

