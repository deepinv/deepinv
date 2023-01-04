from deepinv.diffops.physics.forward import Forward
import os
import torch
import numpy as np


class CompressedSensing(Forward):
    def __init__(self, m, img_shape, dtype=torch.float, device='cuda:0'):
        super().__init__()
        self.name = f'CS_m{m}'
        self.img_shape = img_shape

        n = 1
        for d in img_shape:
            n *= d

        A = np.random.randn(m, n) / np.sqrt(m)
        A_dagger = np.linalg.pinv(A)

        self._A = torch.from_numpy(A).type(dtype).to(device)
        self._A_dagger = torch.from_numpy(A_dagger).type(dtype).to(device)

        self._A = torch.nn.Parameter(self._A, requires_grad=False)
        self._A_dagger = torch.nn.Parameter(self._A_dagger, requires_grad=False)
        self._A_adjoint = torch.nn.Parameter(self._A.t(), requires_grad=False).type(dtype).to(device)


    def A(self, x):
        N = x.shape[0]
        x = x.reshape(N, -1)
        y = torch.einsum('in, mn->im', x, self._A)

        return y

    def A_dagger(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        x = torch.einsum('im, nm->in', y, self._A_dagger)
        x = x.reshape(N, C, H, W)
        return x

    def A_adjoint(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        x = torch.einsum('im, nm->in', y, self._A_adjoint)  # x:(N, n, 1)
        x = x.reshape(N, C, H, W)
        return x

