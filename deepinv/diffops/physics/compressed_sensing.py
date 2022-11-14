from deepinv.diffops.physics.forward import Forward
import os
import torch
import numpy as np


class CompressedSensing(Forward):
    def __init__(self, m, img_shape, save=False, G=None, g=None, dtype=torch.float, device='cuda:0'):
        super().__init__()
        self.img_shape = img_shape

        n = 1
        for d in img_shape:
            n *= d

        if save:
            fname = './physics/forw_cs_{}x{}_G{}_g{}.pt'.format(n, m, G, g)
            if os.path.exists(fname):
                A, A_dagger = torch.load(fname)
            else:
                A = np.random.randn(m, n) / np.sqrt(m)
                A_dagger = np.linalg.pinv(A)
                torch.save([A, A_dagger], fname)
                print('CS matrix has been CREATED & SAVED at {}'.format(fname))
        else:
            A = np.random.randn(m, n) / np.sqrt(m)
            A_dagger = np.linalg.pinv(A)


        self._A = torch.from_numpy(A).type(dtype).to(device)
        self._A_dagger = torch.from_numpy(A_dagger).type(dtype).to(device)
        self._A_adjoint = self._A.t().type(dtype).to(device)

    def A(self, x):
        N,C,H,W = x.shape
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