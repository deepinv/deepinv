from deepinv.diffops.physics.forward import Forward
import os
import torch
import numpy as np


class RandomPhaseRetrieval(Forward):
    def __init__(self, m, img_shape, bias=0, save=False, G=1, g=0, dtype=torch.float, device='cuda:0', save_dir='.'):
        super().__init__()
        self.img_shape = img_shape
        self.sqrtm = np.sqrt(m)
        n = 1
        for d in img_shape:
            n *= d

        if save:
            dir_name = save_dir + '/saved_forward/PhaseRetrieval/randompr_cs_{}x{}_G{}/'.format(n, m, G)
            file_name = 'forw_g{}.pt'.format(g)
            if os.path.exists(dir_name+file_name):
                A, A_dagger, bias = torch.load(dir_name+file_name)
                print('Random PR matrix has been LOADED from {}'.format(dir_name + file_name))
            else:
                A = np.random.randn(m, n) / np.sqrt(m)
                A_dagger = np.linalg.pinv(A)
                os.makedirs(dir_name)
                torch.save([A, A_dagger, bias], dir_name+file_name)
                print('Random PR matrix has been CREATED & SAVED at {}'.format(dir_name+file_name))
        else:
            A = np.random.randn(m, n) / np.sqrt(m)
            A_dagger = np.linalg.pinv(A)

        self.bias = torch.from_numpy(bias).type(dtype).to(device)
        self._A = torch.from_numpy(A).type(dtype).to(device)
        self._A_dagger = torch.from_numpy(A_dagger).type(dtype).to(device)
        self._A_adjoint = self._A.t().type(dtype).to(device)

    def A(self, x):
        N,C,H,W = x.shape
        x = x.reshape(N, -1)
        y = torch.einsum('in, mn->im', x, self._A)
        return y

    def forward(self, x):
        y = torch.abs(self.A(x))
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