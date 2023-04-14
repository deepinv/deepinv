import torch.nn as nn
import torch
import numpy as np
import time as time
from tqdm import tqdm

import deepinv.physics


class DDRM(nn.Module):
    r'''


    '''
    def __init__(self, denoiser:deepinv.models.Denoiser, sigmas, sigma_noise, eta=.85, etab=1., verbose=False):
        super(DDRM, self).__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas
        self.max_iter = len(sigmas)
        self.sigma_noise = sigma_noise
        self.eta = eta
        self.verbose = verbose
        self.etab = etab

    def forward(self, y, physics: deepinv.physics.DecomposablePhysics, seed=None):

        #assert physics.__class__ == deepinv.physics.DecomposablePhysics, 'The forward operator requires a singular value decomposition'
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            mask = physics.mask.abs().type(y.dtype)
            c = np.sqrt(1-self.eta**2)
            y_bar = physics.U_adjoint(y)
            case = mask > 1e-6
            y_bar[case] = y_bar[case]/mask[case]
            nsr = torch.zeros_like(mask)
            nsr[case] = self.sigma_noise/mask[case]

            # iteration 1
            # compute init noise
            mean = torch.zeros_like(y_bar)
            std = torch.ones_like(y_bar)*self.sigmas[0]
            mean[case] = y_bar[case]
            std[case] = (self.sigmas[0]**2-nsr[case].pow(2)).sqrt()
            x_bar = mean + std * torch.randn_like(y_bar)
            x_bar_prev = x_bar.clone()
            # denoise
            x = self.denoiser(physics.V(x_bar), self.sigmas[0])

            for t in tqdm(range(1, self.max_iter), disable=(not self.verbose)):
                # add noise in transformed domain
                x_bar = physics.V_adjoint(x)

                case2 = case + (self.sigmas[t] < nsr)
                case3 = case + (self.sigmas[t] >= nsr)

                mean = x_bar + c*self.sigmas[t]*(x_bar_prev-x_bar)/self.sigmas[t-1]
                mean[case2] = x_bar[case2] + c*self.sigmas[t]*(y_bar[case2]-x_bar[case2])/nsr[case2]
                mean[case3] = (1.-self.etab)*x_bar[case3] + self.etab*y_bar[case3]

                std = torch.ones_like(x)*self.eta*self.sigmas[t]
                std[case3] = (self.sigmas[t]**2 - (nsr[case3]*self.etab).pow(2)).sqrt()

                #print(f'std: {std.isnan().sum()}')
                #print(f'mean: {mean.isnan().sum()}')
                x_bar = mean + std*torch.randn_like(x_bar)
                x_bar_prev = x_bar.clone()
                # denoise
                x = self.denoiser(physics.V(x_bar), self.sigmas[t])

                #dinv.utils.plot_debug([x], titles=f'it {t}')
        return x


if __name__ == "__main__":

    import deepinv as dinv
    from deepinv.models.denoiser import Denoiser
    import torchvision

    x = torchvision.io.read_image('../../datasets/celeba/img_align_celeba/085307.jpg')
    x = x.unsqueeze(0).float().to(dinv.device) / 255

    sigma_noise = .05
    #physics = dinv.physics.Denoising()
    physics = dinv.physics.Inpainting(mask=.5, tensor_size=(3, 218, 178), device=dinv.device)
    physics.noise_model = dinv.physics.GaussianNoise(sigma_noise)

    y = physics(x)
    model_spec = {'name': 'drunet', 'args': {'device': dinv.device, 'pretrained': 'download'}}

    denoiser = Denoiser(model_spec=model_spec)

    sigmas = np.linspace(1, 0, 100)
    f = DDRM(denoiser=denoiser, etab=1., sigma_noise=sigma_noise, sigmas=sigmas, verbose=True)

    xmean = f(y, physics)

    dinv.utils.plot_debug([physics.A_adjoint(y), x, xmean], titles=['meas.', 'ground-truth', 'mean'])