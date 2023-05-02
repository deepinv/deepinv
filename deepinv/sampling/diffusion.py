import torch.nn as nn
import torch
import numpy as np
import time as time
from tqdm import tqdm

import deepinv.physics
from deepinv.sampling.utils import Welford, projbox, refl_projbox
from deepinv.sampling.langevin import MCMC


class DDRM(nn.Module):
    r"""
    Denoising Diffusion Restoration Models (DDRM).

    This class implements the denoising diffusion restoration model (DDRM) described in https://arxiv.org/abs/2201.11793.

    The DDRM is a sampling method that uses a denoiser to sample from the posterior distribution of the inverse problem.

    It requires that the physics operator has a singular value decomposition, i.e.,
    it is :meth:`deepinv.Physics.DecomposablePhysics` class.

    :param deepinv.models.Denoiser, torch.nn.Module denoiser: a denoiser model
    :param list[int], numpy.array sigmas: a list of noise levels
    :param float sigma_noise: the noise level of the data
    :param float eta: hyperparameter
    :param float etab: hyperparameter
    :param bool verbose: if True, print progress
    """

    def __init__(
            self,
            denoiser,
            sigmas,
            sigma_noise,
            eta=0.85,
            etab=1.0,
            verbose=False,
    ):
        super(DDRM, self).__init__()
        self.denoiser = denoiser
        self.sigmas = sigmas
        self.max_iter = len(sigmas)
        self.sigma_noise = sigma_noise
        self.eta = eta
        self.verbose = verbose
        self.etab = etab

    def forward(self, y, physics: deepinv.physics.DecomposablePhysics, seed=None):
        r"""
        Runs the diffusion to obtain a random sample of the posterior distribution.

        :param torch.Tensor y: the measurements
        :param deepinv.physics.DecomposablePhysics physics: the physics operator
        :param int seed: the seed for the random number generator
        """
        # assert physics.__class__ == deepinv.physics.DecomposablePhysics, 'The forward operator requires a singular value decomposition'
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            if physics.__class__ == deepinv.physics.Denoising:
                mask = torch.ones_like(y)
            else:
                mask = physics.mask.abs()

            c = np.sqrt(1 - self.eta ** 2)
            y_bar = physics.U_adjoint(y)
            case = mask > 1e-6
            y_bar[case] = y_bar[case] / mask[case]
            nsr = torch.zeros_like(mask)
            nsr[case] = self.sigma_noise / mask[case]

            # iteration 1
            # compute init noise
            mean = torch.zeros_like(y_bar)
            std = torch.ones_like(y_bar) * self.sigmas[0]
            mean[case] = y_bar[case]
            std[case] = (self.sigmas[0] ** 2 - nsr[case].pow(2)).sqrt()
            x_bar = mean + std * torch.randn_like(y_bar)
            x_bar_prev = x_bar.clone()
            # denoise
            x = self.denoiser(physics.V(x_bar), self.sigmas[0])

            for t in tqdm(range(1, self.max_iter), disable=(not self.verbose)):
                # add noise in transformed domain
                x_bar = physics.V_adjoint(x)

                case2 = torch.logical_and(case, (self.sigmas[t] < nsr))
                case3 = torch.logical_and(case, (self.sigmas[t] >= nsr))

                # n = np.prod(mask.shape)
                # print(f'case: {case.sum()/n:.2f}, case2: {case2.sum()/n:.2f}, case3: {case3.sum()/n:.2f}')

                mean = (
                        x_bar
                        + c * self.sigmas[t] * (x_bar_prev - x_bar) / self.sigmas[t - 1]
                )
                mean[case2] = (
                        x_bar[case2]
                        + c * self.sigmas[t] * (y_bar[case2] - x_bar[case2]) / nsr[case2]
                )
                mean[case3] = (1.0 - self.etab) * x_bar[case3] + self.etab * y_bar[
                    case3
                ]

                std = torch.ones_like(x) * self.eta * self.sigmas[t]
                std[case3] = (
                        self.sigmas[t] ** 2 - (nsr[case3] * self.etab).pow(2)
                ).sqrt()

                x_bar = mean + std * torch.randn_like(x_bar)
                x_bar_prev = x_bar.clone()
                # denoise
                x = self.denoiser(physics.V(x_bar), self.sigmas[t])

        return x


class DiffusionSampler(MCMC):
    def __init__(self, diffusion, max_iter=1e2, clip=(-1, 2), g_statistic=lambda x: x, verbose=True):
        # generate an iterator
        # set the params of the base class
        data_fidelity = None
        diffusion.verbose = False
        prior = diffusion

        def iterator(x, y, physics, likelihood, prior):
            # run one sampling kernel iteration
            x = prior(y, physics)
            return x

        super().__init__(iterator, prior, data_fidelity, max_iter=max_iter, thinning=1,
                         burnin_ratio=0., clip=clip, verbose=verbose, g_statistic=g_statistic)


if __name__ == "__main__":
    import deepinv as dinv
    from deepinv.models.denoiser import Denoiser
    import torchvision

    x = torchvision.io.read_image("../../datasets/celeba/img_align_celeba/085307.jpg")
    x = x.unsqueeze(0).float().to(dinv.device) / 255

    sigma_noise = 0.2
    # physics = dinv.physics.Denoising()
    #physics = dinv.physics.BlurFFT(img_size=x.shape[1:], filter=dinv.physics.blur.gaussian_blur(),
    #                               device=dinv.device)
    physics = dinv.physics.Inpainting(
       mask=0.5, tensor_size=(3, 218, 178), device=dinv.device
    )
    physics.noise_model = dinv.physics.GaussianNoise(sigma_noise)

    y = physics(x)
    model_spec = {
        "name": "drunet",
        "args": {"device": dinv.device, "pretrained": "download"},
    }

    denoiser = Denoiser(model_spec=model_spec)

    sigmas = np.linspace(1, 0, 300)
    f = DDRM(
        denoiser=denoiser,
        etab=1.0,
        sigma_noise=sigma_noise,
        sigmas=sigmas,
        verbose=True,
    )

    xhat = f(y, physics)

    dinv.utils.plot_debug(
        [physics.A_adjoint(y), x, xhat], titles=["meas.", "ground-truth", "mean"]
    )
