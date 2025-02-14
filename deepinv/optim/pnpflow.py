import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import deepinv.physics
from deepinv.models import Reconstructor


class PnPFlow(Reconstructor):
    r"""PnPFlow(self, model, data_fidelity, max_iter=100, n_avg=2, lr=11.0,lr_exp=0.5,device='cuda',verbose=False)
    PnP with Flow Matching model.

    This class implements the pnp flow matching restoration model (PnPFlow) described in https://arxiv.org/pdf/2410.02423.

    PnPFlow is a reconstruction method that uses a denoiser made from a generative flow matching model to in a plug-and-play (PnP) fashion.

    :param torch.nn.Module model: a flowunet model
    :param deepinv.optim.DataFidelity data_fidelity: the data fidelity operator
    :param int max_iter: the number of iterations to run the algorithm (default: 100)
    :param int n_avg: hyperparameter
    :param float lr: hyperparameter to define the time dependant learning rate as lr_t = lr * (1-t)**lr_exp
    :param float lr_exp: hyperparameter to define the time dependant learning rate as lr_t = lr * (1-t)**lr_exp
    :param str device: the device to use for the computations
    :param bool verbose: if True, print progress

    |sep|

    :Examples:

        PnPFlow restoration model using a pretrained FlowUNet denoiser:

        >>> import deepinv as dinv
        >>> device = dinv.utils.get_freer_gpu(verbose=False) if torch.cuda.is_available() else 'cpu'
        >>> seed = torch.manual_seed(0) # Random seed for reproducibility
        >>> seed = torch.cuda.manual_seed(0) # Random seed for reproducibility on GPU
        >>> x = 0.5 * torch.ones(1, 3, 128, 128, device=device) # Define plain gray 128x128 image
        >>> physics = dinv.physics.Inpainting(
        ...   mask=0.5, tensor_size=(3, 128, 128),
        ...   noise_model=dinv.physics.GaussianNoise(0.1),
        ...   device=device,
        ... )
        >>> y = physics(x) # measurements
        >>> model =  FlowUNet(input_channels=3,input_height=128, pretrained=True, device=device)
        >>> method = dinv.optim.PnPFlow(model=model, data_fidelity=L2(),verbose=True, max_iter=100, device=device, lr=1.0, lr_exp=0.5) #define the PnPFlow model
        >>> xhat = model(y, physics) # sample from the posterior distribution
        >>> dinv.metric.PSNR()(xhat, x) > dinv.metric.PSNR()(y, x) # Should be closer to the original
        tensor([True])

    """

    def __init__(
        self,
        model,
        data_fidelity,
        max_iter=100,
        n_avg=2,
        lr=1.0,
        lr_exp=0.5,
        device="cuda",
        verbose=False,
    ):
        super(PnPFlow, self).__init__()
        self.model = model
        self.max_iter = max_iter
        self.data_fidelity = data_fidelity
        self.n_avg = n_avg
        self.lr = lr
        self.lr_exp = lr_exp
        self.verbose = verbose
        self.device = device

    def denoiser(self, x, t):
        return x + (1 - t.view(-1, 1, 1, 1)) * self.model(x, t)

    def interpolation_step(self, x, t):
        return t * x + torch.randn_like(x) * (1 - t)

    def forward(self, y, physics, x_init=None, seed=None):
        r"""
        Runs the iterative pnpflow algorithm for solving :ref:`(1) <optim>`.

        :param torch.Tensor y: measurement vector.
        :param deepinv.physics.Physics physics: physics of the problem for the acquisition of ``y``.
        :param torch.Tensor x_init: (optional) required if ``physics`` does not belong to ``deepinv.physics.Physics`` in order to access to image size
        """
        with torch.no_grad():
            if seed:
                np.random.seed(seed)
                torch.manual_seed(seed)

            if x_init is None:
                x_init = physics.A_adjoint(y)
            x = x_init
            delta = 1 / self.max_iter

            for it in tqdm(range(self.max_iter), disable=(not self.verbose)):
                t = torch.ones(len(x), device=self.device) * delta * it
                lr_t = self.lr * (1 - t.view(-1, 1, 1, 1)) ** self.lr_exp
                z = x - lr_t * self.data_fidelity.grad(x, y, physics)
                x_new = torch.zeros_like(x)
                for _ in range(self.n_avg):
                    z_tilde = self.interpolation_step(z, t.view(-1, 1, 1, 1))
                    x_new += self.denoiser(z_tilde, t)
                x_new /= self.n_avg
                x = x_new
        return x
