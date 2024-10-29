import torch

from deepinv.loss.loss import Loss
from deepinv.physics import GaussianNoise, Denoising


class FlowTrainerModel(torch.nn.Module):
    r"""
    Standard Flow Model:

    TODO !!!
    """

    def __init__(self, model):
        super().__init__()
        self.nn_model = model

    def forward(self, y, physics, x_gt, **kwargs):
        r"""
        Generate the forward process at timestep t for the Flow model and get output of the model.

        :param torch.Tensor x_gt: Target (ground-truth) image.
        :param torch.Tensor y: Noisy measurements derived from `x_gt`, not used in the method.
        :return: (torch.Tensor) Output of `nn_model`.
        """
        ### COMPUTE INTERMEDIATE PHYSICS
        gaussian_noise = GaussianNoise(sigma=0.1)
        physics_clean = Denoising(noise_model=gaussian_noise, device=x_gt.device)
        # timestep t between 0. and 1. (t=0 -> z_t=x, t=1 -> z_t=y)
        # shape (b, 1, 1, 1) if x_gt.shape == (b, c, h, w)
        t = torch.rand((x_gt.size(0),) + (1,) * (x_gt.dim() - 1), device=x_gt.device)
        # for each timestep t, we have an "intermediate physics" that is used as conditioning
        # to guide the flow model, it is also the physics that generate our training samples
        t_diffusion_physics = (1 - t) * physics_clean + t * physics

        ### COMPUTE "INTERMEDIATE REPRESENTATION" FOR FLOW MODEL
        z_t = t_diffusion_physics(x_gt)
        # z_t = (1-t)*x + t*y

        # For reconstruction network
        x_net = self.nn_model(y=z_t, physics=t_diffusion_physics)

        return x_net


class RFLoss(Loss):
    r"""
    `Standard Rectified Flow loss <https://github.com/cloneofsimo/minRF>`_

    TODO !!!
    """

    def __init__(self):
        super().__init__()
        self.name = "Rectified Flow loss"

    def forward(self, x_net, x, y, **kwargs):
        r"""
        Computes the loss.

        TODO: explain why we are doing not (y - x - x_net) ** 2

        :param torch.Tensor x_net: Output of the model.
        :param torch.Tensor x: Target (ground-truth) image.
        :param torch.Tensor y: Noisy measurements from x.
        :return: (torch.Tensor) Loss per data sample.
        """
        # shape : (b,) with b, the batch_size
        batchwise_mse = ((x - x_net) ** 2).mean(dim=list(range(1, len(x.shape))))
        return batchwise_mse

    def adapt_model(self, model):
        return FlowTrainerModel(model)
