import torch

from deepinv.loss.loss import Loss
from deepinv.physics import GaussianNoise, Denoising
from deepinv.loss.metric import MSE


class PhysicsFlowMatchingModel(torch.nn.Module):
    r"""
    `Flow Matching <https://arxiv.org/pdf/2210.02747>`_ with conditioning on physics of inverse image problems.

    In Flow Matching, we modelize a probability flow from y to x with the following ODE:
    .. math:
        dZ_t = v(Z_t,t)dt
    where :math:`Z_0 = x`
          :math:`Z_1 = y`
          :math:`v(Z_t,t)` is a velocity field estimated by our neural network model.

    The method works in 2 parts:
        - a training part where our neural network learns an estimation of `v(Z_t,t)`
        - a sampling part which solves the ODE defined above

    Specifically in case of inverse image problems:
        - we introduce a conditioning on the physics
        - x represents the clean image
        - y represents the noisy measurement which can be computed from x

    :param torch.nn.Module nn_model: Neural network model that estimates the velocity field `v(Z_t,t)`.
    """

    def __init__(self, model):
        super().__init__()
        self.nn_model = model

    @classmethod
    @torch.no_grad()
    def compute_training_samples(cls, x_gt, physics):
        r"""
        Prepare training samples for Physics-conditioned Flow Matching.

        :math:`Z_t = (1-t)x_{gt} + ty`
        where :math:`y = physics(x_{gt})`.

        :param torch.Tensor x_gt: Target (ground-truth) image.
        :param deepinv.physics.Physics physics: Physics model.
        :return: ???
        """
        # for each image in x_gt, uniformely sample a timestep t in [0,1]
        # t.shape == (b, 1, 1, 1) if x_gt.shape == (b, c, h, w)
        t = torch.rand((x_gt.size(0),) + (1,) * (x_gt.dim() - 1), device=x_gt.device)


        # TODO: ideally fix below; for the moment we do this manually
        # for each timestep t, we have an "intermediate physics" that is used as conditioning
        # to guide the flow model, it is also the physics that generate our training samples
        ### COMPUTE INTERMEDIATE PHYSICS
        # params = physics.__dict__
        # physics_clean = type(physics)(**params)  # ???
        # physics_clean.noise_model = GaussianNoise(sigma=0.001)  # ???
        # t_diffusion_physics = (1 - t) * physics_clean + t * physics
        # z_t = t_diffusion_physics.symmetric(x_gt)  # ???

        # Manual version
        z_t = (1-t)*x_gt + t*physics.A_adjoint(physics(x_gt))

        return z_t, t

    def forward(self, y, physics, x_gt=None, **kwargs):
        r"""
        Define the forward pass during the training of Flow model.

        This method prepare the training samples before passing to `nn_model.forward`.

        :param torch.Tensor y: Noisy measurements derived from `x_gt`, not used in the method.
        :param deepinv.physics.Physics physics: Physics model.
        :param torch.Tensor x_gt: Ground-truth in the image domain.
        :return: (torch.Tensor) Output of `nn_model`.
        """
        z_t, t = self.__class__.compute_training_samples(x_gt, physics)

        # For reconstruction network
        x_net = self.nn_model(x_in=z_t, physics=physics, t=t, y=y)

        # # Classical loss (for comparison)
        # x_net = self.nn_model(x_in=y, physics=physics)

        return x_net

    @classmethod
    @torch.no_grad()
    def sample(cls, model, y, physics, sample_steps=50, cfg=2.0):
        r"""
        ODE solver for the FM model.
        TODO ???

        :param torch.Tensor y: Batch of measurements with shape ==(b, ...)
        """
        b = y.size(0)

        # dt.shape == (b,1,1,1) if y.shape == (b,c,h,w)
        dt = 1.0 / sample_steps
        dt = torch.full((b, *[1] * len(y.shape[1:])), dt, device=y.device)

        z_t = physics.A_adjoint(y)  # measurements -> image domain
        images = [z_t]
        for i in range(sample_steps, 0, -1):
            # t.shape == (b,1,1,1) if y.shape == (b,c,h,w)
            curr_step = float(i / sample_steps)
            t = torch.full((b, *[1] * len(y.shape[1:])), curr_step, device=y.device)

            # Proposed (not very elegant)
            # xt = t_diffusion_physics.symmetric(x)
            x_hat = model(x_in=z_t, y=y, physics=physics, t=t)
            vc = (x_hat - z_t) / t

            z_t = z_t + dt * vc
            images.append(z_t)

        return images


class FMLoss(Loss):
    r"""
    Slightly modified`Flow Matching loss <https://github.com/cloneofsimo/minRF>`_

    :Examples:

        Using this class to train a Flow Matching model:

        TODO !!!

    TODO !!!
    """

    def __init__(self, metric=MSE()):
        super().__init__()
        self.name = "Flow Matching loss"
        self.loss_fn = metric

    def forward(self, x_net, x, **kwargs):
        r"""
        Computes the loss.

        TODO: explain why we are doing not (y - x - x_net) ** 2

        :param torch.Tensor x_net: Output of the model.
        :param torch.Tensor x_gt: Ground-truth in the image domain.
        :return: (torch.Tensor) Loss per data sample.
        """
        return self.loss_fn(x_net, x)

    def adapt_model(self, model):
        if not isinstance(model, PhysicsFlowMatchingModel):
            return PhysicsFlowMatchingModel(model)
        return model
