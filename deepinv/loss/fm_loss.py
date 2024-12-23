import torch

from deepinv.loss.loss import Loss
from deepinv.physics import GaussianNoise, Denoising
from deepinv.loss.metric import MSE


class FlowMatchingModel(torch.nn.Module):
    r"""
    `Flow Macthing loss <https://arxiv.org/pdf/2209.03003>`_

    Flow Matching is a method to solve the following ODE:
    .. math:

        dZt = v(Zt,t)dt

    where :math:`Z_0 = x` is an image of :math:`n` pixels,
    :math:`Z_1 = y` is the measurements of same shape :math:`n` pixels,
    :math:`v(Z_t,t)` is a vector field estimated by our RF model.

    This formulation is a simpler form of Flow matching...

    TODO: complete this and missing example of usage
    """

    def __init__(self, model):
        super().__init__()
        self.nn_model = model

    def forward(self, y, physics, x_gt=None, **kwargs):
        r"""
        Generate the forward process at timestep t for the Rectified Flow model and get output of the model.

        :param torch.Tensor x_gt: Target (ground-truth) image.
        :param torch.Tensor y: Noisy measurements derived from `x_gt`, not used in the method.
        :return: (torch.Tensor) Output of `nn_model`.
        """
        ### COMPUTE INTERMEDIATE PHYSICS
        gaussian_noise = GaussianNoise(sigma=0.01)

        params = physics.__dict__
        physics_clean = type(physics)(**params)
        physics_clean.noise_model = gaussian_noise
        # timestep t between 0 and 1
        # if t == 0 then z_t == x_gt
        # if t == 0 then z_t == y
        # t.shape == (b, 1, 1, 1) if x_gt.shape == (b, c, h, w)
        t = torch.rand((x_gt.size(0),) + (1,) * (x_gt.dim() - 1), device=x_gt.device)
        # for each timestep t, we have an "intermediate physics" that is used as conditioning
        # to guide the flow model, it is also the physics that generate our training samples
        t_diffusion_physics = (1 - t) * physics_clean + t * physics

        ### COMPUTE "INTERMEDIATE REPRESENTATION" FOR FLOW MODEL
        z_t = t_diffusion_physics.symmetric(x_gt)

        # For reconstruction network
        x_net = self.nn_model(x_in=z_t, physics=physics, physics_t=t_diffusion_physics, y=y)

        # # Classical loss (for comparison)
        # x_net = self.nn_model(x_in=y, physics=physics)

        return x_net

    @classmethod
    @torch.no_grad()
    def sample(cls, model, y, physics, sample_steps=50, cfg=2.0):
        r"""
        ODE solver for the RF model.

        Rectified Flow is a met
        .. math::

            y = S (h*x)

        TODO:

        :param torch.Tensor y: Batch of measurements with shape ==(b, ...)
        """
        b = y.size(0)

        # dt.shape == (b,1,1,1) if y.shape == (b,c,h,w)
        dt = 1.0 / sample_steps
        dt = torch.full((b, *[1] * len(y.shape[1:])), dt, device=y.device)

        z_t = physics.A_adjoint(y)
        images = [z_t]
        for i in range(sample_steps, 0, -1):
            # t.shape == (b,1,1,1) if y.shape == (b,c,h,w)
            curr_step = float(i / sample_steps)
            t = torch.full((b, *[1] * len(y.shape[1:])), curr_step, device=y.device)

            # compute intermediate physics required by the rf model
            gaussian_noise = GaussianNoise(sigma=0.01)
            physics_clean = Denoising(noise_model=gaussian_noise, device=y.device)
            t_diffusion_physics = (1 - t) * physics_clean + t * physics

            # Initial
            # estimation of x
            # x_hat = model(y=y, physics=t_diffusion_physics)
            # estimation of the velocity from y to x
            # vc = (x_hat - y) / t

            # Proposed (not very elegant)
            # xt = t_diffusion_physics.symmetric(x)
            x_hat = model(x_in=z_t, y=y, physics=t_diffusion_physics, physics_t=t_diffusion_physics)
            vc = (x_hat - z_t) / t

            z_t = z_t + dt * vc
            images.append(z_t)

        return images


class FMLoss(Loss):
    r"""
    Standard `Flow Matching loss <https://github.com/cloneofsimo/minRF>`_

    TODO !!!
    """

    def __init__(self, metric=MSE()):
        super().__init__()
        self.name = "Rectified Flow loss"
        self.metric = metric

    def forward(self, x_net, x, y, **kwargs):
        r"""
        Computes the loss.

        TODO: explain why we are doing not (y - x - x_net) ** 2

        :param torch.Tensor x_net: Output of the model.
        :param torch.Tensor x: Target (ground-truth) image.
        :param torch.Tensor y: Noisy measurements from x.
        :return: (torch.Tensor) Loss per data sample.
        """
        return self.metric(x_net, x)

    def adapt_model(self, model):
        if not isinstance(model, FlowMatchingModel):
            return FlowMatchingModel(model)
        return model
