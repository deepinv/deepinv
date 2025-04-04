import torch
from deepinv.optim import DataFidelity, Distance
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser


class NoisyDataFidelity(DataFidelity):
    r"""
    Preconditioned data fidelity term for noisy data :math:`- \log p(y|x + \sigma(t) \omega)`
    with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This is a base class for the conditional classes for approximating :math:`\log p_t(y|x_t)` used in diffusion
    algorithms for inverse problems, in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method computing the score :math:`\nabla_{x_t} \log p_t(y|x_t)`.

    By default we have

    .. math::

        \begin{equation*}
            \nabla_{x_t} \log p(y|x + \sigma(t) \omega) = P(\forw{x_t'}-y),
        \end{equation*}


    where :math:`P` is a preconditioner and :math:`x_t'` is an estimation of the image :math:`x`.
    By default, :math:`P` is defined as :math:`A^\top`, :math:`x_t' = x_t` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.
    """

    def __init__(self, d: Distance = None, *args, **kwargs):
        super().__init__()
        if d is not None:
            self.d = Distance(d)
        else:
            self.d = dinv.optim.L2Distance()

    def precond(
        self, u: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        The preconditioner :math:`P` for the data fidelity term. Default to :math:`Id`

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.

        :return: (torch.Tensor) preconditionned tensor :math:`P(u)`.
        """
        return physics.A_dagger(u)

    def diff(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the difference :math:`A(x) - y` between the forward operator applied to the current iterate and the input data.


        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics), physics=physics)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y)


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    The DPS data-fidelity term.

    This corresponds to the :math:`p(y|x)` prior as proposed in `Diffusion Probabilistic Models <https://arxiv.org/abs/2209.14687>`_.

    :param deepinv.models.Denoiser denoiser: Denoiser network.

    .. math::
            \begin{aligned}
            \nabla_x \log p_t(y|x) &= \nabla_x \frac{1}{2} \| \forw{\denoisername{x}} - y \|^2 \\
                                 &= \left(\nabla_x \denoisername{x} \right)^\top A^\top \left(\forw{\denoisername{x}} - y\right)
            \end{aligned}

    where :math:`\sigma = \sigma(t)` is the noise level. 

    .. note::
        The preconditioning term is computed with automatic differentiation.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param bool clip: Whether to clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`. 
    """

    def __init__(self, denoiser: Denoiser = None, clip: tuple = None, *args, **kwargs):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        if clip is not None:
            assert len(clip) == 2
            clip = sorted(clip)
        self.clip = clip

    def precond(
        self, x: torch.Tensor, physics: Physics, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, sigma, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param deepinv.physics.Physics physics: physics model
        :param float sigma: Standard deviation of the noise.
        :return: (:class:`torch.Tensor`) score term.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            l2_loss = self.forward(x, y, physics, sigma, *args, **kwargs)
        grad_outputs = torch.ones_like(l2_loss)
        norm_grad = torch.autograd.grad(
            outputs=l2_loss, inputs=x, grad_outputs=grad_outputs
        )[0]
        return norm_grad

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics: Physics, sigma, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        Returns the loss term :math:`\distance{\forw{\denoiser{\sigma}{x}}}{y}`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :return: (torch.Tensor) loss term.
        """

        x0_t = self.denoiser(x, sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional
        l2_loss = self.d(physics.A(x0_t), y)
        return l2_loss
