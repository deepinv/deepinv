import torch
from deepinv.optim import Distance, Potential
import deepinv as dinv
from deepinv.physics import Physics
from deepinv.models import Denoiser
from typing import Union

class NoisyDataFidelity(Potential):
    r"""
    Base class for implementing the noisy data fidelity term :math:`- \log p(y|x_\sigma = x + \sigma \omega)` with :math:`\omega\sim\mathcal{N}(0,\mathrm{I})`.

    This class is used to approximate the untractable likelihood term :math:`\log p_t(y|x_t)` in diffusion-based
    algorithms for inverse problems, as defined in :class:`deepinv.sampling.PosteriorDiffusion`.

    It comes with a `.grad` method for computing an approximation of the score :math:`\nabla_{x_\sigma} \log p_t(y|x_\sigma)`.

    You can either define the gradient :math:`\nabla_{x_\sigma} \log p_t(y|x_\sigma)` by overwriting the :meth:`grad` method,
    or you can define the noisy data fidelity function :math:`- \log p(y|x_\sigma)` by overwriting the :meth:`fn` method.
    In this case, the gradient will be computed automatically using automatic differentiation.

    By default the class matches the :class:`deepinv.optim.DataFidelity` class, i.e. we make the approximation

    .. math::

        \begin{equation*}
            -\log p(y| x_\sigma) \approx d(A x, y)
        \end{equation*}

    By default, the distance function :math:`d` is the L2 distance.
    """

    def __init__(self, fn = None, *args, **kwargs):
        super().__init__(fn = fn)

    def fn(self, x, y, physics, sigma=None, *args, **kwargs):
        r"""
        Calculates the noisy data fidelity term approximating :math:`- \log p(y|x_\sigma)`

        :param torch.Tensor x: Variable :math:`x_\sigma` at which the data fidelity term is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param Union[torch.Tensor, float] sigma: Standard deviation :math:`\sigma` of the noise in :math:`x_\sigma`.
        :return: (:class:`torch.Tensor`) data fidelity term.
        """
        return super().fn(x, y, physics, sigma, *args, **kwargs)

    def grad(self, x, y, physics, sigma=None, *args, **kwargs):
        r"""
        Calculates the gradient of the noisy data fidelity term, approximating :math:`\nabla_{x_\sigma} \log p_t(y|x_\sigma)`.

        :param torch.Tensor x: Variable :math:`x_\sigma` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :param Union[torch.Tensor, float] sigma: Standard deviation :math:`\sigma` of the noise in :math:`x_\sigma`.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return super().grad(x, y, physics, sigma, *args, **kwargs)


class DataFidelity(NoisyDataFidelity):
    
    def __init__(self, d=None):
        super().__init__()
        if not isinstance(d, Distance):
            self.d = Distance(d=d)
    
    def fn(self, x, y, physics, *args, **kwargs):
        r"""
        Computes the data fidelity term :math:`\datafid{x}{y} = \distance{\forw{x}}{y}`.

        :param torch.Tensor x: Variable :math:`x` at which the data fidelity is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) data fidelity :math:`\datafid{x}{y}`.
        """
        return self.d(physics.A(x), y, *args, **kwargs)

    def grad(self, x, y, physics, *args, **kwargs):
        r"""
        Calculates the gradient of the data fidelity term :math:`\datafidname` at :math:`x`.

        The gradient is computed using the chain rule:

        .. math::

            \nabla_x \distance{\forw{x}}{y} = \left. \frac{\partial A}{\partial x} \right|_x^\top \nabla_u \distance{u}{y},

        where :math:`\left. \frac{\partial A}{\partial x} \right|_x` is the Jacobian of :math:`A` at :math:`x`, and :math:`\nabla_u \distance{u}{y}` is computed using ``grad_d`` with :math:`u = \forw{x}`. The multiplication is computed using the ``A_vjp`` method of the physics.

        :param torch.Tensor x: Variable :math:`x` at which the gradient is computed.
        :param torch.Tensor y: Data :math:`y`.
        :param deepinv.physics.Physics physics: physics model.
        :return: (:class:`torch.Tensor`) gradient :math:`\nabla_x \datafid{x}{y}`, computed in :math:`x`.
        """
        return physics.A_vjp(x, self.d.grad(physics.A(x), y, *args, **kwargs))

class L2(DataFidelity):
    """
    Noisy data fidelity term using the L2 distance:

     .. math::

        \begin{equation*}
            -\log p(y| x_\sigma) \approx \frac{1}{2\sigma_y^2}\|\forw{x_\sigma}-y\|^2
        \end{equation*}

    It corresponds to the approximation proposed by the Score ALD method (`Robust compressed sensing mri with deep generative priors <https://proceedings.neurips.cc/paper_files/paper/2021/hash/7d6044e95a16761171b130dcb476a43e-Abstract.html>`_).

    where :math:`\sigma_y` is the standard deviation of the noise in :math:`y`.
    """

    def __init__(self, weight=1.0, sigma_y=1.0, *args, **kwargs):
        self.d = dinv.optim.L2Distance(sigma=sigma_y)
        super().__init__(d=self.d, weight=weight, *args, **kwargs)


class ScoreSDE(NoisyDataFidelity):

    def __init__(self, weight=1.0, sigma_y=1.0, *args, **kwargs):
        super().__init__(weight=weight, sigma_y=sigma_y, *args, **kwargs)

    def grad(self, x, y, physics, sigma=None, *args, **kwargs):
        y_noisy = y + sigma * torch.randn_like(x)
        return self.weight * physics.A_vjp(
            x, self.d.grad(physics.A(x), y_noisy, *args, **kwargs)
        )


class ILVR(NoisyDataFidelity):

    def __init__(self, weight=1.0, sigma_y=1.0, *args, **kwargs):
        super().__init__(weight=weight, sigma_y=sigma_y, *args, **kwargs)

    def grad(self, x, y, physics, sigma=None, *args, **kwargs):
        y_noisy = y + sigma * torch.randn_like(x)
        return self.weight * physics.A_dagger(
            x, self.d.grad(physics.A(x), y_noisy, *args, **kwargs)
        )


class DPSDataFidelity(NoisyDataFidelity):

    def __init__(
        self, weight=1.0, denoiser=lambda x, sigma: x, sigma_y=1.0, *args, **kwargs
    ):
        super().__init__(weight=weight, sigma_y=sigma_y, *args, **kwargs)
        self.denoiser = denoiser

    def fn(self, x, y, physics, sigma=None, *args, **kwargs):
        return self.weight * self.d(
            physics.A(self.denoiser(x, sigma)), y, *args, **kwargs
        )

    def grad(self, x, y, physics, sigma=None, *args, **kwargs):
        return self.weight * physics.A_vjp(
            x, self.d.grad(physics.A(x), y, *args, **kwargs)
        )


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    Diffusion posterior sampling data-fidelity term.

    Using the fact that

    .. math::

        p(y | x_\sigma) = \int p(y|x) p(x | x_\sigma)dx,

    This approximation comes back to approximating the denoising posterior by a dirac i.e:

    .. math::

        p(x | x_\sigma) \approx \delta(x - D_\sigma(x_\sigma))`

    where :math:`\delta` is the Dirac delta function.

    This corresponds to the :math:`p(y|x_\sigma)` approximation proposed in `Diffusion Posterior Sampling for General Noisy Inverse Problems <https://arxiv.org/abs/2209.14687>`_.

    .. math::
            \begin{aligned}
            \nabla_x \log p_t(y|x) &= \nabla_x \frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|
            \end{aligned}

    where :math:`\sigma = \sigma(t)` is the noise level, :math:`m` is the number of measurements (size of :math:`y`),
    and :math:`\lambda` controls the strength of the approximation.

    .. seealso::
        This class can be used for building custom DPS-based diffusion models.
        A self-contained implementation of the original DPS algorithm can be find in :class:`deepinv.sampling.DPS`.

    :param deepinv.models.Denoiser denoiser: Denoiser network
    :param float weight: Weighting factor for the data fidelity term. Default to 100.
    :param tuple[float] clip: If not `None`, clip the denoised output into `[clip[0], clip[1]]` interval. Default to `None`.
    """

    def __init__(
        self,
        denoiser: Denoiser = None,
        weight=1.0,
        clip: tuple = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.d = dinv.optim.L2Distance()
        self.denoiser = denoiser
        if clip is not None:
            assert len(clip) == 2
            clip = sorted(clip)
        self.clip = clip
        self.weight = weight

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
        Returns the loss term :math:`\frac{\lambda}{2\sqrt{m}} \| \forw{\denoiser{x}{\sigma}} - y \|`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :return: (torch.Tensor) loss term.
        """

        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(torch.float32)

        x0_t = self.denoiser(x.to(torch.float32), sigma, *args, **kwargs)

        if self.clip is not None:
            x0_t = torch.clip(x0_t, self.clip[0], self.clip[1])  # optional
        return (self.d(physics.A(x0_t), y) * y.numel() / y.size(0)).sqrt() * self.weight
