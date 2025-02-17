import torch
from deepinv.optim.data_fidelity import L2

# This file implements the p(y|x) terms as proposed in the `review paper <https://arxiv.org/pdf/2410.00083>`_ by Daras et al.

class NoisyDataFidelity(L2):
    r"""
    Preconditioned data fidelity term for noisy data :math:`\datafid{x}{y}=\distance{\forw{x'}}{y'}`.

    This is a base class for the conditional classes for approximating :math:`\log p(y|x)` used in diffusion
    algorithms for inverse problems. Here, :math:`x'` and :math:`y'` are perturbed versions of :math:`x` and :math:`y`
    and the associated data fidelity term is :math:`\datafid{x}{y}=\distance{\forw{x'}}{y'}`.

    It comes with a `.grad` method computing the score

    .. math::

        \begin{equation*}
            \nabla_x \log p(y|x) = P(\forw{x'}-y'),
        \end{equation*}


    where :math:`P` is a preconditioner. By default, :math:`P` is defined as :math:`A^\top` and this class matches the
    :class:`deepinv.optim.DataFidelity` class.
    """

    def __init__(self):
        super(NoisyDataFidelity, self).__init__()

    def precond(self, u: torch.Tensor, physics) -> torch.Tensor:
        r"""
        The preconditioner :math:`P = A^\top` for the data fidelity term.

        :param torch.Tensor u: input tensor.
        :param deepinv.physics.Physics physics: physics model.
        :return: (torch.FloatTensor) preconditionned tensor :math:`P(u)`.
        """
        return physics.A_adjoint(u)

    def diff(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        Computes the difference between the forward operator applied to the current iterate and the input data.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.

        :return: (torch.Tensor) difference between the forward operator applied to the current iterate and the input data.
        """
        return physics.A(x) - y

    def grad(self, x: torch.Tensor, y: torch.Tensor, physics, sigma) -> torch.Tensor:
        r"""
        Computes the gradient of the data-fidelity term.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param physics: physics model
        :param float sigma: Standard deviation of the noise.
        :return: (torch.Tensor) data-fidelity term.
        """
        return self.precond(self.diff(x, y, physics, sigma))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, **kwargs
    ) -> torch.Tensor:
        r"""
        Computes the data-fidelity term.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :return: (torch.Tensor) loss term.
        """
        return self.d(physics.A(x), y)


class DPSDataFidelity(NoisyDataFidelity):
    r"""
    The DPS data-fidelity term.

    This corresponds to the :math:`p(y|x)` prior as proposed in `Diffusion Probabilistic Models <https://arxiv.org/abs/2209.14687>`_.

    :param denoiser: Denoiser network. # TODO: type?
    """

    def __init__(self, denoiser=None):
        super(DPSDataFidelity, self).__init__()

        self.denoiser = denoiser

    def precond(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, *args, **kwargs
    ) -> torch.Tensor:
        r"""
        As explained in `Daras et al. <https://arxiv.org/abs/2410.00083>`_, the score is defined as

        .. math::

                \nabla_x \log p(y|x) = \left(\operatorname{Id}+\nabla_x D(x)\right)^\top A^\top \left(y-\forw{D(x)}\right)

        .. note::
            The preconditioning term is computed with autodiff.

        :param torch.Tensor x: Current iterate.
        :param torch.Tensor y: Input data.
        :param physics: physics model
        :param float sigma: Standard deviation of the noise. (unused)
        :return: (torch.Tensor) score term.
        """
        with torch.enable_grad():
            x.requires_grad_(True)
            l2_loss = self.forward(x, y, physics, sigma, *args, **kwargs)

        norm_grad = torch.autograd.grad(outputs=l2_loss, inputs=x)[0]
        norm_grad = norm_grad.detach()

        return norm_grad

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, physics, sigma, clip=False
    ) -> torch.Tensor:
        r"""
        Returns the loss term :math:`\distance{\forw{D(x)}}{y}`.

        :param torch.Tensor x: input image
        :param torch.Tensor y: measurements
        :param deepinv.physics.Physics physics: forward operator
        :param float sigma: standard deviation of the noise.
        :param bool clip: whether to clip the output of the denoiser to the range [-1, 1].
        :return: (torch.Tensor) loss term.
        """
        # TODO: check that the following does not belong to the grad method but to the denoiser method itself.
        # aux_x = x / 2 + 0.5
        # x0_t = 2 * self.denoiser(x, sigma / 2) - 1

        x0_t = self.denoiser(x, sigma)

        if clip:
            x0_t = torch.clip(x0_t, 0.0, 1.0)  # optional

        l2_loss = self.d(physics.A(x0_t), y)

        return l2_loss
