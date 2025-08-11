from __future__ import annotations
from typing import TYPE_CHECKING

from torch import Tensor
import torch

from deepinv.physics.mri import MRI, MRIMixin
from deepinv.physics.inpainting import Inpainting
from deepinv.loss.sure import SureGaussianLoss, mc_div

if TYPE_CHECKING:
    from deepinv.physics.generator.base import PhysicsGenerator
    from deepinv.models.base import Reconstructor
    from deepinv.physics.forward import Physics


class ENSURELoss(SureGaussianLoss):
    r"""
    ENSURE loss for image reconstruction in Gaussian noise.

    The loss function is a special case of :class:`deepinv.loss.SureGaussianLoss` for MRI/inpainting with varying masks, and is designed for the following noise model:

    .. math::

        y \sim\mathcal{N}(u,\sigma^2 I) \quad \text{with}\quad u= A_i(x).

    where :math:`A_i\sim\mathcal{A}` is assumed to be drawn from a set of measurement operators.
    The loss is computed as

    .. math::

        \frac{1}{m}\|\Beta(A^{\dagger}y - \inverse{y})\|_2^2 +\frac{2\sigma^2}{m\tau}b^{\top} \left(\inverse{A^{\top}y+\tau b_i} -
        \inverse{A^{\top}y}\right)

    where :math:`R` is the trainable network (which takes :math:`A^\top y` as input),
    :math:`A` is the forward operator,
    :math:`y` is the noisy measurement vector of size :math:`m`,
    :math:`b\sim\mathcal{N}(0,I)`, :math:`\tau\geq 0` is a hyperparameter controlling the
    Monte Carlo approximation of the divergence, and :math:`\Beta=W^{-1}P`
    where :math:`P` is the projection operator onto the range space of :math:`\A^\top`
    and :math:`W` is a weighting determined by the set of measurement operators where :math:`W^2=\mathbb{E}\left[P\right]`.

    The ENSURE loss was proposed in :footcite:t:`aggarwal2023ensure` for MRI.

    .. warning::

        This loss was originally proposed only to be used with :class:`artifact removal models <deepinv.models.ArtifactRemoval>` which can be written in the form :math:`\inverse{\cdot}=r(A^\top\cdot)`.
        If an artifact removal model is not used, then we evaluate the network directly instead.

        We currently only provide an implementation for :class:`single-coil MRI <deepinv.physics.MRI>` and :class:`inpainting <deepinv.physics.Inpainting>`,
        where `A^\top=A^\dagger` such that :math:`P=A^{\top}A` and then :math:`W` is a weighted average over sampling masks.

    :param float sigma: Standard deviation of the Gaussian noise.
    :param deepinv.physics.generator.PhysicsGenerator physics_generator: random physics generator used to compute the weighting :math:`W`.
    :param float tau: Approximation constant for the Monte Carlo approximation of the divergence. Defaults to :math:`0.1\sigma`.
    :param torch.Generator rng: Optional random number generator. Default is None.
    """

    def __init__(
        self,
        sigma: float,
        physics_generator: PhysicsGenerator,
        tau: float = None,
        rng: torch.Generator = None,
    ):
        super().__init__(
            sigma=sigma, tau=tau if tau is not None else sigma * 0.1, rng=rng
        )
        d = physics_generator.average()["mask"]
        self.dsqrti = 1 / d.sqrt()

    def div(
        self, x_net: Tensor, y: Tensor, f: Reconstructor, physics: Physics
    ) -> Tensor:
        r"""
        Monte-Carlo estimation for the divergence of f(x).

        :param torch.Tensor x_net: Reconstructions.
        :param torch.Tensor y: Measurements.
        :param deepinv.physics.Physics physics: Forward operator associated with the measurements.
        :param torch.nn.Module f: Reconstruction network.
        """
        return mc_div(physics.A(x_net), y, f, physics, tau=self.tau, rng=self.rng)

    def forward(
        self, y: Tensor, x_net: Tensor, physics: Physics, model: Reconstructor, **kwargs
    ) -> Tensor:
        if isinstance(physics, MRI):
            metric = lambda y: MRIMixin().kspace_to_im(y * self.dsqrti)
        elif isinstance(physics, Inpainting):
            metric = lambda y: y * self.dsqrti
        else:
            raise ValueError(
                "ENSURE loss is currently only implemented for single-coil MRI or inpainting."
            )

        y1 = physics.A(x_net)
        div = 2 * self.sigma2 * self.div(x_net, y, model, physics)
        mse = metric(y1 - y).pow(2).reshape(y.size(0), -1).mean(1)
        return mse + div - self.sigma2
