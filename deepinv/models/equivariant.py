from __future__ import annotations

from torch import Tensor
from deepinv.transform import Transform, Rotate, Reflect
from .base import Denoiser, Reconstructor
from deepinv.physics.virtual import VirtualLinearPhysics
import torch


class EquivariantDenoiser(Denoiser):
    r"""
    Turns the input denoiser into an equivariant denoiser with respect to geometric transforms.

    Recall that a denoiser is equivariant with respect to a group of transformations if it commutes with the action of
    the group. More precisely, let :math:`\mathcal{G}` be a group of transformations :math:`\{T_g\}_{g\in \mathcal{G}}`
    and :math:`\denoisername` a denoiser. Then, :math:`\denoisername` is equivariant with respect to :math:`\mathcal{G}`
    if :math:`\denoisername(T_g(x)) = T_g(\denoisername(x))` for any image :math:`x` and any :math:`g\in \mathcal{G}`.

    The denoiser can be turned into an equivariant denoiser by averaging over the group of transforms, i.e.

    .. math::
        \operatorname{D}^{\text{eq}}_{\sigma}(x) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

    Otherwise, as proposed by :footcite:t:`terris2024equivariant`, a Monte Carlo approximation can be obtained by
    sampling :math:`g \sim \mathcal{G}` at random and applying

    .. math::
        \operatorname{D}^{\text{MC}}_{\sigma}(x) = T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

    .. note::

        We have implemented many popular geometric transforms, see :ref:`docs <transform>`. You can set the number of Monte Carlo samples by passing ``n_trans``
        into the transforms, for example ``Rotate(n_trans=2)`` will average over 2 samples per call. For rotate and reflect, by setting ``n_trans``
        to the maximum (e.g. 4 for 90 degree rotations, 2 for 1D reflections), it will average over the whole group, for example:

        ``Rotate(n_trans=4, multiples=90, positive=True) * Reflect(n_trans=2, dims=[-1])``

    .. note::

        It is customary to sample a single transformation at training time and do a full averaging at evaluation time to ensure true equivariance. This can be done by setting a ``eval_transform`` that averages over the whole group, while leaving ``transform`` computing a single random transformation.

    See :ref:`sphx_glr_auto_examples_transforms-equivariance_demo_transforms.py` for an example.

    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param Transform transform: geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
        See :ref:`docs <transform>` for list of available transforms.
    :param Transform eval_transform: transformations to be used in evaluation mode. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. If set to `None`, evaluation transformations are the same as training transformations.
    :param bool random: if True, the denoiser is applied to a randomly transformed version of the input image
        each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
        If False, the denoiser is applied to the average of all the transformed images, turning the denoiser into an
        equivariant denoiser with respect to the chosen group of transformations. Ignored if ``transform`` is provided.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        transform: Transform | None = None,
        eval_transform: Transform | None = None,
        random: bool = True,
    ):
        super().__init__()
        self.denoiser = denoiser

        if transform is None:
            if random:
                transform = Rotate(n_trans=1, multiples=90, positive=True) * Reflect(
                    n_trans=1, dims=[-1]
                )
            else:
                transform = Rotate(n_trans=4, multiples=90, positive=True) * Reflect(
                    n_trans=2, dims=[-1]
                )

        self.transform = transform

        if eval_transform is None:
            eval_transform = transform

        self.eval_transform = eval_transform

    def forward(self, x: Tensor, *denoiser_args, **denoiser_kwargs) -> Tensor:
        r"""
        Symmetrize the denoiser by the transformation to create an equivariant denoiser and apply to input.

        The symmetrization collects the average if multiple samples are used (controlled with ``n_trans`` in the transform).

        :param torch.Tensor x: input image.
        :param \*denoiser_args: args for denoiser function e.g. sigma noise level.
        :param \**denoiser_kwargs: kwargs for denoiser function e.g. sigma noise level.
        :return: denoised image.
        """
        transform = self.transform if self.training else self.eval_transform
        return transform.symmetrize(self.denoiser, average=True)(
            x, *denoiser_args, **denoiser_kwargs
        )


class EquivariantReconstructor(Reconstructor):
    r"""
    Equivariant reconstructor

    Make a base reconstructor :math:`\tilde{R}` equivariant by averaging over the transformations, i.e.,

    .. math::

        R(y, A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g \tilde{R}(y, A T_g)

    An equivariant reconstructor is a reconstructor that satisfies :footcite:p:`sechaud26Equivariant`

    .. math::

        R(y, A T_g) = T_g^{-1} R(y, A)

    for all :math:`g \in \mathcal{G}` where :math:`T_g` is a transform (eg shifts, rotations, etc).

    :param Reconstructor model: base reconstructor to be made equivariant.
    :param Transform, None transform: geometric transformation. By default, it is set to a single random 90° rotation and flip.
    :param Transform eval_transform: transformations to be used in evaluation mode. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. By default, if training transformations are specified, evaluation transformations default to them, otherwise they default to the eight 90° rotations and flips.
    """

    def __init__(
        self,
        model: Reconstructor,
        transform: Transform | None = None,
        eval_transform: Transform | None = None,
    ):
        super().__init__()
        self.model = model

        if transform is None:
            transform = Rotate(n_trans=1, multiples=90, positive=True) * Reflect(
                n_trans=1, dims=[-1]
            )
            eval_transform = Rotate(n_trans=4, multiples=90, positive=True) * Reflect(
                n_trans=2, dims=[-1]
            )
        elif eval_transform is None:
            # NOTE: It does not do a full averaging automatically in this case.
            eval_transform = transform

        self.transform = transform
        self.eval_transform = eval_transform

    def forward(
        self, y: torch.Tensor, physics, *reconstructor_args, **reconstructor_kwargs
    ) -> torch.Tensor:
        r"""
        Apply the reconstructor to an input

        :param torch.Tensor y: input measurement.
        :param deepinv.physics.Physics physics: physics operator associated with the measurements.
        :param \*reconstructor_args: args for reconstructor function.
        :param \**reconstructor_kwargs: kwargs for reconstructor function.
        :return: (:class:`torch.Tensor`) output of the reconstructor.
        """
        # Different transforms can be used for training and evaluation to allow
        # for true Reynolds averaging at evaluation time, and Monte Carlo
        # estimation at training time.
        if self.training:
            transform = self.transform
        else:
            transform = self.eval_transform

        # NOTE: We assume that transform.get_params returns either all of the
        # group elements (true Reynolds averaging) or a single one (Monte Carlo
        # estimation).
        x0 = physics.A_adjoint(y)  # Used for inferring the group
        G_params = transform.get_params(x0)

        # Compute the terms in the sum, i.e., T_g(R(y, AT_g)) for each g
        terms = []
        for g_params in transform.iterate_params(G_params):
            ATg = VirtualLinearPhysics(
                physics=physics, transform=transform, g_params=g_params
            )

            fyATg = self.model(
                y, ATg, *reconstructor_args, **reconstructor_kwargs
            )  # R(y, AT_g)

            TgfyATg = transform.transform(fyATg, **g_params)  # T_g(R(y, AT_g))
            terms.append(TgfyATg)

        # Average over the group elements
        terms = torch.stack(terms, dim=1)  # (B, G, C, H, W)
        return terms.mean(dim=1)  # (B, C, H, W)
