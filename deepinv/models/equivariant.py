from __future__ import annotations

from torch import Tensor
from deepinv.transform import Transform, Rotate, Reflect
from .base import Denoiser, Reconstructor
from deepinv.physics.virtual import VirtualPhysics
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

    See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_transforms.py` for an example.

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
    Turns the reconstructor model into an equivariant reconstructor with respect to geometric transforms.

    Recall that a reconstructor is equivariant with respect to a group of transformations if it commutes with the action of
    the group. More precisely, let :math:`\mathcal{G}` be a group of transformations :math:`\{T_g\}_{g\in \mathcal{G}}`
    and :math:`\inversename` a reconstruction model. Then, :math:`\inversename` is equivariant with respect to :math:`\mathcal{G}`
    if :math:`\inversef{y,AT_g} = T_g\inversef{y,A}` for any measurement :math:`y` and any :math:`g\in \mathcal{G}`.

    The reconstruction model can be turned into an equivariant denoiser by averaging over the group of transforms, i.e.

    .. math::
        \operatorname{R}^{\text{eq}}(y,A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g(\inversef{y}{AT_g}).

    Otherwise, as proposed in https://arxiv.org/abs/2312.01831, a Monte Carlo approximation can be obtained by
    sampling :math:`g \sim \mathcal{G}` at random and applying

    .. math::
        \operatorname{R}^{\text{MC}}(y,A) = T_g(\inversef{y}{AT_g}).

    .. note::

        We have implemented many popular geometric transforms, see :ref:`docs <transform>`. You can set the number of Monte Carlo samples by passing ``n_trans``
        into the transforms, for example ``Rotate(n_trans=2)`` will average over 2 samples per call. For rotate and reflect, by setting ``n_trans``
        to the maximum (e.g. 4 for 90 degree rotations, 2 for 1D reflections), it will average over the whole group, for example:

        ``Rotate(n_trans=4, multiples=90, positive=True) * Reflect(n_trans=2, dims=[-1])``

    See :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_transforms.py` for an example.

    :param Callable model: Reconstruction model :math:`\inversef{y}{A}`.
    :param Transform transform: geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
        See :ref:`docs <transform>` for list of available transforms.
    """

    def __init__(
        self,
        model: Reconstructor,
        transform: Transform,
        eval_transform: Transform | None = None,
    ):
        super().__init__()
        self.model = model

        if eval_transform is None:
            eval_transform = transform

        self.transform = transform
        self.eval_transform = eval_transform

    def forward(self, y, physics, *reconstructor_args, **reconstructor_kwargs):
        r"""
        Symmetrize the reconstructor by the transformation to create an equivariant reconstructor and apply to input.

        The symmetrization collects the average if multiple samples are used (controlled with ``n_trans`` in the transform).

        :param torch.Tensor x: input image.
        :param \*denoiser_args: args for denoiser function e.g. sigma noise level.
        :param \**denoiser_kwargs: kwargs for denoiser function e.g. sigma noise level.
        :return: denoised image.
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

        # Compute the terms in the sum, i.e., T_g(f(y, AT_g)) for each g
        terms = []
        for g_params in transform.iterate_params(G_params):
            Tg = lambda x: transform.transform(x, **g_params)
            Tg_inv = lambda x: transform.inverse(x, **g_params)

            ATg = VirtualPhysics(physics=physics, T=Tg, T_inv=Tg_inv)

            fyATg = self.model(
                y, ATg, *reconstructor_args, **reconstructor_kwargs
            )  # f(y, AT_g)
            TgfyATg = Tg(fyATg)  # T_g(f(y, AT_g))
            terms.append(TgfyATg)

        # Average over the group elements
        terms = torch.stack(terms, dim=1)  # (B, G, C, H, W)
        return terms.mean(dim=1)  # (B, C, H, W)
