from typing import Optional
import torch
from deepinv.transform import Transform, Rotate, Reflect
from .base import Denoiser


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

    Otherwise, as proposed in https://arxiv.org/abs/2312.01831, a Monte Carlo approximation can be obtained by
    sampling :math:`g \sim \mathcal{G}` at random and applying

    .. math::
        \operatorname{D}^{\text{MC}}_{\sigma}(x) = T_g^{-1}(\operatorname{D}_{\sigma}(T_g(x))).

    .. note::

        We have implemented many popular geometric transforms, see :ref:`docs <transform>`. You can set the number of Monte Carlo samples by passing ``n_trans``
        into the transforms, for example ``Rotate(n_trans=2)`` will average over 2 samples per call. For rotate and reflect, by setting ``n_trans``
        to the maximum (e.g. 4 for 90 degree rotations, 2 for 1D reflections), it will average over the whole group, for example:

        ``Rotate(n_trans=4, multiples=90, positive=True) * Reflect(n_trans=2, dims=[-1])``

    See :ref:`sphx_glr_auto_examples_basics_demo_transforms.py` for an example.

    :param Callable denoiser: Denoiser :math:`\operatorname{D}_{\sigma}`.
    :param Transform transform: geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
        See :ref:`docs <transform>` for list of available transforms.
    :param bool random: if True, the denoiser is applied to a randomly transformed version of the input image
        each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
        If False, the denoiser is applied to the average of all the transformed images, turning the denoiser into an
        equivariant denoiser with respect to the chosen group of transformations. Ignored if ``transform`` is provided.
    """

    def __init__(
        self, denoiser: Denoiser, transform: Optional[Transform] = None, random=True
    ):
        super().__init__()
        self.denoiser = denoiser

        if transform is not None:
            self.transform = transform
        else:
            if random:
                self.transform = Rotate(
                    n_trans=1, multiples=90, positive=True
                ) * Reflect(n_trans=1, dims=[-1])
            else:
                self.transform = Rotate(
                    n_trans=4, multiples=90, positive=True
                ) * Reflect(n_trans=2, dims=[-1])

    def forward(self, x, *denoiser_args, **denoiser_kwargs):
        r"""
        Symmetrize the denoiser by the transformation to create an equivariant denoiser and apply to input.

        The symmetrization collects the average if multiple samples are used (controlled with ``n_trans`` in the transform).

        :param torch.Tensor x: input image.
        :param \*denoiser_args: args for denoiser function e.g. sigma noise level.
        :param \**denoiser_kwargs: kwargs for denoiser function e.g. sigma noise level.
        :return: denoised image.
        """
        return self.transform.symmetrize(self.denoiser, average=True)(
            x, *denoiser_args, **denoiser_kwargs
        )
