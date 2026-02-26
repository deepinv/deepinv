from __future__ import annotations

from torch import Tensor
from deepinv.transform import Transform, Rotate, Reflect
from .base import Denoiser, Reconstructor
from deepinv.physics.forward import LinearPhysics
from typing import Any, Callable
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
    :param bool random: if True, the denoiser is applied to a randomly transformed version of the input image
        each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
        If False, the denoiser is applied to the average of all the transformed images, turning the denoiser into an
        equivariant denoiser with respect to the chosen group of transformations. Ignored if ``transform`` is provided.
    """

    def __init__(
        self,
        denoiser: Denoiser,
        transform: Transform | None = None,
        random: bool = True,
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

    def forward(self, x: Tensor, *denoiser_args, **denoiser_kwargs) -> Tensor:
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


def _symmetrize(
    transform,
    f: Callable[[torch.Tensor, Any], torch.Tensor],
    average: bool = False,
    collate_batch: bool = True,
) -> Callable[[torch.Tensor, Any], torch.Tensor]:
    r"""
    Symmetrise a function with a transform and its inverse.

    Given a function :math:`f(\cdot):X\rightarrow X` and a transform :math:`T_g`, returns the group averaged function  :math:`\sum_{i=1}^N T_{g_i}^{-1} f(T_{g_i} \cdot)` where :math:`N` is the number of random transformations.

    For example, this is useful for Reynolds averaging a function over a group. Set ``average=True`` to average over ``n_trans``.
    For example, use ``Rotate(n_trans=4, positive=True, multiples=90).symmetrize(f)`` to symmetrize f over the entire group.

    :param Callable[[torch.Tensor, Any], torch.Tensor] f: function acting on tensors.
    :param bool average: monte carlo average over all random transformations (in range ``n_trans``) when symmetrising to get same number of output images as input images. No effect when ``n_trans=1``.
    :param bool collate_batch: if ``True``, collect ``n_trans`` transformed images in batch dim and evaluate ``f`` only once.
        However, this requires ``n_trans`` extra memory. If ``False``, evaluate ``f`` for each transformation.
        Always will be ``False`` when transformed images aren't constant shape.
    :return Callable[[torch.Tensor, Any], torch.Tensor]: decorated function.
    """

    def symmetrized_reconstructor(y, physics, *args, **kwargs):
        params = transform.get_params(physics.A_adjoint(y))
        if transform.constant_shape and collate_batch:
            # construct n_tran problems and solve them in parallel
            B = y.size(0)
            y = torch.cat([y] * transform.n_trans)
            t = LinearPhysics(
                A=lambda x: transform.transform(x, batchwise=False, **params),
                A_adjoint=lambda x: transform.inverse(x, batchwise=False, **params),
            )
            xt = transform.transform(
                f(y, physics=physics * t, *args, **kwargs),
                batchwise=False,
                **params,
            )
            return xt.reshape((-1, B) + xt.size()[1:]).mean(axis=0) if average else xt
        else:
            out = []
            for _params in transform.iterate_params(params):
                # Step through n_trans (or combinations) one-by-one
                t = LinearPhysics(
                    A=lambda x: transform.transform(x, **_params),
                    A_adjoint=lambda x: transform.inverse(x, **_params),
                )
                out.append(
                    transform.transform(
                        f(y, physics=physics * t, *args, **kwargs), **_params
                    )
                )
            return torch.stack(out, dim=1).mean(dim=1) if average else torch.cat(out)

    def symmetrized(x, *args, **kwargs):
        params = transform.get_params(x)
        if transform.constant_shape and collate_batch:
            # Collect over n_trans
            xt = transform.inverse(
                f(transform.transform(x, **params), *args, **kwargs),
                batchwise=False,
                **params,
            )
            return xt.reshape(-1, *x.shape).mean(axis=0) if average else xt
        else:
            # Step through n_trans (or combinations) one-by-one
            out = []
            for _params in transform.iterate_params(params):
                print(_params)
                out.append(
                    transform.inverse(
                        f(transform.transform(x, **_params), *args, **kwargs), **_params
                    )
                )

            return torch.stack(out, dim=1).mean(dim=1) if average else torch.cat(out)

    if isinstance(f, Reconstructor):
        return lambda y, physics, *args, **kwargs: symmetrized_reconstructor(
            y, physics, *args, **kwargs
        )
    else:
        return lambda x, *args, **kwargs: (
            transform.wrap_flatten_C(symmetrized)(x, *args, **kwargs)
            if transform._check_x_5D(x) and transform.flatten_video_input
            else symmetrized(x, *args, **kwargs)
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

    See :ref:`sphx_glr_auto_examples_basics_demo_transforms.py` for an example.

    :param Callable model: Reconstruction model :math:`\inversef{y}{A}`.
    :param Transform transform: geometric transformation. If None, defaults to rotations of multiples of 90 with horizontal flips (see note above).
        See :ref:`docs <transform>` for list of available transforms.
    :param bool random: if True, the model is applied to a randomly transformed version of the input image
        each time i.e. a Monte-Carlo approximation of an equivariant denoiser.
        If False, the model is applied to the average of all the transformed images, turning the reconstructor into an
        equivariant reconstructor with respect to the chosen group of transformations. Ignored if ``transform`` is provided.
    """

    def __init__(
        self,
        model: Reconstructor,
        train_transform,
        eval_transform=None,
    ):
        super().__init__()
        self._model = model

        if eval_transform is None:
            eval_transform = train_transform
        self._transform = train_transform
        self._eval_transform = eval_transform

    def forward(self, y, physics, *reconstructor_args, **reconstructor_kwargs):
        r"""
        Symmetrize the reconstructor by the transformation to create an equivariant reconstructor and apply to input.

        The symmetrization collects the average if multiple samples are used (controlled with ``n_trans`` in the transform).

        :param torch.Tensor x: input image.
        :param \*denoiser_args: args for denoiser function e.g. sigma noise level.
        :param \**denoiser_kwargs: kwargs for denoiser function e.g. sigma noise level.
        :return: denoised image.
        """
        if self.training:
            transform = self._transform
        else:
            transform = self._eval_transform
        return _symmetrize(transform, self._model, average=True)(
            y, physics, *reconstructor_args, **reconstructor_kwargs
        )
