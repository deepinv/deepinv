from __future__ import annotations
import deepinv as dinv
import deepinv as dinv

from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.physics.generator.base import PhysicsGenerator
from deepinv.transform.base import Transform

import torch

import weakref


class EquivariantSplittingLoss(Loss):
    r"""
    Equivariant splitting loss.

    Implements the measurement splitting loss proposed by :footcite:t:`sechaud26Equivariant`. It generalizes the regular :class:`deepinv.loss.SplittingLoss` by providing an additional measurement consistency term supporting noise-less losses like :class:`deepinv.loss.MCLoss`, but also noise-aware losses including :class:`deepinv.loss.R2RLoss` and :class:`deepinv.loss.SureGaussianLoss`. Moreover, it automatically renders the base reconstructor equivariant using the Reynolds averaging implemented in :class:`deepinv.models.EquivariantReconstructor`.

    The training loss takes the general form:

    .. math::

        \mathcal{L}_{\mathrm{ES}} (y, A, f) = \mathbb{E}_g \left\{ \mathbb{E}_{y_1, A_1 \mid y, A T_g} \left\{ \| A_1 f(y_1, A_1) - A_1 x \|^2 + \| A_2 f(y_1, A_1) - A_2 x \|^2 \right\} \right\}

    where :math:`f` denotes the reconstructor, :math:`A` the physics operator, :math:`x` the ground truth image, :math:`y` the measurement, :math:`T_g` a group action (e.g., rotations).

    The second expectation is taken over the distribution specified by ``mask_generator`` of all possible splittings of :math:`A T_g`, i.e., :math:`A T_g = [A_1^\top, A_2^\top]^\top`, with the associated measurements denoted as :math:`y_1` and :math:`y_2`.

    The main idea behind equivariant splitting is that the more the reconstructor is equivariant to suitable transformations, the better the final performance will be. A general way to make a reconstructor equivariant is to add a Reynolds averaging step in the reconstructor, which is generally estimated using a Monte Carlo approach at training time. For this reason, :class:`EquivariantSplittingLoss` takes two different instances of :class:`deepinv.transform.Transform` as input: one for training ``transform`` and one for evaluation ``eval_transform``.

    It is also possible to design an equivariant reconstructor without Reynolds averaging, using equivariant layers. In that case, Reynolds averaging can be disabled to avoid its additional computational cost by leaving ``transform`` and ``eval_transform`` to ``None``.

    The training loss consists in two terms, a consistency term where the comparison is performed against :math:`A_1 x` and a prediction term where the comparison is performed against :math:`A_2 x`. Two parameters control the way these two terms are computed: ``consistency_loss`` and ``prediction_loss``.

    In the absence of noise, the equivariant splitting loss :math:`\mathcal{L}_{\mathrm{ES}}` can be computed exactly without having access to ground truth images. Indeed, in that case, :math:`A_1 x = y_1` and :math:`A_2 x = y_2`. Setting ``consistency_loss`` and ``prediction_loss`` to ``deepinv.loss.MCLoss(metric=deepinv.metric.MSE())`` allows to compute the loss this way.

    In the presence of noise, as long as the splitting scheme is chosen so that the resulting noise components are independent, the prediction term can be estimated without bias using ``deepinv.loss.MCLoss(metric=deepinv.metric.MSE())`` for ``prediction_loss``. This is notably the case for typical splitting schemes, e.g., :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator` when the noise is pixel-wise independent, e.g., :class:`deepinv.physics.GaussianNoise`.

    The consistency term can be estimated using one of the self-supervised denoising losses listed in :ref:`self-supervised-losses`, e.g., :class:`deepinv.loss.R2RLoss` or :class:`deepinv.loss.SureGaussianLoss` if the noise distribution is known exactly. If the noise parameters are unknown, UNSURE can be used instead, i.e., :class:`deepinv.loss.SureGaussianLoss` with the option ``unsure`` enabled, and if the noise distribution is unknown altogether, the consistency term can be estimated using the Noise2x family of losses.

    At training time, a single splitting is performed for each sample in the batch, however, at evaluation time, the reconstructions are averaged over multiple splittings as specified by ``eval_n_samples``.

    :param PhysicsGenerator mask_generator: the generator specifying the distribution of splittings.
    :param Loss consistency_loss: the loss used to compute the consistency term.
    :param Loss prediction_loss: the loss used to compute the prediction term.
    :param Transform transform: transformations to be used in training mode for Reynolds averaging.
    :param Transform eval_transform: transformations to be used in evaluation mode for Reynolds averaging. It can be used to have true Reynolds averaging at evaluation time and efficient Monte Carlo estimation at training time. If left unspecified, the value of ``transform`` is used at evaluation time as well.

    |sep|

    :Example:

    >>> import torch
    >>> import deepinv as dinv
    >>> physics = dinv.physics.Inpainting(img_size=(1, 8, 8), mask=0.5)
    >>> model = dinv.models.RAM(pretrained=True)
    >>> mask_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    ...     img_size=(1, 8, 8),
    ...     split_ratio=0.9,
    ...     pixelwise=True,
    ... )
    >>> train_transform = dinv.transform.Rotate(
    ...     n_trans=1, multiples=90, positive=True
    ... ) * dinv.transform.Reflect(n_trans=1, dim=[-1])
    >>> eval_transform = dinv.transform.Rotate(
    ...     n_trans=4, multiples=90, positive=True
    ... ) * dinv.transform.Reflect(n_trans=2, dim=[-1])
    >>> loss = dinv.loss.EquivariantSplittingLoss(
    ...     mask_generator=mask_generator,
    ...     consistency_loss=dinv.loss.MCLoss(metric=dinv.metric.MSE()),
    ...     prediction_loss=dinv.loss.MCLoss(metric=dinv.metric.MSE()),
    ...     transform=train_transform,
    ...     eval_transform=eval_transform,
    ...     eval_n_samples=5,
    ... )
    >>> model = loss.adapt_model(model)
    >>> x = torch.ones((1, 1, 8, 8))
    >>> y = physics(x)
    >>> x_net = model(y, physics, update_parameters=True)
    >>> l = loss(x_net, y, physics, model)
    >>> print(l.item() > 0)
    True

    """

    consistency_loss: Loss
    prediction_loss: Loss

    def __init__(
        self,
        *,
        mask_generator: PhysicsGenerator,
        consistency_loss: Loss,
        prediction_loss: Loss,
        eval_n_samples: int = 5,
        transform: Transform | None = None,
        eval_transform: Transform | None = None,
    ):
        super().__init__()

        self._name = "es"

        if eval_transform is None:
            eval_transform = transform
        elif transform is None:
            raise ValueError(
                "If eval_transform is specified, transform must also be specified."
            )

        self.transform = transform
        self.eval_transform = eval_transform

        self.mask_generator = mask_generator
        self.consistency_loss = consistency_loss
        self.prediction_loss = prediction_loss

        self.eval_n_samples = eval_n_samples
        # Store the SplittingModel possibly wrapped in the adapted model
        # It is also used to avoid adapting the same model more than once.
        self._splitting_model_mapping = weakref.WeakKeyDictionary()

    def forward(self, x_net, y, physics, model, **kwargs):
        r"""
        Compute the equivariant splitting loss.

        :param torch.Tensor x_net: the reconstructed image.
        :param torch.Tensor y: the measurement.
        :param Physics physics: the physics operator.
        :param Reconstructor model: the reconstruction function.
        :return: (:class:`torch.Tensor`) the loss value.
        """

        if model in self._splitting_model_mapping:
            splitting_model = self._splitting_model_mapping[model]
        else:
            raise RuntimeError(
                f"Unregistered model {type(model)}. Make sure to adapt the model using `adapt_model` before computing the loss."
            )
        masks = splitting_model.get_masks()
        loss_values = []
        for mask1 in masks:
            mask1 = mask1 * getattr(physics, "mask", 1.0)
            mask2 = getattr(physics, "mask", 1.0) - mask1
            y2, physics2 = SplittingLoss.split(mask2, y, physics)
            prediction_loss_value = self.prediction_loss(
                x_net=x_net,
                y=y2,
                physics=physics2,
                model=model,
                **kwargs,
            )
            prediction_loss_value = prediction_loss_value / mask2.mean()

            if self.consistency_loss is not None:
                y1, physics1 = SplittingLoss.split(mask1, y, physics)
                consistency_loss_value = self.consistency_loss(
                    x_net=x_net,
                    y=y1,
                    physics=physics1,
                    model=model,
                    **kwargs,
                )
                consistency_loss_value = consistency_loss_value / mask1.mean()

            loss_value = prediction_loss_value + consistency_loss_value
            loss_values.append(loss_value)
        loss_values = torch.stack(loss_values, dim=0)
        return loss_values.mean(0)

    def adapt_model(self, model):
        r"""
        Adapt the reconstructor for equivariant splitting.

        It wraps the input reconstructor in a splitting model and optionally in a :class:`deepinv.models.EquivariantReconstructor` if requested.

        :param Reconstructor model: the reconstructor to adapt.
        :return: the adapted reconstructor.
        """
        if model not in self._splitting_model_mapping:
            # Apply Reynolds averaging if requested
            if self.transform is not None:
                model = dinv.models.EquivariantReconstructor(
                    model=model,
                    transform=self.transform,
                    eval_transform=self.eval_transform,
                )

            splitting_model = SplittingLoss.SplittingModel(
                model,
                mask_generator=self.mask_generator,
                eval_n_samples=self.eval_n_samples,
                eval_split_input=True,
                eval_split_output=False,
                # Necessary but unused when mask_generator is specified
                split_ratio=None,
                pixelwise=None,
            )

            if self.consistency_loss is not None:
                model = self.consistency_loss.adapt_model(splitting_model)
            else:
                model = splitting_model

            self._splitting_model_mapping[model] = splitting_model
        return model
