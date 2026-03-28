import deepinv as dinv
import deepinv as dinv

from deepinv.loss.loss import Loss
from deepinv.loss.mc import MCLoss
from deepinv.loss.metric.distortion import MSE
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.r2r import R2RLoss

import weakref


class ESLoss(Loss):

    consistency_loss: Loss
    prediction_loss: Loss

    def __init__(
        self,
        *,
        mask_generator,
        noise_model,
        alpha: float = 0.2,
        weight: float = 1.0,
        eval_n_samples: int = 10,
        transform,
        eval_transform=None,
        equivariant_model: bool = False,
    ):
        super().__init__()
        if not isinstance(noise_model, dinv.physics.ZeroNoise):
            consistency_loss = R2RLoss(alpha=alpha, eval_n_samples=eval_n_samples)
        else:
            consistency_loss = None
        self.consistency_loss = consistency_loss
        self.prediction_loss = MCLoss(metric=MSE())
        self.transform = transform
        self.eval_transform = eval_transform
        self.equivariant_model = equivariant_model
        # Store the SplittingModel possibly wrapped in the adapted model
        # It is also used to avoid adapting the same model more than once.
        self._splitting_model_mapping = weakref.WeakKeyDictionary()

    def forward(self, x_net, y, physics, model, **kwargs):
        if model in self._splitting_model_mapping:
            splitting_model = self._splitting_model_mapping[model]
        else:
            raise RuntimeError(
                f"Unregistered model {type(model)}. Make sure to adapt the model using `adapt_model` before computing the loss."
            )
        masks = splitting_model.get_masks()
        loss_total = 0
        N_masks = 0
        for mask in masks:
            mask = mask * getattr(physics, "mask", 1.0)
            mask2 = getattr(physics, "mask", 1.0) - mask
            y2, physics2 = SplittingLoss.split(mask2, y, physics)
            loss_value = self.prediction_loss(
                x_net=x_net,
                y=y2,
                physics=physics2,
                model=model,
                **kwargs,
            )
            # Normalization
            loss_value = loss_value / mask2.mean()

            if self.consistency_loss is not None:
                y1, physics1 = SplittingLoss.split(mask, y, physics)
                loss_value = loss_value + self.consistency_loss(
                    x_net=x_net,
                    y=y1,
                    physics=physics1,
                    model=model,
                    **kwargs,
                )
            loss_total = loss_total + loss_value
            N_masks += 1
        return loss_total / N_masks

    def adapt_model(self, model):
        if model not in self._splitting_model_mapping:
            # if the model is not already equivariant, we make it so using Reynolds averaging
            if not self.equivariant_model:
                model = dinv.models.EquivariantReconstructor(
                    model=model,
                    transform=self.transform,
                    eval_transform=self.eval_transform,
                )

            splitting_model = SplittingLoss.SplittingModel(
                model,
                split_ratio=0.9,
                mask_generator=None,
                eval_n_samples=5,
                eval_split_input=True,
                eval_split_output=False,
                pixelwise=True,
            )

            if self.consistency_loss is not None:
                model = self.consistency_loss.adapt_model(splitting_model)
            else:
                model = splitting_model

            self._splitting_model_mapping[model] = splitting_model
        return model
