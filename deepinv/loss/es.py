import deepinv as dinv
import math
import deepinv as dinv

from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.r2r import R2RLoss


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
        self.splitting_loss = SplittingLoss()
        if not isinstance(noise_model, dinv.physics.ZeroNoise):
            consistency_loss = R2RLoss(alpha=alpha, eval_n_samples=eval_n_samples)
        else:
            consistency_loss = None
        self.consistency_loss = consistency_loss
        self.transform = transform
        self.eval_transform = eval_transform
        self.equivariant_model = equivariant_model

    def forward(self, x_net, y, physics, model, **kwargs):
        masks = self.splitting_model.get_masks()
        loss_total = 0
        N_masks = 0
        for mask in masks:
            mask = mask * getattr(physics, "mask", 1.0)
            mask2 = getattr(physics, "mask", 1.0) - mask
            y2, physics2 = self.splitting_loss.split(mask2, y, physics)
            l = self.splitting_loss.metric(physics2.A(x_net), y2)

            loss_value = l / mask2.mean() if self.splitting_loss.normalize_loss else l

            if self.consistency_loss is not None:
                y1, physics1 = self.splitting_loss.split(mask, y, physics)
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
        if not getattr(model, "_ESLoss_adapted", False):
            # if the model is not already equivariant, we make it so using Reynolds averaging
            if not self.equivariant_model:
                model = dinv.models.EquivariantReconstructor(
                    model=model,
                    transform=self.transform,
                    eval_transform=self.eval_transform,
                )
            model = self.splitting_loss.adapt_model(model)
            self.splitting_model = model
            if self.consistency_loss is not None:
                model = self.consistency_loss.adapt_model(model)
            setattr(model, "_ESLoss_adapted", True)
        return model
