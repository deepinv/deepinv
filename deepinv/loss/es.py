import deepinv as dinv
import math
import deepinv as dinv

from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.r2r import R2RLoss


# A R2R-like loss to be used in conjunction with the splitting loss
class _SplitR2RLoss(R2RLoss):

    def __init__(
        self,
        mask_generator,
        noise_model,
        alpha=0.2,
        split_ratio=0.6,
        weight=1.0,
        **kwargs,
    ):
        super().__init__(alpha=alpha, **kwargs)
        self.split_ratio = split_ratio
        self.noise_model = noise_model
        self.noise_model.update_parameters(
            sigma=noise_model.sigma * math.sqrt(alpha / (1 - alpha))
        )
        self.weight = weight
        self.mask_generator = mask_generator

    def forward(self, x_net, y, physics, model, **kwargs):
        ya = model.get_corruption()
        yb = (y - ya * (1 - self.alpha)) / self.alpha

        masks = model.get_masks()
        loss_total = 0
        N_masks = 0
        for mask in masks:
            mask = mask * getattr(physics, "mask", 1.0)
            r2rloss = self.metric(mask * physics.A(x_net), mask * yb)
            loss = self.weight * r2rloss / mask.mean()
            loss_total = loss_total + loss
            N_masks += 1

        return loss_total / N_masks

    def adapt_model(self, model):
        return (
            model
            if isinstance(model, self.R2RSplittingModel)
            else self.R2RSplittingModel(
                model,
                split_ratio=self.split_ratio,
                mask_generator=self.mask_generator,
                noise_model=self.noise_model,
                eval_n_samples=self.eval_n_samples,
            )
        )

    class R2RSplittingModel(SplittingLoss.SplittingModel):
        def __init__(
            self, model, split_ratio, mask_generator, noise_model, eval_n_samples
        ):
            super().__init__(
                model,
                split_ratio=split_ratio,
                mask_generator=mask_generator,
                eval_n_samples=eval_n_samples,
                eval_split_input=True,
                eval_split_output=False,
                pixelwise=True,
            )
            self.noise_model = noise_model

        def split(self, mask, y, physics=None):
            y1, physics1 = SplittingLoss.split(mask, y, physics)
            noiser_y1 = self.noise_model(y1)
            self.corruption = noiser_y1
            return mask * noiser_y1, physics1

        def get_corruption(self):
            return self.corruption


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
            # Use R2R Splitting loss
            split_r2r_loss = _SplitR2RLoss(
                mask_generator=mask_generator,
                noise_model=noise_model,
                alpha=alpha,
                weight=weight,
                eval_n_samples=eval_n_samples,
            )
            consistency_loss = R2RLoss(alpha=alpha, eval_n_samples=eval_n_samples)
        else:
            # Use only the splitting loss
            split_r2r_loss = None
            consistency_loss = None
        self.split_r2r_loss = split_r2r_loss
        self.consistency_loss = consistency_loss
        self.transform = transform
        self.eval_transform = eval_transform
        self.equivariant_model = equivariant_model

    def forward(self, x_net, y, physics, model, **kwargs):
        masks = model.get_masks()
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
        if not isinstance(model, dinv.loss.SplittingLoss.SplittingModel):
            # if the model is not already equivariant, we make it so using Reynolds averaging
            if not self.equivariant_model:
                model = dinv.models.EquivariantReconstructor(
                    model=model,
                    transform=self.transform,
                    eval_transform=self.eval_transform,
                )
            if self.split_r2r_loss is not None:
                model = self.split_r2r_loss.adapt_model(model)
            else:
                model = self.splitting_loss.adapt_model(model)
        return model
