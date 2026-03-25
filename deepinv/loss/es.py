import torch
import deepinv as dinv
import torchvision
import math
import torch
import deepinv as dinv

from deepinv.transform import Rotate, Reflect
from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.r2r import R2RLoss

import torch


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

        mask = model.get_mask() * getattr(physics, "mask", 1.0)
        r2rloss = self.metric(mask * physics.A(x_net), mask * yb)
        return self.weight * r2rloss / mask.mean()

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

    def __init__(
        self,
        *,
        mask_generator,
        noise_model,
        alpha: float = 0.2,
        weight: float = 1.0,
        eval_n_samples: int = 10,
        train_transform,
        eval_transform=None,
        equivariant_model: bool = False,
    ):
        super().__init__()
        self.splitting_loss = SplittingLoss()
        if not isinstance(noise_model, dinv.physics.ZeroNoise):
            # Use R2R Splitting loss
            split_r2r_loss = _SplitR2RLoss(
                mask_generator=mask_generator,
                noise_model=physics.noise_model,
                alpha=alpha,
                weight=weight,
                eval_n_samples=eval_n_samples,
            )
        else:
            # Use only the splitting loss
            split_r2r_loss = None
        self.split_r2r_loss = split_r2r_loss
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.equivariant_model = equivariant_model

    def forward(self, x_net, y, physics, model, **kwargs):
        loss_value = self.splitting_loss(
            x_net=x_net,
            y=y,
            physics=physics,
            model=model,
            **kwargs,
        )
        if self.split_r2r_loss is not None:
            loss_value = loss_value + self.split_r2r_loss(
                x_net=x_net,
                y=y,
                physics=physics,
                model=model,
                **kwargs,
            )
        return loss_value

    def adapt_model(self, model):
        if not isinstance(model, dinv.loss.SplittingLoss.SplittingModel):
            # if the model is not already equivariant, we make it so using Reynolds averaging
            if not self.equivariant_model:
                model = dinv.models.EquivariantReconstructor(
                    model=model,
                    train_transform=self.train_transform,
                    eval_transform=self.eval_transform,
                )
            if self.split_r2r_loss is not None:
                model = self.split_r2r_loss.adapt_model(model)
            else:
                model = self.splitting_loss.adapt_model(model)
        return model
