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
        self._splitting_loss = SplittingLoss()
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
        self._split_r2r_loss = split_r2r_loss
        self._train_transform = train_transform
        self._eval_transform = eval_transform
        self._equivariant_model = equivariant_model

    def forward(self, x_net, y, physics, model, **kwargs):
        loss_value = self._splitting_loss(
            x_net=x_net,
            y=y,
            physics=physics,
            model=model,
            **kwargs,
        )
        if self._split_r2r_loss is not None:
            loss_value = loss_value + self._split_r2r_loss(
                x_net=x_net,
                y=y,
                physics=physics,
                model=model,
                **kwargs,
            )
        return loss_value

    def adapt_model(self, model):
        # if the model is not already equivariant, we make it so using Reynolds averaging
        if not self._equivariant_model:
            model = dinv.models.EquivariantReconstructor(
                model=model,
                train_transform=self._train_transform,
                eval_transform=self._eval_transform,
            )
        if self._split_r2r_loss is not None:
            model = self._split_r2r_loss.adapt_model(model)
        else:
            model = self._splitting_loss.adapt_model(model)
        return model


if __name__ == "__main__":
    for new_impl in [False, True]:
        device = "cuda:0"

        torch.manual_seed(0)
        rng = torch.Generator(device=device).manual_seed(0)

        transform = torchvision.transforms.Resize(128)
        dataset = dinv.datasets.SimpleFastMRISliceDataset(
            dinv.utils.get_data_home(),
            anatomy="knee",
            transform=transform,
            train=True,
            download=True,
        )
        x = dataset[0].to(device).unsqueeze(0)
        img_size = x.shape[-2:]

        physics_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=img_size,
            acceleration=8,
            center_fraction=0.03,
            rng=rng,
            device=device,
        )
        physics = dinv.physics.MRI(img_size=img_size, device=device)
        mask = physics_generator.step()["mask"]
        physics.update(mask=mask)
        if False:
            physics.noise_model = dinv.physics.GaussianNoise(0.05, rng=rng)

        y = physics(x)

        img_size = (128, 128)

        split_generator = dinv.physics.generator.GaussianMaskGenerator(
            img_size=img_size,
            acceleration=2,
            center_fraction=0.0,
            rng=rng,
            device=device,
        )

        mask_generator = dinv.physics.generator.MultiplicativeSplittingMaskGenerator(
            (1, *img_size), split_generator, device=device
        )

        # A random transformation from the group D4
        train_transform = Rotate(n_trans=1, multiples=90, positive=True) * Reflect(
            n_trans=1, dim=[-1]
        )

        # # All of the transformations from the group D4
        # eval_transform = Rotate(
        #     n_trans=4, multiples=90, positive=True
        # ) * Reflect(n_trans=2, dim=[-1])
        eval_transform = None  # use same as train

        if new_impl:
            loss = [
                ESLoss(
                    mask_generator=mask_generator,
                    noise_model=physics.noise_model,
                    alpha=0.2,
                    weight=1.0,
                    eval_n_samples=10,
                    train_transform=train_transform,
                    eval_transform=eval_transform,
                )
            ]
            backbone = dinv.models.UNet(in_channels=2, out_channels=2, scales=4)
            model = dinv.models.ArtifactRemoval(backbone_net=backbone, mode="adjoint")
            model = loss[-1].adapt_model(model)
        else:
            loss = [SplittingLoss()]
            if not isinstance(physics.noise_model, dinv.physics.ZeroNoise):
                loss.append(
                    _SplitR2RLoss(
                        mask_generator=mask_generator,
                        noise_model=physics.noise_model,
                        alpha=0.2,
                        weight=1.0,
                        eval_n_samples=10,
                    )
                )
            backbone = dinv.models.UNet(in_channels=2, out_channels=2, scales=4)
            model = dinv.models.ArtifactRemoval(backbone_net=backbone, mode="adjoint")
            model = _EquivariantReconstructor(
                model=model,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
            model = loss[-1].adapt_model(model)
        model.to(device)
        model.train()

        x_hat = model(y, physics, update_parameters=True)

        train_loss = 0.0
        model.eval()
        for l in loss:
            train_loss += l(
                x=x,
                x_net=x_hat,
                y=y,
                physics=physics,
                model=model,
            ).item()

        x_hat = model(y, physics, update_parameters=True)

        print(f"Train loss: {train_loss:.4e}")
