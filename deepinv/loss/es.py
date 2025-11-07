import torch
import deepinv as dinv
import torchvision
import math
import torch
import deepinv as dinv

from deepinv.transform import Rotate, Reflect
from deepinv.models.base import Reconstructor
from deepinv.loss.loss import Loss
from deepinv.loss.measplit import SplittingLoss
from deepinv.loss.r2r import R2RLoss

from typing import Callable, Any
import torch
from deepinv.physics.forward import LinearPhysics


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


class _EquivariantReconstructor(Reconstructor):
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
            model = _EquivariantReconstructor(
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
