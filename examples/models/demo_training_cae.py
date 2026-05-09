r"""
Training or finetuning a CAE (or a VAE) model
====================================================================================================

This example provides a simple quick start introduction to train or finetune a compressive autoencoder (CAE).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module
from deepinv.loss.loss import Loss
import deepinv as dinv

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
rng = torch.Generator(device=device).manual_seed(0)

HALF_LOG_TWO_PI = 0.91894

# %%
# Define a custom trainer for CAE models


class CustomTrainer(dinv.Trainer):
    def model_inference(self, y, physics, x=None, train=True, **kwargs):
        # check if the forward has 'update_parameters' method (cached), and if so, update the parameters
        if self._model_accepts_update_parameters:
            kwargs["update_parameters"] = True

        if train:
            self.model.train()
            return self.model(y, **kwargs)
        else:
            self.model.eval()
            with torch.no_grad():
                if self.plot_convergence_metrics:
                    x_net, self.conv_metrics = self.model(
                        y,
                        **kwargs,
                    )
                else:
                    x_net = self.model(y, **kwargs)

            return x_net

    def compute_loss(
        self, physics, x, y, train=True, epoch: int = None, step: bool = False
    ):
        logs = {}

        if train and step:
            # set_to_none=True can slightly reduce overhead vs. zeroing memory
            self.optimizer.zero_grad(set_to_none=True)

        if train or self.compute_eval_losses:
            # Evaluate reconstruction network
            out_dict = self.model_inference(
                y=y,
                physics=physics,
                x=x,
                train=True,
                decode_mean_only=self.decode_mean_only,
            )
            x_net = out_dict["x_rec"]

            # Compute the losses
            loss_total = 0
            for k, l in enumerate(self.losses):
                loss = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                    epoch=epoch,
                    out_dict=out_dict,
                )
                loss_total += loss.mean()
                meters = (
                    self.logs_losses_train[k] if train else self.logs_losses_eval[k]
                )
                meters.update(loss.detach().cpu().numpy())
                if len(self.losses) > 1 and self.verbose_individual_losses:
                    logs[l.__class__.__name__] = meters.avg

            meters = self.logs_total_loss_train if train else self.logs_total_loss_eval
            meters.update(loss_total.item())
            logs[f"TotalLoss"] = meters.avg
        else:
            loss_total = 0
            x_net = None

        if train:
            loss_total.backward()  # Backward the total loss

            norm = self.check_clip_grad()
            if norm is not None:
                logs["gradient_norm"] = self.check_grad_val.avg

            if step:
                self.optimizer.step()  # Optimizer step

        return loss_total, x_net, logs

    def compute_metrics(
        self, x, x_net, y, physics, logs, train=True, epoch: int = None
    ):
        if len(self.metrics) > 0 and (
            not (train and self.compute_train_metrics) or x_net is None
        ):
            # re-evaluate the model in eval mode if needed
            out_dict = self.model_inference(
                y=y,
                physics=physics,
                x=x,
                train=False,
                decode_mean_only=self.decode_mean_only,
            )
            x_net = out_dict["x_rec"]

        # Compute the metrics over the batch
        with torch.no_grad():
            for k, l in enumerate(self.metrics):
                metric = l(
                    x=x,
                    x_net=x_net,
                    y=y,
                    physics=physics,
                    model=self.model,
                )

                current_log = (
                    self.logs_metrics_train[k] if train else self.logs_metrics_eval[k]
                )
                current_log.update(metric.detach().cpu().numpy())
                logs[l.__class__.__name__] = current_log.avg

                if not train and self.compare_no_learning:
                    x_lin = self.no_learning_inference(y, physics)
                    no_learning_model = self._NoLearningModel(trainer=self)
                    metric = l(
                        x=x, x_net=x_lin, y=y, physics=physics, model=no_learning_model
                    )
                    self.logs_metrics_no_learning[k].update(
                        metric.detach().cpu().numpy()
                    )
                    logs[f"{l.__class__.__name__} no learning"] = (
                        self.logs_metrics_no_learning[k].avg
                    )
        return x_net, logs


# %% Losses for CAE models


class MLELoss(dinv.loss.Loss):
    def __init__(self, reduction: str = "none", use_gamma: bool = True):
        super(MLELoss, self).__init__()
        self.reduction = reduction
        self.use_gamma = use_gamma

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: None,
        physics: None,
        model: Module,
        out_dict: Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if "x_rec_std" not in out_dict or self.use_gamma:
            x_net_std = model.get_gamma()
        else:
            x_net_std = out_dict["x_rec_std"]

        mle_loss = (
            0.5 * ((x - x_net) / x_net_std).pow(2) + x_net_std.log() + HALF_LOG_TWO_PI
        )
        if self.reduction == "mean":
            mle_loss = mle_loss.mean()
        elif self.reduction == "sum":
            mle_loss = mle_loss.sum()
        elif self.reduction == "n_pixels":
            mle_loss = mle_loss.mean() * x_net.shape[1]

        return mle_loss


class KLLoss(Loss):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(
        self,
        x_net: Tensor,
        x: Tensor,
        y: None,
        physics: None,
        model: Module,
        out_dict: dict,
        **kwargs,
    ) -> torch.Tensor:
        return model.compute_kl_loss(out_dict)


# %%
# Then define the dataset. Here we simulate a dataset of measurements from Urban100.
#
# .. tip::
#     See :ref:`datasets <datasets>` for types of datasets DeepInverse supports: e.g. paired, ground-truth-free,
#     single-image...
#

from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Grayscale

dataset = dinv.datasets.Urban100HR(
    ".",
    download=True,
    transform=Compose([ToTensor(), Grayscale(), Resize(256), CenterCrop(64)]),
)

train_dataset, test_dataset = torch.utils.data.random_split(
    torch.utils.data.Subset(dataset, range(50)), (0.8, 0.2)
)

dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=dinv.physics.Denoising(dinv.physics.ZeroNoise()),
    device=device,
    save_dir=".",
    batch_size=1,
)

train_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=True), shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=False), shuffle=False
)

# %%
# Visualize a data sample:
#

# x, _ = next(iter(test_dataloader))
# dinv.utils.plot({"Ground truth": x})


# %%
# The model is a CAE (dinv.models.MbtCAE), or a VAE (dinv.models.VAE)

alpha = 0.0483  # compression trade-off (the higher, the less compression). Common compression factors are between 0.003 and 0.2
path_to_pretrained = None  # path/to/pretrained/model.pth or None to train from scratch or 'download' to start from a pretrained model that will be downloaded
model = dinv.models.MbtCAE(
    in_channels=1,
    decode_mean_only=False,
    pretrained=path_to_pretrained,
    alpha=alpha,
    n_bits=8,
).to(device)

# %%
# Train the model
# ----------------------------------------------------------------------------------------
# First Step of the training: training of the CAE autoencoder using MLE + KL losses
#
losses = [MLELoss(reduction="n_pixels", use_gamma=True), KLLoss()]

# all params except model.variance_decoder params
parameters_to_optimize = [
    p for n, p in model.named_parameters() if "variance_decoder" not in n
]

trainer = CustomTrainer(
    model=model,
    physics=dinv.physics.Denoising(dinv.physics.ZeroNoise()),
    optimizer=torch.optim.Adam(parameters_to_optimize, lr=1e-4),
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=5,
    losses=losses,
    metrics=dinv.metric.PSNR(),
    device=device,
    plot_images=False,
    show_progress_bar=False,
)
# decode only mean during the first step of the training
trainer.decode_mean_only = True

_ = trainer.train()


# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :func:`deepinv.test` function.
#
# The testing function will compute metrics and plot and save the results.

trainer.test(test_dataloader)

# %%
# Second Step of the training: training of the CAE variance decoder using MLE loss

losses = MLELoss(reduction="n_pixels", use_gamma=False)
parameters_to_optimize = model.variance_decoder.parameters()

trainer = CustomTrainer(
    model=model,
    physics=dinv.physics.Denoising(dinv.physics.ZeroNoise()),
    optimizer=torch.optim.Adam(parameters_to_optimize, lr=1e-4),
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=5,
    grad_clip=1.0,
    check_grad=True,
    losses=losses,
    metrics=dinv.metric.PSNR(),
    device=device,
    plot_images=False,
    show_progress_bar=False,
)

# decode mean and variance during the second step of the training
trainer.decode_mean_only = False

_ = trainer.train()

# %%
# Test the network
trainer.test(test_dataloader)
