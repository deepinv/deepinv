"""
Reconstruct Anything Model (RAM) for solving inverse problems.
====================================================================================================

This example shows how to use the RAM foundation model to solve inverse problems. The RAM model, described in
the following `paper <https://arxiv.org/abs/2503.08915>`_, is a modified DRUNet architecture that is trained on
a large number of inverse problems.
"""

import torch
import deepinv as dinv
from deepinv.models import RAM

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained model
model = RAM(device=device)

# load image
x = dinv.utils.load_example("butterfly.png", img_size=(127, 129)).to(device)

# create forward operator
physics = dinv.physics.Inpainting(
    img_size=(3, 127, 129),
    mask=0.3,
    noise_model=dinv.physics.GaussianNoise(0.05),
    device=device,
)

# generate measurement
y = physics(x)

# run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)

# compute PSNR
in_psnr = dinv.metric.PSNR()(x, y).item()
out_psnr = dinv.metric.PSNR()(x, x_hat).item()

# plot
dinv.utils.plot(
    [x, y, x_hat],
    [
        "Original",
        "Measurement\n PSNR = {:.2f}dB".format(in_psnr),
        "Reconstruction\n PSNR = {:.2f}dB".format(out_psnr),
    ],
    figsize=(8, 3),
)

# %%
# This model was also trained on various denoising problems, in particular on Poisson-Gaussian denoising.

sigma, gain = 0.2, 0.5
physics = dinv.physics.Denoising(
    noise_model=dinv.physics.PoissonGaussianNoise(sigma=sigma, gain=gain),
)

# generate measurement
y = physics(x)

# run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)
    # or alternatively, we can use the model without physics:
    # x_hat = model(y, sigma=sigma, gain=gain)

# compute PSNR
in_psnr = dinv.metric.PSNR()(x, y).item()
out_psnr = dinv.metric.PSNR()(x, x_hat).item()

# plot
dinv.utils.plot(
    [x, y, x_hat],
    [
        "Original",
        "Measurement\n PSNR = {:.2f}dB".format(in_psnr),
        "Reconstruction\n PSNR = {:.2f}dB".format(out_psnr),
    ],
    figsize=(8, 3),
)

# %%
# This model is not trained on all degradations, so it may not perform well on all inverse problems out-of-the-box.
# For instance, it is not trained on image demosaicing. Applying it to a demosaicing problem out-of-the-box will yield poor results,
# as shown in the following example:


# Define the Demosaicing physics
physics = dinv.physics.Demosaicing(
    img_size=x.shape[1:], noise_model=dinv.physics.PoissonNoise(0.1), device=device
)

# generate measurement
y = physics(x)

# run inference
with torch.no_grad():
    x_hat = model(y, physics=physics)

# compute PSNR
in_psnr = dinv.metric.PSNR()(x, y).item()
out_psnr = dinv.metric.PSNR()(x, x_hat).item()

# plot
dinv.utils.plot(
    [x, y, x_hat],
    [
        "Original",
        "Measurement\n PSNR = {:.2f}dB".format(in_psnr),
        "0 shot reconstruction\n PSNR = {:.2f}dB".format(out_psnr),
    ],
    figsize=(8, 3),
)


# %%
# To improve results, we can fine-tune the model on the specific degradation for the sample of interest.
# This can be done even in the absence of ground truth data, using unsupervised training.
# We showcase this in the following, where the model is fine-tuned on the measurement vector `y` itself.
# Here, since this example is run in a no-GPU environment, we will use a small patch of the image to speed up training,
# but in practice, we can use the full image.
#
# First, we will create a dataset for unsupervised training that


class UnsupDataset(torch.utils.data.Dataset):
    r"""
    Dataset for unsupervised learning tasks.

    This dataset is used to return only the data without any labels.

    :param torch.Tensor data: Input data tensor of shape (N, ...), where N is the number of samples and ... represents the data dimensions.
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return torch.nan, self.data[idx]


physics_train = dinv.physics.Demosaicing(
    img_size=(3, 64, 64),
    noise_model=dinv.physics.PoissonNoise(0.1, clip_positive=True),
    device=device,
)
x_train = x[..., :64, :64]  # take a small patch of the image
y_train = physics_train(x_train)

losses = [
    dinv.loss.R2RLoss(),
    dinv.loss.EILoss(dinv.transform.Shift(shift_max=0.4), weight=0.1),
]

dataset = UnsupDataset(y_train)

train_dataloader = torch.utils.data.DataLoader(dataset)

# %%
# In order to check the performance of the fine-tuned model, we will use a validation set.
# We will use a small patch of another image. Note that this validation is also performed in an unsupervised manner,
# so we will not use the ground truth validation image.
y_val = physics_train(dinv.utils.load_example("leaves.png")[..., :64, :64].to(device))
eval_dataloader = torch.utils.data.DataLoader(UnsupDataset(y_val))

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

max_epochs = 20
trainer = dinv.Trainer(
    model=model,
    physics=physics_train,
    eval_interval=5,
    ckp_interval=max_epochs - 1,
    metrics=losses[0],
    early_stop=True,
    device=device,
    losses=losses,
    epochs=max_epochs,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
)

# finetune
finetuned_model = trainer.train()

# %%
# We can now use the fine-tuned model to reconstruct the image from the measurement vector `y`.

with torch.no_grad():
    x_hat = finetuned_model(y, physics=physics)

# compute PSNR
in_psnr = dinv.metric.PSNR()(x, y).item()
out_psnr = dinv.metric.PSNR()(x, x_hat).item()

# plot
dinv.utils.plot(
    [x, y, x_hat],
    [
        "Original",
        "Measurement\n PSNR = {:.2f}dB".format(in_psnr),
        "Finetuned reconstruction\n PSNR = {:.2f}dB".format(out_psnr),
    ],
    figsize=(8, 3),
)
