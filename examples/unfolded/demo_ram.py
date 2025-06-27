"""
RAM model for solving inverse problems.
====================================================================================================

This example shows how to use the RAM model method to solve inverse problems. The RAM model, described in
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
x = dinv.utils.load_example("butterfly.png").to(device)

# create forward operator
physics = dinv.physics.Inpainting(
    tensor_size=(3, 256, 256),
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
# This model is not trained on all degradations, so it may not perform well on all inverse problems.
# For instance, it is not trained on image demosaicing. Applying it to a demosaicing problem will yield poor results,
# as shown in the following example:


# Define the Demosaicing physics
physics = dinv.physics.Demosaicing(
    img_size=(3, 256, 256), noise_model=dinv.physics.PoissonNoise(0.1), device=device
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
from deepinv.datasets.utils import UnsupDataset


physics_train = dinv.physics.Demosaicing(
    img_size=(3, 64, 64),
    noise_model=dinv.physics.PoissonNoise(0.1, clip_positive=True),
    device=device,
)
x_train = x[..., :64, :64]  # take a small patch of the image
y_train = physics_train(x_train)


mc_loss = dinv.loss.R2RLoss()

t = dinv.transform.Shift(shift_max=0.4)
eq_loss = dinv.loss.EILoss(t, weight=0.1)

losses = [mc_loss, eq_loss]

dataset = UnsupDataset(y_train)

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# %%
# In order to check the performance of the fine-tuned model, we will use a validation set.
# We will use a small patch of another image. Note that this validation is also performed in an unsupervised manner,
# so we will not use the ground truth validation image.
y_val = physics_train(dinv.utils.load_example("leaves.png")[..., :64, :64].to(device))
eval_dataloader = torch.utils.data.DataLoader(
    UnsupDataset(y_val), batch_size=1, shuffle=True
)

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
