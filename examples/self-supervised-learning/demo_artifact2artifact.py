r"""
Self-supervised MRI reconstruction with Artifact2Artifact
=========================================================

We demonstrate the self-supervised Artifact2Artifact loss for solving an
undersampled sequential MRI reconstruction problem without ground truth.

The Artifact2Artifact loss was introduced in Liu et al. `RARE: Image
Reconstruction using Deep Priors Learned without
Groundtruth <https://ieeexplore.ieee.org/document/9103213>`__.

In our example, we use it to reconstruct **static** images, where the
k-space measurements is a time-sequence, where each time step (phase)
consists of sampled lines such that the whole measurement is a set of
non-overlapping lines.

For a description of how Artifact2Artifact constructs the loss, see
:class:`deepinv.loss.Artifact2ArtifactLoss`.

Note in our implementation, this is a special case of the generic
splitting loss: see :class:`deepinv.loss.SplittingLoss` for more
details. See :class:`deepinv.loss.Phase2PhaseLoss` for the related
Phase2Phase.

"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

import deepinv as dinv
from deepinv.utils.demo import load_dataset, demo_mri_model
from deepinv.models.utils import get_weights_url
from deepinv.physics.generator import (
    GaussianMaskGenerator,
    BernoulliSplittingMaskGenerator,
)

torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load data
# ---------
#
# We use a subset of single coil knee MRI data from the fastMRI challenge
# and resize to 128x128. We use a train set of size 5 in this demo for
# speed to fine-tune the original model. Set to 150 to train the original
# model from scratch.
#

batch_size = 1
H = 128

transform = transforms.Compose([transforms.Resize(H)])

train_dataset = load_dataset("fastmri_knee_singlecoil", transform, train=True)
test_dataset = load_dataset("fastmri_knee_singlecoil", transform, train=False)

train_dataset = Subset(train_dataset, torch.arange(5))
test_dataset = Subset(test_dataset, torch.arange(30))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
# Define physics
# --------------
#
# We simulate a sequential k-space sampler, that, over the course of 4
# phases (i.e. frames), samples 64 lines (i.e 2x total undersampling from
# 128) with Gaussian weighting (plus a few extra for the ACS signals in
# the centre of the k-space). We use
# :class:`deepinv.physics.SequentialMRI` to do this.
#
# First, we define a static 2x acceleration mask that all measurements use
# (of shape [B,C,H,W]):
#

mask_full = GaussianMaskGenerator((2, H, H), acceleration=2, device=device).step(
    batch_size=batch_size
)["mask"]


# %%
# Next, we randomly share the sampled lines across 4 time-phases into a
# time-varying mask:
#

# Split only in horizontal direction
masks = [mask_full[..., 0, :]]
splitter = BernoulliSplittingMaskGenerator((2, H), split_ratio=0.5, device=device)

acs = 10

# Split 4 times
for _ in range(2):
    new_masks = []
    for m in masks:
        m1 = splitter.step(batch_size=batch_size, input_mask=m)["mask"]
        m2 = m - m1
        m1[..., H // 2 - acs // 2 : H // 2 + acs // 2] = 1
        m2[..., H // 2 - acs // 2 : H // 2 + acs // 2] = 1
        new_masks.extend([m1, m2])
    masks = new_masks

# Merge masks into time dimension
mask = torch.stack(masks, 2)

# Convert to vertical lines
mask = torch.stack([mask] * H, -2)


# %%
# Now define physics using this time-varying mask of shape [B,C,T,H,W]:
#

physics = dinv.physics.SequentialMRI(mask=mask)


# %%
# Let's visualise the sequential measurements using a sample image (run
# this notebook yourself to display the video). We also visualise the
# frame-by-frame no-learning zero-filled reconstruction.
#

x = next(iter(train_dataloader))
y = physics(x)
dinv.utils.plot_videos(
    [physics.repeat(x, mask), y, mask, physics.A_adjoint(y, keep_time_dim=True)],
    titles=["x", "y", "mask", "x_init"],
    display=True,
)


# %%
# Also visualise the flattened time-series, recovering the original 2x
# undersampling mask (note the actual undersampling factor is much lower
# due to ACS lines):
#

dinv.utils.plot(
    [x, physics.average(y), physics.average(mask), physics.A_adjoint(y)],
    titles=["x", "y", "orig mask", "x_init"],
)

print("Total acceleration:", (2 * 128 * 128) / mask.sum())


# %%
# Define model
# ------------
#
# As a (static) reconstruction network, we use an unrolled network
# (half-quadratic splitting) with a trainable denoising prior based on the
# DnCNN architecture as an example of a model-based deep learning architecture
# from `MoDL <https://ieeexplore.ieee.org/document/8434321>`_.
# See :meth:`deepinv.utils.demo.demo_mri_model` for details.
#

model = demo_mri_model(device=device)


# %%
# Prep loss
# ---------
#
# Perform loss on all collected lines by setting ``dynamic_model`` to
# False. Then adapt model to perform Artifact2Artifact. We set
# ``split_size=1`` to mean that each Artifact chunk containes only 1
# frame.
#

loss = dinv.loss.Artifact2ArtifactLoss(
    (2, 4, H, H), split_size=1, dynamic_model=False, device=device
)
model = loss.adapt_model(model)


# %%
# Train model
# -----------
#
# Original model is trained for 100 epochs. We demonstrate loading the
# pretrained model then fine-tuning with 1 epoch. Report PSNR and SSIM. To
# train from scratch, simply comment out the model loading code and
# increase the number of epochs.
#

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

# Load pretrained model
file_name = "demo_artifact2artifact_mri.pth"
url = get_weights_url(model_name="measplit", file_name=file_name)
ckpt = torch.hub.load_state_dict_from_url(
    url, map_location=lambda storage, loc: storage, file_name=file_name
)

model.load_state_dict(ckpt["state_dict"])
optimizer.load_state_dict(ckpt["optimizer"])

# Initialize the trainer
trainer = dinv.Trainer(
    model,
    physics=physics,
    epochs=1,
    losses=loss,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    metrics=[dinv.metric.PSNR(), dinv.metric.SSIM()],
    online_measurements=True,
    device=device,
    save_path=None,
    verbose=True,
    wandb_vis=False,
    show_progress_bar=False,
)

model = trainer.train()


# %%
# Test the model
# ==============
#

trainer.plot_images = True
trainer.test(test_dataloader)
