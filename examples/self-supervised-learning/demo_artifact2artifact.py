r"""
Self-supervised MRI reconstruction with Artifact2Artifact, Phase2Phase and SSDU
===============================================================================

TODO description

"""

import deepinv as dinv
from torch.utils.data import DataLoader, Subset
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset, demo_mri_model
from deepinv.models.utils import get_weights_url
from deepinv.physics.generator import (
    GaussianMaskGenerator,
    BernoulliSplittingMaskGenerator,
)

torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


# %%
# Prep data
# ---------
#
# Original model trained with a dataset of 150 samples. We use 5 here in
# this demo to fine-tune the original model. Set to 150 to train model
# from scratch.
#

batch_size = 1
H = 128

transform = transforms.Compose([transforms.Resize(H)])

train_dataset = load_dataset(
    "fastmri_knee_singlecoil", Path("."), transform, train=True
)
test_dataset = load_dataset(
    "fastmri_knee_singlecoil", Path("."), transform, train=False
)

train_dataset = Subset(train_dataset, torch.arange(5))
test_dataset = Subset(test_dataset, torch.arange(30))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# %%
# Prep physics
# ------------
#
# Simulate a sequential k-space sampler, that, over the course of 4
# phases, samples 64 lines (i.e. 2x total undersampling) with Gaussian
# weighting (plus a few extra for the ACS signals in the centre of the
# k-space)
#

x = next(iter(train_dataloader))

mask_full = GaussianMaskGenerator((2, H, H), acceleration=2, device=device).step(
    batch_size=batch_size
)["mask"]

full_physics = dinv.physics.MRI(mask=mask_full)
y_full = full_physics(x)
dinv.utils.plot([x, mask_full, y_full, full_physics.A_adjoint(y_full)])

# Split only in horizontal direction
masks = [mask_full[..., 0, :]]
splitter = BernoulliSplittingMaskGenerator((2, H), split_ratio=0.5, device=device)

acs = 10

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

physics = dinv.physics.SequentialMRI(mask=mask)

y = physics(x)

physics.A_adjoint(y, keep_time_dim=True).shape

print(x.shape, y.shape, mask.shape)
dinv.utils.plot_videos(
    [physics.repeat(x, mask), y, mask, physics.A_adjoint(y, keep_time_dim=True)],
    display=True,
)

# just for debug
dinv.utils.plot([x, physics.average(y), physics.average(mask), physics.A_adjoint(y)])

# Total acceleration
print((2 * 128 * 128) / mask.sum())


# %%
# Prep model
# ----------
#
# As a reconstruction network, we use an unrolled network (half-quadratic
# splitting) with a trainable denoising prior based on the DnCNN
# architecture. See ``demo`` for details TODO
#

model = demo_mri_model(device=device)


# %%
# Prep loss
# ---------
#
# Perform loss on all collected spokes by setting dynamic_model to False.
# Then adapt model to perform Artifact2Artifact
#

loss = dinv.loss.Artifact2ArtifactLoss(
    (2, 4, H, H), split_size=1, dynamic_model=False, device=device
)

model = loss.adapt_model(model)


# %%
# Train
# -----
#
# Original model trained for 100 epochs. We demonstrate fine-tuning with 1
# epoch for speed. Report PSNR and SSIM.
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
    show_progress_bar=False,  # disable progress bar for better vis in sphinx gallery.
)

model = trainer.train()

# torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, "a2a_model.pth")

trainer.test(test_dataloader)


# %%
# -  TODO remove metrics.mse() and consolidate utils.metric and
#    normalisations and maxpixel variants
# -  EI example using model from demo
# -  Compare P2P and SSDU too
# -  Demo intros
# -  Quarto render and show Mike and Julian
# -  Extra docs
# -  Check pytests
# -  TODOs in files
#
