r"""
Self-supervised learning with Equivariant Splitting
===================================================

This example demonstrates how to train a reconstruction model in a fully self-supervised way using equivariant splitting :footcite:p:`sechaud26Equivariant`.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv

# %%
# Setup random seeds, paths and device
# ---------------------------------------------------------------
#

# For reproducilibity
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# Setup paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "measurements"
CKPT_DIR = BASE_DIR / "ckpts"

# Select the device
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Forward model
# -------------
#
# First, we define the forward model, here an inpainting problem with a fixed mask:
#

channels = 3
img_size = 64
physics = dinv.physics.Inpainting(mask=0.7, img_size=(channels, img_size, img_size), device=device)

# %%
# Create the imaging dataset
# --------------------------
#
# Using the forward model and a base dataset, here :class:`deepinv.datasets.Urban100HR`, we generate an imaging dataset that we further split into a 80 training samples, 10 evaluation samples and 10 test samples.
#

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
])

dataset = dinv.datasets.Urban100HR(root=".", transform=transform, download=True)
train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [80, 10, 10], generator=torch.Generator().manual_seed(0)
)

batch_size = 4
num_workers = 4 if torch.cuda.is_available() else 0

dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    test_dataset=eval_dataset,
    physics=physics,
    device=device,
    save_dir=".",
    batch_size=batch_size,
    num_workers=num_workers,
)

train_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, split="train")
eval_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, split="eval")
test_dataset = dinv.datasets.HDF5Dataset(path=dataset_path, split="test")

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Visualizing the problem
# ------------------------
#

x, y = test_dataset[0]
x, y = x.unsqueeze(0), y.unsqueeze(0)

psnr_fn = dinv.metric.PSNR()
psnr_y = psnr_fn(y, x)

dinv.utils.plot([ y, x ], ["Measurements", "Ground truth"], subtitles=[f"PSNR={psnr_y:.1f}dB", ""])

# %%
# Create the base model
# ----------------------
#
# Here, we fine tune a pretrained :class:`deepinv.models.RAM` model but any trainable reconstructor would do.
#
# In order to track the improvements brought by fine-tuning the model, we also create a copy of the pre-trained model that will not be fine-tuned.
#

model = dinv.models.RAM(pretrained=True).to(device)
model_no_learning = dinv.models.RAM(pretrained=True).to(device)

x_pretrained = model_no_learning(y, physics)

psnr_pretrained = psnr_fn(x_pretrained, x)

dinv.utils.plot(
    [ y, x_pretrained, x ],
    ["Measurement", "RAM (Pre-trained)", "Ground truth"],
    subtitles=[f"PSNR={psnr_y:.1f}dB", f"PSNR={psnr_pretrained:.1f}dB", ""],
)

# %%
# Setup the equivariant splitting loss
# ------------------------------------
#
# We create an instance of :class:`deepinv.loss.ESLoss` that implements the equivariant splitting loss.
#
# The equivariant splitting loss requires the definition of a splitting scheme similarly to :class:`deepinv.loss.SplittingLoss`. Here, we choose a pixel-wise Bernoulli splitting scheme with a split ratio of ``0.9`` using :class:`deepinv.physics.generator.BernoulliSplittingMaskGenerator`.
#
# Equivariant splitting requires choosing a set of transformations based on the forward matrix. For inpainting, valid choices include shifts, rotations and reflections :footcite:p:`sechaud26Equivariant`. Here, we choose rotations and reflections.
#
# Since the base model RAM is not already equivariant to these transformations, we use Reynolds averaging by passing in ``transform`` and ``eval_transform`` to the loss. Internally, it wraps the input model in a :class:`deepinv.models.EquivariantReconstructor` when calling ``ESLoss.adapt_model``.
#
# .. note::
#
#      The equivariant splitting loss consists in a prediction term and a consistency term. In the absence of noise, they are computed exactly using :class:`deepinv.loss.MCLoss`. In the presence of noise, they can be estimated without bias using denoising losses such as :class:`deepinv.loss.R2RLoss` and :class:`deepinv.loss.SureGaussianLoss`.
#

# Splitting scheme
mask_generator = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    img_size=(1, img_size, img_size),
    split_ratio=0.9,
    pixelwise=True,
    device=device,
)

# Underlying measurement comparison losses
consistency_loss = dinv.loss.MCLoss(metric=dinv.metric.MSE())
prediction_loss = dinv.loss.MCLoss(metric=dinv.metric.MSE())

# A random grid-preserving transformation
train_transform = dinv.transform.Rotate(
    n_trans=1, multiples=90, positive=True
) * dinv.transform.Reflect(n_trans=1, dim=[-1])
# All grid-preserving transformations
eval_transform = dinv.transform.Rotate(
    n_trans=4, multiples=90, positive=True
) * dinv.transform.Reflect(n_trans=2, dim=[-1])

es_loss = dinv.loss.ESLoss(
    mask_generator=mask_generator,
    consistency_loss=consistency_loss,
    prediction_loss=prediction_loss,
    transform=train_transform,
    eval_transform=eval_transform,
    eval_n_samples=5,
)

# Wrap the model so it takes split measurements as input and apply Reynolds averaging
model = es_loss.adapt_model(model)

# %%
# Train the model
# ---------------
#
# Starting from the pre-trained model, we fine-tune it on the imaging dataset using the equivariant splitting loss:
#

trainer = dinv.Trainer(
    model,
    physics=physics,
    epochs=10,
    ckp_interval=10,
    scheduler=None,
    losses=losses,
    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-8),
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    metrics=[dinv.metric.PSNR()],  # Supervised oracle metric for monitoring, not used for training and early stopping
    plot_images=False,
    device=device,
    verbose=True,
    show_progress_bar=False,
    no_learning_method=model_no_learning,
    early_stop=3,  # Patience parameter, stop if there is no improvement for multiple epochs
    early_stop_on_losses=True,  # Use the evaluation loss as a self-supervised stopping criterion
)

trainer.train()
trainer.load_best_model()

# %%
# Evaluation of the trained model
# --------------------------------
#
# We can now evaluate the trained model on the test set using the PSNR metric.
#
# We also compare it to the pre-trained model without fine-tuning to see the benefit of fine-tuning with equivariant splitting:

# Compute the performance metrics on the whole test set
trainer.compute_eval_losses = False
trainer.early_stop_on_losses = False
trainer.test(test_dataloader, metrics=dinv.metric.PSNR())

# Display the reconstructions for a single test sample
model.eval()
model_no_learning.eval()

with torch.no_grad():
    x_hat = model(y, physics)
    x_pretrained = model_no_learning(y, physics)

psnr = psnr_fn(x_hat, x)
psnr_pretrained = psnr_fn(x_pretrained, x)

dinv.utils.plot(
    [ y, x_pretrained, x_hat, x ],
    ["Measurement", "RAM (Pre-trained)", "Equivariant Splitting", "Ground truth"],
    subtitles=[f"PSNR={psnr_y:.1f}dB", f"PSNR={psnr_pretrained:.1f}dB", f"PSNR={psnr:.1f}dB", ""],
)

# %%
# :References:
#
# .. footbibliography::
