r"""
Training a denoiser.
====================================================================================================

This example shows how to train a standard denoiser such as the [DRUNet by Zhang et al., (2020)](https://arxiv.org/abs/2008.13751)! using the :class:`deepinv.Trainer` class.


"""

# %%
import deepinv as dinv
import torch
from pathlib import Path
from torchvision import transforms
from deepinv.utils.demo import load_dataset
from torch.utils.data import DataLoader

# %%
# Setup paths for data loading and results.
# --------------------------------------------
#

BASE_DIR = Path(".")
CKPT_DIR = BASE_DIR / "ckpts"

# Set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# Load train and eval datasets
# --------------------------------------------------------------------------------------------
# In this example, we use the CBSD68 dataset for training and the set3c dataset for validation.
# We work with images of size 32x32 if no GPU is available, else 128x128.

train_dataset_name = "CBSD68"
eval_dataset_name = "set3c"
img_size = 128 if torch.cuda.is_available() else 32

train_transform = transforms.Compose(
    [transforms.RandomCrop(img_size), transforms.ToTensor()]
)
eval_transform = transforms.Compose(
    [transforms.CenterCrop((128, 128)), transforms.ToTensor()]
)

train_dataset = load_dataset(train_dataset_name, train_transform)
eval_dataset = load_dataset(eval_dataset_name, eval_transform)

num_workers = 4 if torch.cuda.is_available() else 0
train_batch_size = 32 if torch.cuda.is_available() else 1
test_batch_size = 32 if torch.cuda.is_available() else 1

# Create data loaders
train_dataloader = DataLoader(
    train_dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=test_batch_size, num_workers=num_workers, shuffle=False
)

# %%
# Set up the denoising model and physics
# --------------------------------------------------------
# We use the DRUNet as a denoiser, which is a widely used architecture for image denoising.
# We also set up the physics for denoising, which can be either Gaussian or Poisson noise.
# A standard way to train such a denoiser is to use random noise levels during training, which
# we achieve using a physics generator that samples noise levels uniformly between a minimum and maximum value.
# We use the :class:`deepinv.physics.generator.SigmaGenerator` for Gaussian noise or the :class:`deepinv.physics.generator.GainGenerator` for Poisson noise.

model = dinv.models.DRUNet(
    in_channels=3,
    out_channels=3,
    pretrained=None,
    device=device,
)

sigma_min = 0.0
sigma_max = 0.5

physics_generator = dinv.physics.generator.SigmaGenerator(
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    device=device,
)

physics = dinv.physics.Denoising(
    dinv.physics.GaussianNoise(sigma=sigma_max),
)

# %%
# Train the denoiser
# --------------------------------------------------------
# We train the model using the :class:`deepinv.Trainer` class.
# We use the Mean Squared Error (MSE) loss function, which is standard for Gaussian denoising tasks.
# We monitor the PSNR and SSIM metrics during training using :class:`deepinv.loss.metric.PSNR` and :class:`deepinv.loss.metric.SSIM`.
#
# .. note::
#
#       In this example, we only train for a few epochs to keep the training time short.
#       The number of epochs and the batch size should be based on the available hardware and the size of the dataset.
#

verbose = True  # print training information
wandb_vis = False  # plot curves and images in Weight&Bias

loss = dinv.loss.SupLoss(metric=dinv.metric.MSE())
metrics = [dinv.metric.PSNR(), dinv.metric.SSIM()]

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)
epochs = 4

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    physics_generator=physics_generator,
    online_measurements=True,
    losses=loss,
    optimizer=optimizer,
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    metrics=metrics,
    eval_interval=1,
    device=device,
    epochs=epochs,
    ckp_interval=2,
    save_path=CKPT_DIR,
)

trainer.train()

model = trainer.load_best_model()

# %%
# Test the network
# --------------------------------------------
# We can now test the trained network using the :func:`deepinv.test` function.
#
# Here the testing function will compute the PSNR and SSIM metrics on the evaluation dataset.

trainer.test(eval_dataloader)
