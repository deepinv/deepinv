"""
Using huggingface dataset
====================================================================================================

This simple example shows how to load and prepare properly a huggingface dataset.
Context: having a quick access to several datasets available under the huggingface format.
Available datasets: https://huggingface.co/datasets?search=deepinv

Here we use drunet_dataset (https://github.com/samuro95/GSPnP)
"""

# %%
# Load libraries
# ----------------------------------------------------------------------------------------
#

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms

import deepinv as dinv


# %%
# Download dataset from Internet, save it on disk, and load from disk
# ----------------------------------------------------------------------------------------
#

DATA_DIR = "DRUNET_preprocessed"

# download from Internet
# https://huggingface.co/datasets/deepinv/drunet_dataset
dataset = load_dataset("deepinv/drunet_dataset")

# save it to disk, which is useful to avoid downloading again
dataset.save_to_disk(DATA_DIR)

# load from disk (useless here, as we already have the dataset in memory)
dataset = load_from_disk(DATA_DIR)


# %%
# Apply transformation on dataset
# ----------------------------------------------------------------------------------------
#
# We define transformation with `torchvision.transforms` module.
# You can use any other function.
#

# Define your transformations
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
    ]
)


# Define a function to apply transformations on a batch of data
def transform_images(examples):
    # before : examples['png'] = [<PIL.PngImagePlugin.PngImageFile image>, etc.]
    # after : examples['png'] = [transform(<PIL.PngImagePlugin.PngImageFile image>), etc.]
    examples["png"] = list(map(transform, examples["png"]))
    return examples


# Add function that will be applied when accessing train data (on-the-fly)
train_dataset = dataset.with_transform(transform_images)

# Define your DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# display a batch
batch = next(iter(train_dataloader))
dinv.utils.plot(batch["png"])


# %%
# Generate a dataset of degraded images and load it.
# --------------------------------------------------------------------------------
# We use a simple denoising forward operator with Gaussian noise.
#
# .. note::
#      :func:`dinv.datasets.generate_dataset` will ignore other attributes than the image,
#      e.g. the class labels if there are any.


class HF_dataset(torch.utils.data.Dataset):
    """To make the HF dataset compatible with deepinv.datasets.generate_dataset"""

    def __init__(self, hf_dataset: datasets.arrow_dataset.Dataset) -> None:
        super().__init__()
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        return self.hf_dataset[idx]["png"]


train_drunet = HF_dataset(train_dataset)

# Physic applied to degrade data
physics = dinv.physics.Denoising(dinv.physics.GaussianNoise(0.2))

# Save generated noisy dataset
dinv_dataset_path = dinv.datasets.generate_dataset(
    train_dataset=train_drunet,
    test_dataset=None,
    physics=physics,
    save_dir=DATA_DIR + "/noisy",
)

# Load noisy dataset
noisy_train_dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)

# Define your noisy DataLoader
noisy_train_dataloader = DataLoader(noisy_train_dataset, batch_size=4, shuffle=True)

# display a batch of noisy images with its associated clean images
noisy_batch = next(iter(noisy_train_dataloader))
dinv.utils.plot([noisy_batch[0], noisy_batch[1]])
