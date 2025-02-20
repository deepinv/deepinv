"""
Using huggingface dataset
====================================================================================================

| This simple example shows how to load and prepare properly a huggingface dataset.
| Context: having a quick access to several datasets available under the huggingface format.
| Available datasets: https://huggingface.co/datasets?search=deepinv

| Here we use `drunet_dataset <https://github.com/samuro95/GSPnP>`_.
"""

# %%
# Load libraries
# ----------------------------------------------------------------------------------------
#

from datasets import load_dataset as load_dataset_hf
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

import deepinv as dinv


# %%
# Stream data from Internet
# ----------------------------------------------------------------------------------------
#
# Stream data from huggingface servers: only a limited number of samples is loaded on memory at all time,
# which avoid having to save the dataset on disk and avoid overloading the memory capacity.
#

# source : https://huggingface.co/datasets/deepinv/drunet_dataset
# type : datasets.iterable_dataset.IterableDataset
raw_hf_train_dataset = load_dataset_hf(
    "deepinv/drunet_dataset", split="train", streaming=True
)
print("Number of data files used to store raw data: ", raw_hf_train_dataset.n_shards)

# %%
# Shuffle data with buffer shuffling
# ----------------------------------------------------------------------------------------
#
# | In streaming mode, we can only read sequentially the data sample in a certain order thus we are not able to do exact shuffling.
# | An alternative way is the buffer shuffling which load a fixed number of samples in memory and let us pick randomly one sample among this fixed number of samples.
#

# https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
raw_hf_train_dataset = raw_hf_train_dataset.shuffle(seed=42, buffer_size=100)


# %%
# Apply transformation on dataset
# ----------------------------------------------------------------------------------------
#
# We define transformation with ``torchvision.transforms`` module, but it can be any other function.
#

# Function that should be applied to a PIL Image
img_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
    ]
)


# Class that apply `transform` on data samples of a datasets.iterable_dataset.IterableDataset
class HFDataset(IterableDataset):
    r"""
    Creates an iteratble dataset from a Hugging Face dataset to enable streaming.
    """

    def __init__(self, hf_dataset, transforms=None, key="png"):
        self.hf_dataset = hf_dataset
        self.transform = transforms
        self.key = key

    def __iter__(self):
        for sample in self.hf_dataset:
            if self.transform:
                out = self.transform(sample[self.key])
            else:
                out = sample[self.key]
            yield out


hf_train_dataset = HFDataset(raw_hf_train_dataset, transforms=img_transforms)


# %%
# Create a dataloader
# --------------------------------------------------------------------
#
# | With ``datasets.iterable_dataset.IterableDataset``, data samples are stored in 1 file or in a few files.
# | In case of few files, we can use ``num_workers`` argument to load data samples in parallel.
#

if raw_hf_train_dataset.n_shards > 1:
    # num_workers <= raw_hf_train_dataset.n_shards (number of data files)
    # num_workers <= number of available cpu cores
    num_workers = ...
    train_dataloader = DataLoader(
        hf_train_dataset, batch_size=2, num_workers=num_workers
    )
else:
    train_dataloader = DataLoader(hf_train_dataset, batch_size=2)

# display a batch
batch = next(iter(train_dataloader))
dinv.utils.plot([batch[0], batch[1]])
