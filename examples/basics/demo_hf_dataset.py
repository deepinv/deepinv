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

from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

import deepinv as dinv


# %%
# Stream data from Internet
# ----------------------------------------------------------------------------------------
#

# stream data from huggingface
# only a limited number of samples should be in memory and nothing should be saved on disk
# https://huggingface.co/datasets/deepinv/drunet_dataset
# type : datasets.iterable_dataset.IterableDataset
raw_hf_train_dataset = load_dataset(
    "deepinv/drunet_dataset", split="train", streaming=True
)
print("Number of data files used to store raw data: ", raw_hf_train_dataset.n_shards)

# in streaming mode, we can only read sequentially the data sample in a certain order
# thus we are not able to do exact shuffling
# an alternative way is the buffer shuffling which load a fixed number of samples in memory
# and let us pick randomly one sample among this fixed number of samples
# https://huggingface.co/docs/datasets/about_mapstyle_vs_iterable
raw_hf_train_dataset = raw_hf_train_dataset.shuffle(seed=42, buffer_size=100)


# %%
# Apply transformation on dataset
# ----------------------------------------------------------------------------------------
#
# We define transformation with `torchvision.transforms` module.
# You can use any other function.
#

# Function that should be applied to a PIL Image
img_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor(),
    ]
)


# Class that apply transform on data samples of a datasets.iterable_dataset.IterableDataset
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

# Define your DataLoader
# if raw_hf_train_dataset.n_shards > 1,
# it may be interesting to define argument `num_workers > 1`,
# to have parallel processing of data samples
# of course, num_workers <= n_shards (number of data files)
#            num_workers <= number of available cpu cores
# DataLoader(hf_train_dataset, batch_size=5, num_workers=?)
train_dataloader = DataLoader(raw_hf_train_dataset, batch_size=5)

# display a batch
batch = next(iter(train_dataloader))
dinv.utils.plot(batch["png"])
