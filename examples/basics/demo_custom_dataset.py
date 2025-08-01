"""
Bring your own dataset
=======================

This example shows how to use DeepInverse with your own dataset.

A dataset in DeepInverse can consist of optional ground-truth images `x`, measurements `y`, or
:ref:`physics parameters <parameter-dependent-operators>` `params`, or any combination of these.

See :class:`deepinv.datasets.BaseDataset` for the formats we expect data to be returned in
for compatibility with DeepInverse (e.g., to be used with :class:`deepinv.Trainer`).

DeepInverse provides multiple ways of bringing your own dataset. This example has two parts:
firstly how to load images/data into a dataset, and secondly how to use this dataset with DeepInverse.
"""

import deepinv as dinv
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# %%
# Part 1: Loading data into a dataset
# -----------------------------------

# %%
# You have a folder of ground truth images
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Here we imagine we have a folder with one ground truth image of a butterfly.
#
# .. tip::
#    :class:`deepinv.datasets.ImageFolder` can load any type of data (e.g. MRI, CT etc.)
#    by passing in a custom `loader` function and `transform`.

DATA_DIR = dinv.utils.demo.get_data_home() / "demo_custom_dataset"
dinv.utils.download_example("butterfly.png", DATA_DIR / "GT")

dataset1 = dinv.datasets.ImageFolder(DATA_DIR / "GT", transform=ToTensor())
x = next(iter(DataLoader(dataset1)))

dinv.utils.plot({"x": x})

# %%
# You have a folder of paired ground truth and measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now imagine we have a ground truth folder with a butterfly, and a measurements folder
# with a masked butterfly.

dinv.utils.download_example("butterfly_masked.png", DATA_DIR / "Measurements")

dataset2 = dinv.datasets.ImageFolder(
    DATA_DIR, x_glob="GT/*.png", y_glob="Measurements/*.png", transform=ToTensor()
)

x, y = next(iter(DataLoader(dataset2)))

dinv.utils.plot({"x": x, "y": y})

# %%
# .. note::
#
#    If you're loading measurements which have randomly varying `params`, your dataset must return
#    tuples `(x, y, params)` so that the physics is modified accordingly every image.
#    We provide a convenience argument `ImageFolder(estimate_params=...)` to help you estimate these
#    `params` on the fly.

# %%
# You have a folder of only measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Imagine you have no ground truth, only measurements. Then `x` should be loaded in as NaN:

dataset3 = dinv.datasets.ImageFolder(
    DATA_DIR, y_glob="Measurements/*.png", transform=ToTensor()
)

x, y = next(iter(DataLoader(dataset3)))
print(x)

# %%
# You already have tensors
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Sometimes you might already have tensor(s). You can construct a dataset using
# :class:`deepinv.datasets.TensorDataset`:

y = dinv.utils.load_example("butterfly_masked.png")

dataset4 = dinv.datasets.TensorDataset(y=y)
x, y = next(iter(DataLoader(dataset4)))
print(x)

# %%
# You already have a PyTorch dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Say you already have your own PyTorch dataset:


class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, i):  # Returns (x, y, params)
        return torch.zeros(1), torch.zeros(1), {"mask": torch.zeros(1)}


dataset5 = MyDataset()

# %%
# You can check that it is compatible using :func:`deepinv.datasets.check_dataset`
# (alternatively you can inherit from :class:`deepinv.datasets.BaseDataset`):
dinv.datasets.check_dataset(dataset5)

# %%
# Part 2: Using your dataset with DeepInverse
# -------------------------------------------

# %%
# Say you have a DeepInverse problem already set up:

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
physics = dinv.physics.Inpainting(img_size=(3, 256, 256))
# model = dinv.models.RAM(pretrained=True, device=device)
model = dinv.models.MedianFilter()

# %%
# If your dataset already returns measurements in the form `(x, y)` or `(x, y, params)`,
# you can directly test with it.
#
# Our physics does not yet know the `params` (here, the inpainting mask). Since it is fixed
# across the dataset, we can define it manually:

y = dataset2[0][1]
params = {"mask": (y != 0).float()}
physics.update(**params)

dinv.test(model, DataLoader(dataset2), physics, plot_images=True)

# %%
# .. note::
#
#    If you're loading measurements which have randomly varying `params`, your dataset must
#    return tuples `(x, y, params)` so that the physics is modified accordingly every image.

# %%
# Even if the dataset doesn't have ground truth:

dinv.test(model, DataLoader(dataset3), physics, plot_images=True)

# %%
# Generating measurements
# -----------------------
# If your dataset returns only ground-truth `x`, you can generate a dataset of measurements using
# :func:`deepinv.datasets.generate_dataset`:

pth = dinv.datasets.generate_dataset(
    dataset1, physics, save_dir=DATA_DIR / "measurements"
)
dinv.test(model, DataLoader(dinv.datasets.HDF5Dataset(pth)), physics, plot_images=True)

# %%
# .. tip::
#
#    Pass in a :ref:`physics_generator <physics_generators>` to simulate random physics and then use
#    `load_physics_generator_params=True` to load these `params` alongside the data during testing.

# %%
# If you don't want to generate a dataset offline, you can also generate measurements online
# ("on-the-fly") during testing or training:

dinv.test(
    model, DataLoader(dataset1), physics, plot_images=True, online_measurements=True
)
