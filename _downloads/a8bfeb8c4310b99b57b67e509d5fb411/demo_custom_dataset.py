"""
Bring your own dataset
=======================

This example shows how to use DeepInverse with your own dataset.

A dataset in DeepInverse can consist of optional ground-truth images `x`, measurements `y`, or
:ref:`physics parameters <parameter-dependent-operators>` `params`, or any combination of these.

See :ref:`datasets user guide <datasets>` for the formats we expect data to be returned in
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
#    :class:`deepinv.datasets.ImageFolder` can load any type of data (e.g. MRI, CT, etc.)
#    by passing in a custom `loader` function and `transform`.

DATA_DIR = dinv.utils.demo.get_data_home() / "demo_custom_dataset"
dinv.utils.download_example("butterfly.png", DATA_DIR / "GT")

dataset1 = dinv.datasets.ImageFolder(DATA_DIR / "GT", transform=ToTensor())

# Load one image from dataset
x = next(iter(DataLoader(dataset1)))

dinv.utils.plot({"x": x})

# %%
# You have a folder of paired ground truth and measurements
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Now imagine we have a ground truth folder with a butterfly, and a measurements folder
# with a masked butterfly.

dinv.utils.download_example("butterfly_masked.png", DATA_DIR / "Measurements")

dataset2 = dinv.datasets.ImageFolder(
    DATA_DIR, x_path="GT/*.png", y_path="Measurements/*.png", transform=ToTensor()
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
    DATA_DIR, y_path="Measurements/*.png", transform=ToTensor()
)

x, y = next(iter(DataLoader(dataset3)))
print(x)

# %%
# You already have tensors
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Sometimes you might already have tensor(s). You can construct a dataset using
# :class:`deepinv.datasets.TensorDataset`, for example here an unsupervised dataset
# containing just a single measurement (and will be loaded in as a tuple `(nan, y)`):

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
# You should check that your dataset is compatible using :func:`deepinv.datasets.check_dataset`
# (alternatively inherit from :class:`deepinv.datasets.ImageDataset` and use `self.check_dataset()`):

dinv.datasets.check_dataset(dataset5)

# %%
# Part 2: Using your dataset with DeepInverse
# -------------------------------------------

# %%
# Say you have a DeepInverse problem already set up:

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
physics = dinv.physics.Inpainting(img_size=(3, 256, 256))
model = dinv.models.RAM(pretrained=True, device=device)

# %%
# If your dataset already returns measurements in the form `(x, y)` or `(x, y, params)`,
# you can directly test with it.
#
# Our physics does not yet know the `params` (here, the inpainting mask). Since it is fixed
# across the dataset, we can define it manually by estimating it from y:
#
# .. note::
#
#    If you're loading measurements which have randomly varying `params`, your dataset must
#    return tuples `(x, y, params)` so that the physics is modified accordingly every image.

params = {"mask": (dataset2[0][1].to(device) != 0).float()}
physics.update(**params)

dinv.test(model, DataLoader(dataset2), physics, plot_images=True, device=device)

# %%
# Even if the dataset doesn't have ground truth:
#
# Here reference-metrics such as PSNR will give NaN due to lack of ground truth, but
# no-reference metrics can be used.

metrics = [dinv.metric.PSNR(), dinv.metric.NIQE(device=device)]

dinv.test(
    model,
    DataLoader(dataset3),
    physics,
    plot_images=True,
    metrics=metrics,
    device=device,
)

# %%
# Generating measurements
# -----------------------
# If your dataset returns only ground-truth `x`, you can generate a dataset of measurements using
# :func:`deepinv.datasets.generate_dataset`:

path = dinv.datasets.generate_dataset(
    dataset1, physics, save_dir=DATA_DIR / "measurements", device=device
)
dinv.test(
    model,
    DataLoader(dinv.datasets.HDF5Dataset(path)),
    physics,
    plot_images=True,
    device=device,
)

# %%
# .. tip::
#
#    Pass in a :ref:`physics generator <physics_generators>` to simulate random physics and then use
#    `load_physics_generator_params=True` to load these `params` alongside the data during testing.

# %%
# If you don't want to generate a dataset offline, you can also generate measurements online
# ("on-the-fly") during testing or training:

dinv.test(
    model,
    DataLoader(dataset1),
    physics,
    plot_images=True,
    device=device,
    online_measurements=True,
)

# %%
# 🎉 Well done, you now know how to use your own dataset with DeepInverse!
#
# What's next?
# ~~~~~~~~~~~~
# * Check out :ref:`the example on how to test a state-of-the-art general pretrained model <sphx_glr_auto_examples_basics_demo_pretrained_model.py>` with your new dataset.
# * Check out the :ref:`example on how to fine-tune a foundation model <sphx_glr_auto_examples_models_demo_foundation_model.py>` to your own data.
# * Check out the :ref:`example on how to train a reconstruction model <sphx_glr_auto_examples_models_demo_training.py>` with your dataset.
# * Advanced: how to :ref:`stream or download a dataset from HuggingFace <sphx_glr_auto_examples_external-libraries_demo_hf_dataset.py>`.
