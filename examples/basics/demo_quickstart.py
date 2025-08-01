r"""
5 minute quickstart tutorial
============================

Follow this example to get started with DeepInverse in under 5 minutes.

**Contents**

1. `Install <#install>`__
2. `Physics <#physics>`__
3. `Models <#models>`__
4. `Datasets <#datasets>`__
5. `What's next <#what-s-next>`__

"""

# %%
# 1. Install
# ~~~~~~~~~~
#
# First, install and import the latest stable release of `deepinv`:
#
# .. code:: bash
#
#    pip install deepinv
#

import deepinv as dinv


# %%
# 2. Physics
# ~~~~~~~~~~
#


# %%
# In DeepInverse, `x` are images:
#

x = dinv.utils.load_example("butterfly.png")


# %%
# :ref:`Imaging forward operators <physics_intro>` are called `physics` and simulate
# measurements `y` from `x`.
#

physics = dinv.physics.Inpainting(x.shape[1:], mask=0.8)

y = physics(x)


# %%
# DeepInverse implements
# :ref:`many different types of physics <physics>` and noise
# models across various imaging modalities.
#
# Most physics also take
# :ref:`physics parameters <parameter-dependent-operators>` such as `mask`, `filter`, `sigma` etc.
# You can easily use your own params by passing these into the `physics`,
# or you can use a `generator`` to :ref:`generate random params <physics_generators>`.
#

# Blur with Gaussian filter parameter
physics = dinv.physics.Blur(filter=dinv.physics.blur.gaussian_blur())

# MRI with random mask parameter
physics = dinv.physics.MRI(
    **dinv.physics.generator.RandomMaskGenerator(x.shape[2:]).step()
)

# Inpainting with noise model
physics = dinv.physics.Inpainting(
    x.shape[1:], mask=0.8, noise_model=dinv.physics.GaussianNoise(0.1)
)

y = physics(x)


# %%
# Physics are powerful objects and :ref:`have many methods <physics_intro>`, for example a
# pseudo-inverse:
#

x_pinv = physics.A_dagger(y)


# %%
# .. tip::
#
#    Want to use DeepInverse with your own physics operator? Check out TODO
#


# %%
# 3. Models
# ~~~~~~~~~
#
# In DeepInverse, a `model` is a reconstruction algorithm that
# **reconstructs** images from `y` and knowledge of `physics`.
#

model = dinv.models.RAM(pretrained=True)

x_net = model(y, physics)


# %%
# DeepInverse covers
# :ref:`many frameworks of reconstruction algorithms <reconstructors>`
# including plug-and-play, variational optimization, sampling algorithms
# (e.g.'diffusion models), unrolled models and foundation models. TODO
# link to each of these items in user guide
#
# :ref:`Many models are pretrained <pretrained-reconstructors>` and can
# be used out of the box. TODO tip: ref new example for how to inference / finetune pretrained model
#

# DPIR plug-and-play algorithm
model = dinv.optim.DPIR(sigma=0.1)

# Reconstruct Anything Model
model = dinv.models.RAM(pretrained=True)


# %%
# Some models are only :ref:`denoisers <denoisers>` that **denoise**
# images from `y` and `sigma`, which can be used to build certain
# reconstruction algorithms.
#

denoiser = dinv.models.DRUNet()

x_denoised = denoiser(y, sigma=0.1)

model = dinv.optim.DPIR(sigma=0.1, denoiser=denoiser)


# %%
# Plot the image `x`, the measurement `y` and the reconstructed image
# `x_net`:
#

dinv.utils.plot({"x": x, "y": y, "x_net": x_net})


# %%
# **TODO** building optimization algos/PnP/unfolded/sampling?
#

data_fidelity = dinv.optim.data_fidelity.L2()
prior = dinv.optim.prior.PnP(denoiser=dinv.models.MedianFilter())
model = dinv.optim.optim_builder(
    iteration="HQS",
    prior=prior,
    data_fidelity=data_fidelity,
    params_algo={"stepsize": 1.0, "g_param": 0.1},
)


# %%
# .. tip::
#
#    Want to use DeepInverse with your own network? Just inherit from the reconstructor base class :class:`deepinv.models.Reconstructor`!
#


# %%
# 4. Datasets
# ~~~~~~~~~~~
#
# You can use DeepInverse with :ref:`dataset <datasets>`, for testing or training. First,
# define a ground-truth dataset. We implement wrappers for
# :ref:`many popular imaging datasets <datasets>` across domains including natural images,
# medical imaging, satellite imaging etc.
#
# .. tip::
#     It's easy to use your own dataset with DeepInverse. See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py` for a tutorial.
#

dataset = dinv.datasets.SimpleFastMRISliceDataset("data", download=True)


# %%
# :ref:`Datasets <datasets>` return either `x`, tuples `x, y` or `x, y, params` of images,
# measurements, and optional physics parameters. Given a ground-truth
# dataset, you can simulate a dataset:
#
#

pth = dinv.datasets.generate_dataset(dataset, physics, save_dir="data")

dataset = dinv.datasets.HDF5Dataset(pth)


# %%
# You can use this dataset to test or train a model:
#

import torch

dinv.test(model, torch.utils.data.DataLoader(dataset), physics)


# %%
# .. tip::
#
#    Want to use DeepInverse with your own dataset? Check out :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py` for a tutorial!
#


# %%
# 🎉 Well done, you now know how to use DeepInverse!
#
# What's next?
# ~~~~~~~~~~~~
#
# -  Try basic examples, including
#    :ref:`how to inference a pretrained model <sphx_glr_auto_examples_basics_demo_pretrained_model.py>`,
#    :ref:`how to use DeepInverse with your own dataset <sphx_glr_auto_examples_basics_demo_custom_dataset.py>`, or
#    :ref:`how to use DeepInverse with your custom physics operator <sphx_glr_auto_examples_basics_demo_custom_physics.py>`.
# -  Dive deeper into our full library of examples
# -  Read the :ref:`User Guide <user_guide>` for further details on the
#    concepts introduced here.
# -  Want help?
#    `Open an issue <https://github.com/deepinv/deepinv/issues>`_ ask
#    a message on our `Discord <https://discord.gg/qBqY5jKw3p>`_ or
#    get in touch with our
#    `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.
#
