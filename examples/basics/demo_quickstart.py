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
# We then get the device (CPU in the case of this example).
#

import deepinv as dinv
import torch

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

# %%
# 2. Physics
# ~~~~~~~~~~
#


# %%
# In DeepInverse, `x` are images:
#

x = dinv.utils.load_example("butterfly.png", device=device)

# %%
# Images are tensors of shape `B, C, ...` where `B` is batch size, `C` are channels and `...` are spatial dimensions:

print(x.shape)

# %%
# :ref:`Imaging forward operators <physics_intro>` are called `physics` and simulate
# measurements `y` from `x`.
#

physics = dinv.physics.Inpainting(x.shape[1:], mask=0.3, device=device)

y = physics(x)


# %%
# DeepInverse implements
# :ref:`many different types of physics <physics>` across various imaging modalities.
# Physics also possess noise models such as Gaussian or Poisson noise.

physics.noise_model = dinv.physics.GaussianNoise(sigma=0.1)

y = physics(x)

dinv.utils.plot({"GT": x, "Noisy inpainting measurement": y})


# %%
# Many physics also take
# :ref:`physics parameters <parameter-dependent-operators>` such as `mask`, `filter`, `sigma` etc.:

# Blur with Gaussian filter parameter
filter = dinv.physics.blur.gaussian_blur((5, 5))

physics = dinv.physics.BlurFFT(x.shape[1:], filter=filter, device=device)

# Simulate measurements
y = physics(x)

# %%
# You can easily use your own params by passing these into the `physics`,
# or you can use a `generator` to :ref:`generate random params <physics_generators>`:

# Blur kernel random generator
physics_generator = dinv.physics.generator.MotionBlurGenerator(
    psf_size=(31, 31), num_channels=3, device=device
)

# Generate a dict of random params {"filter": ...}
params = physics_generator.step()

# Update physics during forward call
y2 = physics(x, **params)

dinv.utils.plot(
    {
        "GT": x,
        "Blurred...": y,
        "... with Gaussian kernel": filter,
        "Blurred ...": y2,
        "...with motion kernel": params["filter"],
    }
)

# %%
# Physics are powerful objects and :ref:`have many methods <physics_intro>`, for example a
# pseudo-inverse:
#

# You can also update params like so
physics.update(filter=filter.to(device))

x_pinv = physics.A_dagger(y)

# %%
# As it is well-known in the field of inverse problems, the pseudo-inverse can give good results
# if the problem is noiseless, but it completely fails in the presence of noise - this is why we need reconstructors!

physics.noise_model = dinv.physics.GaussianNoise(sigma=0.1)

y = physics(x)

x_pinv_noise = physics.A_dagger(y)

dinv.utils.plot({"Pseudoinv w/o noise": x_pinv, "Pseudoinv with noise": x_pinv_noise})


# %%
# .. tip::
#
#    Want to use DeepInverse with your own physics operator? Check out :ref:`sphx_glr_auto_examples_basics_demo_custom_physics.py` for a tutorial!
#


# %%
# 3. Models
# ~~~~~~~~~
#
# In DeepInverse, a `model` is a reconstruction algorithm that
# **reconstructs** images from `y` and knowledge of `physics`.
#
# .. tip::
#     Many models, such as :class:`Reconstruct Anything Model <deepinv.models.RAM>`, are :ref:`pretrained reconstructors <pretrained-models>` and can
#     be used out of the box. See :ref:`sphx_glr_auto_examples_basics_demo_pretrained_model.py` for a full example.
#

model = dinv.models.RAM(pretrained=True, device=device)

x_hat = model(y, physics)

# %%
# Plot the image `x`, the measurement `y` and the reconstructed image
# `x_hat` and compute :ref:`metrics <metric>`:
#

metric = dinv.metric.PSNR()

psnr_y = metric(y, x).item()
psnr_x_hat = metric(x_hat, x).item()

dinv.utils.plot(
    {
        f"Measurement\n {psnr_y:.2f} dB": y,
        f"Reconstruction\n {psnr_x_hat:.2f} dB": x_hat,
        "GT": x,
    }
)

# %%
# Some models are only :ref:`denoisers <denoisers>` that **denoise**
# images from `y` and `sigma`, which can be used to build many
# :ref:`model-based reconstruction algorithms <iterative>`.
#

denoiser = dinv.models.DRUNet(device=device)

x_denoised = denoiser(y, sigma=0.1)

model = dinv.optim.DPIR(sigma=0.1, denoiser=denoiser, device=device)

x_hat = model(y, physics)

dinv.utils.plot(
    {"Measurement": y, "Denoised": x_denoised, "Reconstructed": x_hat, "GT": x}
)

# %%
# DeepInverse covers
# :ref:`many frameworks of reconstruction algorithms <reconstructors>`
# including :ref:`deep model architectures <deep-reconstructors>`, :ref:`iterative algorithms <iterative>`, :ref:`sampling algorithms <sampling>`
# (e.g. diffusion models), and :ref:`unfolded models <unfolded>`.
#

# Reconstruct Anything Model foundation model
model = dinv.models.RAM(pretrained=True, device=device)

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
# medical imaging, satellite imaging, etc.
#
# .. tip::
#     It's easy to use your own dataset with DeepInverse. See :ref:`sphx_glr_auto_examples_basics_demo_custom_dataset.py` for a tutorial.
#

dataset = dinv.datasets.SimpleFastMRISliceDataset(
    "data", anatomy="brain", download=True
)


# %%
# :ref:`Datasets <datasets>` return either `x`, tuples `x, y` or `x, y, params` of images,
# measurements, and optional physics parameters. Given a ground-truth
# dataset, you can simulate a dataset with random physics:
#

physics = dinv.physics.MRI(device=device)

physics_generator = dinv.physics.generator.RandomMaskGenerator(
    (320, 320), device=device
)

path = dinv.datasets.generate_dataset(
    dataset,
    physics,
    save_dir="data",
    physics_generator=physics_generator,
    device=device,
)

dataset = dinv.datasets.HDF5Dataset(path, load_physics_generator_params=True)


# %%
# You can use this dataset to :ref:`test or train <trainer>` a model:
#

import torch

dinv.test(
    model,
    torch.utils.data.DataLoader(dataset),
    physics,
    plot_images=True,
    device=device,
)


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
# -  Try more basic examples, including
#    :ref:`how to inference a pretrained model <sphx_glr_auto_examples_basics_demo_pretrained_model.py>`,
#    :ref:`how to use your own dataset <sphx_glr_auto_examples_basics_demo_custom_dataset.py>`, or
#    :ref:`how to use your custom physics operator <sphx_glr_auto_examples_basics_demo_custom_physics.py>`.
# -  Dive deeper into our full library of examples.
# -  Read the :ref:`User Guide <user_guide>` for further details on the
#    concepts introduced here.
# -  Want help?
#    `Open an issue <https://github.com/deepinv/deepinv/issues>`_ ask
#    a message on our `Discord <https://discord.gg/qBqY5jKw3p>`_ or
#    get in touch with our
#    `MAINTAINERS <https://github.com/deepinv/deepinv/blob/main/MAINTAINERS.md>`_.
#
