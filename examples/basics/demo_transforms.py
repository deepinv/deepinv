r"""
Image transforms for equivariance & augmentations
=================================================

We demonstrate the use of our ``deepinv.transform`` module for use in
solving imaging problems. These can be used for:

1. Data augmentation (similar to ``torchvision.transforms``)
2. Building equivariant reconstructors or denoisers
   (:class:`deepinv.models.EquivariantReconstructor` and :class:`deepinv.models.EquivariantDenoiser`) for robust reconstruction/denoising
   (e.g from :footcite:t:`terris2024equivariant`)
3. Self-supervised learning using Equivariant Imaging from :footcite:t:`chen2021equivariant`. See
   :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py`,
   :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_equivariant_imaging.py`
   for thorough examples.

See :ref:`docs <transform>` for full list of implemented transforms.

1. Data augmentation
--------------------

We can use ``deepinv`` transforms in the same way as ``torchvision``
transforms, and chain them together for data augmentation. Our
transforms are customisable and offer some group-theoretic properties.

We demonstrate a random roto-scale combined with a random masking, and a
constrained pixel-shift with a random colour jitter.
Note that all our transforms can easily be inverted using the method ``transform.inverse()``.

First, load a sample image.

"""

import torch

import deepinv as dinv
from torchvision.transforms import Compose, ColorJitter, RandomErasing, Resize

x = dinv.utils.load_example("celeba_example.jpg")

# Random roto-scale with random masking
transform = Compose(
    [
        dinv.transform.Rotate() * dinv.transform.Scale(),
        RandomErasing(),
    ]
)

# Constrained pixel-shift with a random colour jitter
transform2 = Compose(
    [
        dinv.transform.Shift(shift_max=0.2),
        ColorJitter(hue=0.5),
    ]
)

# Random diffeomorphism
transform3 = dinv.transform.CPABDiffeomorphism()

dinv.utils.plot(
    [x, transform(x), transform2(x), transform3(x)],
    titles=["Original", "Transform 1", "Transform 2", "Transform 3"],
)


# %%
# By letting ``n_trans`` be equal to the full group size, all transforms
# are recovered:
#

reflect = dinv.transform.Reflect(dim=[-2, -1], n_trans=4)
rotate = dinv.transform.Rotate(multiples=90, positive=True, n_trans=4)
dinv.utils.plot(
    [reflect(x), rotate(x)], titles=["Full 2D reflect group", "Full rotate group"]
)


# %%
# 2. Equivariant denoiser or plug-and-play
# ----------------------------------------
#
# Suppose we want to make a denoiser equivariant to the rotoreflect group,
# taken as the group product of the 90 degree rotations (order 4) and 1D reflects (order 2).
# We can do this with our transform arithmetic (note this results in the full dihedral group
# :math:`\text{Dih}_4` of order 8):
#

transform = rotate * dinv.transform.Reflect(dim=[-1], n_trans=2)


# %%
# Let's simulate some Gaussian noise and turn a simple (median filter)
# denoiser into an equivariant denoiser
# (:class:`deepinv.models.EquivariantDenoiser`):
#

sigma = 0.05
physics = dinv.physics.GaussianNoise(sigma=sigma)
x = Resize(128)(x)
y = physics(x)

model = dinv.models.DRUNet()
model_eq = dinv.models.EquivariantDenoiser(model, transform=transform, random=False)

with torch.no_grad():
    x_hat = model(y, sigma=sigma)
    x_hat_eq = model_eq(y, sigma=sigma)

dinv.utils.plot(
    [x, y, x_hat, x_hat_eq],
    [
        "Ground-truth",
        "Noisy {:.2f}dB".format(dinv.metric.PSNR()(x, y).item()),
        "Non-Eq. {:.2f}dB".format(dinv.metric.PSNR()(x, x_hat).item()),
        "Equiv. {:.2f}dB".format(dinv.metric.PSNR()(x, x_hat_eq).item()),
    ],
)


# %%
# We can also turn any reconstruction model into an equivariant counterpart
# (:class:`deepinv.models.EquivariantReconstructor`):
#
# .. math::
#    \mathcal{R}^{\text{eq}}(y, A) = \frac{1}{|\mathcal{G}|}\sum_{g\in \mathcal{G}} T_g(\mathcal{R}(y, AT_g))
#

filter = torch.zeros((1, 1, 7, 7), device=x.device)
filter[0, 0, 3, :] = 1 / 7
physics = dinv.physics.BlurFFT(
    img_size=x.shape[1:],
    filter=filter,
    device=x.device,
    noise_model=dinv.physics.GaussianNoise(sigma=sigma),
)
y = physics(x)

model = dinv.optim.DPIR(device="cpu")
model_eq = dinv.models.EquivariantReconstructor(
    model, transform=transform, random=False
)

with torch.no_grad():
    x_hat = model(y, physics)
    x_hat_eq = model_eq(y, physics)

dinv.utils.plot(
    [x, y, x_hat, x_hat_eq],
    [
        "Ground-truth",
        "Blurry {:.2f}dB".format(dinv.metric.PSNR()(x, y).item()),
        "Non-Eq. {:.2f}dB".format(dinv.metric.PSNR()(x, x_hat).item()),
        "Equiv. {:.2f}dB".format(dinv.metric.PSNR()(x, x_hat_eq).item()),
    ],
)


# %%
# What's going on under the hood? We use the ``transform.symmetrize``
# method to symmetrize the function :math:`f` with respect to a projective
# transform (with a Monte Carlo approach of ``n_trans=2`` transforms per call):
#

# Example non-equivariant function
f = lambda x: x[..., [0]] * x

# Example non-linear transform with n=2
t = dinv.transform.projective.PanTiltRotate(n_trans=2, theta_max=10, theta_z_max=0)

# Symmetrize function with respect to transform
f_s = t.symmetrize(f, average=True)
dinv.utils.plot(
    [x, f(x), f_s(x)], titles=["Original", "$f(x)$", "$\\sum_i T_i^{-1}f(T_ix)$"]
)


# %%
# 3. Equivariant imaging
# ----------------------
#
# We can also use our transforms to create the self-supervised equivariant
# imaging loss. See
# :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_ei_transforms.py`,
# :ref:`sphx_glr_auto_examples_self-supervised-learning_demo_equivariant_imaging.py`
# for examples of self-supervised learning for MRI and inpainting. For
# example, the EI loss can easily be defined using any combination of
# transforms:
#

loss = dinv.loss.EILoss(
    transform=dinv.transform.projective.Affine() | dinv.transform.projective.Euclidean()
)
