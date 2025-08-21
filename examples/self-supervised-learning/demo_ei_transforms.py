r"""
Image transformations for Equivariant Imaging
=============================================

This example demonstrates various geometric image transformations
implemented in ``deepinv`` that can be used in Equivariant Imaging (EI)
for self-supervised learning:

-  Shift: integer pixel 2D shift;
-  Rotate: 2D image rotation;
-  Scale: continuous 2D image downscaling;
-  Euclidean: includes continuous translation, rotation, and reflection,
   forming the group :math:`\mathbb{E}(2)`;
-  Similarity: as above but includes scale, forming the group
   :math:`\text{S}(2)`;
-  Affine: as above but includes shear effects, forming the group
   :math:`\text{Aff}(3)`;
-  Homography: as above but includes perspective (i.e pan and tilt)
   effects, forming the group :math:`\text{PGL}(3)`;
-  PanTiltRotate: pure 3D camera rotation i.e pan, tilt and 2D image
   rotation.

See :ref:`docs <transform>` for full list.

These were proposed in the papers:

-  ``Shift``, ``Rotate``: :footcite:t:`chen2021equivariant`.
-  ``Scale``: :footcite:t:`scanvic2025scale`.
-  ``Homography`` and the projective geometry framework: :footcite:t:`wang2024perspective`.

"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize

import deepinv as dinv
from deepinv.utils.demo import get_data_home

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

ORIGINAL_DATA_DIR = get_data_home() / "Urban100"


# %%
# Define transforms. For the transforms that involve 3D camera rotation
# (i.e pan or tilt), we limit ``theta_max`` for display.
#

transforms = [
    dinv.transform.Shift(),
    dinv.transform.Rotate(),
    dinv.transform.Scale(),
    dinv.transform.Homography(theta_max=10),
    dinv.transform.projective.Euclidean(),
    dinv.transform.projective.Similarity(),
    dinv.transform.projective.Affine(),
    dinv.transform.projective.PanTiltRotate(theta_max=10),
]


# %%
# Plot transforms on a sample image. Note that, during training, we never
# have access to these ground truth images ``x``, only partial and noisy
# measurements ``y``.
#

x = dinv.utils.load_example("celeba_example.jpg")
dinv.utils.plot(
    [x] + [t(x) for t in transforms],
    ["Orig"] + [t.__class__.__name__ for t in transforms],
    fontsize=24,
)


# %%
# Now, we run an inpainting experiment to reconstruct images from images
# masked with a random mask, without ground truth, using EI. For this
# example we use the Urban100 images of natural urban scenes. As these
# scenes are imaged with a camera free to move and rotate in the world,
# all of the above transformations are valid invariances that we can
# impose on the unknown image set :math:`x\in X`.
#

dataset = dinv.datasets.Urban100HR(
    root=ORIGINAL_DATA_DIR,
    download=True,
    transform=Compose([ToTensor(), Resize(256), CenterCrop(256)]),
)

train_dataset, test_dataset = random_split(dataset, (0.8, 0.2))

train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset)

# Use physics to generate data online
physics = dinv.physics.Inpainting((3, 256, 256), mask=0.6, device=device)


# %%
# For training, use a small UNet, Adam optimizer, EI loss with homography
# transform, and the ``deepinv.Trainer`` functionality:
#
# .. note::
#
#       We only train for a single epoch in the demo, but it is recommended to train multiple epochs in practice.
#

model = dinv.models.UNet(
    in_channels=3, out_channels=3, scales=2, circular_padding=True, batch_norm=False
).to(device)

losses = [
    dinv.loss.MCLoss(),
    dinv.loss.EILoss(dinv.transform.Homography(theta_max=10, device=device)),
]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

model = dinv.Trainer(
    model=model,
    physics=physics,
    online_measurements=True,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=1,
    losses=losses,
    optimizer=optimizer,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
).train()


# %%
# Show results of a pretrained model trained using a larger UNet for 40
# epochs:
#

model = dinv.models.UNet(
    in_channels=3, out_channels=3, scales=3, circular_padding=True, batch_norm=False
).to(device)

ckpt = torch.hub.load_state_dict_from_url(
    dinv.models.utils.get_weights_url("ei", "Urban100_inpainting_homography_model.pth"),
    map_location=device,
)

model.load_state_dict(ckpt["state_dict"])

x = next(iter(train_dataloader))
x = x.to(device)
y = physics(x)
x_hat = model(y)

dinv.utils.plot([x, y, x_hat], ["x", "y", "reconstruction"])

# %%
# :References:
#
# .. footbibliography::
