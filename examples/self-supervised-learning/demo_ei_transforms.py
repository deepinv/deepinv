r"""
Image transformations for Equivariant Imaging
=============================================

This example demonstrates various image transformations that can be used
in Equivariant Imaging (EI) for self-supervised learning. These were
proposed in the papers:

-  ``Shift``, ``Rotate``: `Chen et al., Equivariant Imaging: Learning
   Beyond the Range
   Space <https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Equivariant_Imaging_Learning_Beyond_the_Range_Space_ICCV_2021_paper.pdf>`__
-  ``Scale``: `Scanvic et al., Self-Supervised Learning for Image
   Super-Resolution and Deblurring <https://arxiv.org/abs/2312.11232>`__
-  ``Homography`` and the projective geometry framework: `Wang et al.,
   Perspective-Equivariant Imaging: an Unsupervised Framework for
   Multispectral Pansharpening <https://arxiv.org/abs/2403.09327>`__

TODO list transforms and equations

TODO Cite my paper and Jeremy

"""

import deepinv as dinv
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torchvision.datasets.utils import download_and_extract_archive

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"


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

x = dinv.utils.load_url_image(dinv.utils.demo.get_image_url("celeba_example.jpg"))
dinv.utils.plot(
    [x] + [t(x) for t in transforms],
    ["Orig"] + [t.__class__.__name__ for t in transforms],
)


# %%
# Now, we run an inpainting experiment to reconstruct images from images
# masked with a random mask, without ground truth, using EI. For this
# example we use the Urban100 images of natural urban scenes. As these
# scenes are imaged with a camera free to move and rotate in the world,
# all of the above transformations are valid invariances that we can
# impose on the unknown image set :math:`x\in X`.
#

physics = dinv.physics.Inpainting((3, 256, 256), mask=0.3, device=device)

download_and_extract_archive(
    "https://huggingface.co/datasets/eugenesiow/Urban100/resolve/main/data/Urban100_HR.tar.gz?download=true",
    "Urban100",
    filename="Urban100_HR.tar.gz",
    md5="65d9d84a34b72c6f7ca1e26a12df1e4c",
)

train_dataset, test_dataset = random_split(
    ImageFolder(
        "Urban100", transform=Compose([ToTensor(), Resize(256), CenterCrop(256)])
    ),
    (0.8, 0.2),
)

train_dataloader, test_dataloader = DataLoader(train_dataset, shuffle=True), DataLoader(
    test_dataset
)


# %%
# For training, use a small UNet, Adam optimizer, EI loss with homography
# transform, and the ``deepinv.Trainer`` functionality:
#

model = dinv.models.UNet(
    in_channels=3, out_channels=3, scales=2, circular_padding=True, batch_norm=False
).to(device)

losses = [
    dinv.loss.MCLoss(),
    dinv.loss.EILoss(dinv.transform.Homography(theta_max=10, device=device)),
]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

ckpt = torch.load("model.pth", map_location=device)
model.load_state_dict(ckpt["state_dict"])

model = dinv.Trainer(
    model=model,
    physics=physics,
    online_measurements=True,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=50,
    losses=losses,
    optimizer=optimizer,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
).train()

torch.save({"state_dict": model.state_dict()}, "model.pth")


x, _ = next(iter(train_dataloader))
y = physics(x)
x_hat = model(y)

dinv.utils.plot([x, y, x_hat], ["x", "y", "reconstruction"])