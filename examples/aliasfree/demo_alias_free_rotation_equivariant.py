import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.physics.generator import MotionBlurGenerator
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torchvision.datasets.utils import download_and_extract_archive

import numpy
import random

from os.path import exists

torch.manual_seed(0)
torch.cuda.manual_seed(0)
numpy.random.seed(0)
random.seed(0)

gen = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    (3, 128, 128), split_ratio=0.7
)
params = gen.step(batch_size=1, seed=0)
physics = dinv.physics.Inpainting(tensor_size=(3, 128, 128))
physics.update_parameters(**params)

device = "cuda:0"
root = "Urban100"
dataset_path = f"{root}/dinv_dataset0.h5"
if not exists(dataset_path):
    dataset = dinv.datasets.Urban100HR(
        root=root,
        download=True,
        transform=Compose([ToTensor(), Resize(256), CenterCrop(128)]),
    )

    train_dataset, test_dataset = random_split(dataset, (0.8, 0.2))

    # Generate data pairs x,y offline using a physics generator
    dinv.datasets.generate_dataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        physics=physics,
        save_dir=root,
        batch_size=1,
        device=device,
    )

torch.manual_seed(0)
torch.cuda.manual_seed(0)
numpy.random.seed(0)
random.seed(0)

train_dataloader = DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=True), shuffle=True
)
test_dataloader = DataLoader(
    dinv.datasets.HDF5Dataset(dataset_path, train=False), shuffle=False
)

model = dinv.models.AliasFreeDenoiser(
    in_channels=3,
    out_channels=3,
    scales=5,
    rotation_equivariant=True,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

losses = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
]

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=20,
    losses=losses,
    optimizer=optimizer,
    scheduler=scheduler,
    verbose=True,
    show_progress_bar=False,
    save_path=None,
    device=device,
    no_learning_method="y",
)

model = trainer.train()

trainer.plot_images = True
trainer.test(test_dataloader)

transform = dinv.transform.Shift()

x, y = next(iter(test_dataloader))
x = x.to(device)
y = y.to(device)
params = {"x_shift": [64], "y_shift": [64]}
dinv.utils.plot(
    [
        x,
        y,
        model(y),
        transform(x, **params),
        transform(y, **params),
        transform(model(y), **params),
    ],
    ["GT", "Noisy", "Denoised", "Shifted GT", "Shifted Noisy", "Shifted Denoised"],
)

equiv_psnr = transform.equivariance_test(model, y, metric=dinv.loss.metric.PSNR())
print(f"Equivariance test (Shift, PSNR): {equiv_psnr.item():.1f} dB")

transform = dinv.transform.Rotate()

x, y = next(iter(test_dataloader))
x = x.to(device)
y = y.to(device)
params = {"theta": [15]}
dinv.utils.plot(
    [
        x,
        y,
        model(y),
        transform(x, **params),
        transform(y, **params),
        transform(model(y), **params),
    ],
    ["GT", "Noisy", "Denoised", "Rotated GT", "Rotated Noisy", "Rotated Denoised"],
)

equiv_psnr = transform.equivariance_test(model, y, metric=dinv.loss.metric.PSNR())
print(f"Equivariance test (Rotation, PSNR): {equiv_psnr.item():.1f} dB")
