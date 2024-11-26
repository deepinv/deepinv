import deepinv as dinv
from deepinv.loss import adversarial
from deepinv.physics.generator import MotionBlurGenerator
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

import numpy
import random

from os.path import exists, dirname
from os import makedirs

device = "cuda:0"
dataset_root = "Urban100"
dataset_path = f"{dataset_root}/dinv_dataset0.h5"
# model_kind = "AliasFreeUNet"
model_kind = "UNet"
rotation_equivariant = False
# out_dir = "results/Inpainting_AliasFreeUNet"
out_dir = "results/Inpainting_UNet"

torch.manual_seed(0)
torch.cuda.manual_seed(0)
numpy.random.seed(0)
random.seed(0)

# Step 0. Print the configuration.

print()
print("Configuration:")
print(f"  Device: {device}")
print(f"  Dataset root: {dataset_root}")
print(f"  Dataset path: {dataset_path}")
print(f"  Model kind: {model_kind}")
print(f"  Rotation equivariant: {rotation_equivariant}")
print(f"  Out dir: {out_dir}")
print()

# Step 1. Synthesize the dataset.

dataset = dinv.datasets.Urban100HR(
    root=dataset_root,
    download=True,
    transform=Compose([ToTensor(), Resize(256), CenterCrop(128)]),
)

gen = dinv.physics.generator.BernoulliSplittingMaskGenerator(
    (3, 128, 128), split_ratio=0.7
)
params = gen.step(batch_size=1, seed=0)
physics = dinv.physics.Inpainting(tensor_size=(3, 128, 128))
physics.update_parameters(**params)

train_dataset, test_dataset = random_split(dataset, (0.8, 0.2))

dinv.datasets.generate_dataset(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    physics=physics,
    save_dir=dataset_root,
    batch_size=1,
    device="cpu",
)

physics.mask.to(device)

# Step 2. Load the dataset.

train_dataset = dinv.datasets.HDF5Dataset(dataset_path, train=True)
test_dataset = dinv.datasets.HDF5Dataset(dataset_path, train=False)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)

test_dataloader = DataLoader(
    test_dataset,
)

# Step 3. Train the model.

if model_kind == "AliasFreeUNet":
    model = dinv.models.AliasFreeUNet(
        in_channels=3,
        out_channels=3,
        scales=5,
        rotation_equivariant=rotation_equivariant,
    )
elif model_kind == "UNet":
    model = dinv.models.UNet(
        in_channels=3,
        out_channels=3,
        scales=5,
    )
else:
    raise ValueError(f"Unknown model kind: {model_kind}")

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = None

losses = [
    dinv.loss.SupLoss(metric=torch.nn.MSELoss()),
]

trainer = dinv.Trainer(
    model=model,
    physics=physics,
    train_dataloader=train_dataloader,
    eval_dataloader=test_dataloader,
    epochs=50,
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

# Save the final weights.
weights = model.state_dict()
path = f"{out_dir}/weights.pt"
makedirs(dirname(path), exist_ok=True)
torch.save(weights, path)

# Step 4. Evaluate the model.

model.eval()

splits_map = {
    "train": train_dataset,
    "test": test_dataset,
}

transforms_map = {
    "shifts": dinv.transform.Shift(),
    "rotations": dinv.transform.Rotate(
        interpolation_mode=InterpolationMode.BILINEAR,
        padding="circular"
    ),
}

for split_name, dataset in splits_map.items():
    for i, (target, predictor) in tqdm(enumerate(dataset)):
        target = target.to(device).unsqueeze(0)
        predictor = predictor.to(device).unsqueeze(0)

        im_m = model(predictor)
        path = f"{out_dir}/{split_name}/m/{i}.png"
        makedirs(dirname(path), exist_ok=True)
        save_image(im_m, path)

        for transform_dirname, transform in transforms_map.items():
            # Sample a random transform.
            params = transform.get_params(predictor)

            im_t = transform(predictor, **params)
            path = f"{out_dir}/{split_name}/{transform_dirname}/t/{i}.png"
            makedirs(dirname(path), exist_ok=True)
            save_image(im_t, path)

            im_mt = model(im_t)
            path = f"{out_dir}/{split_name}/{transform_dirname}/mt/{i}.png"
            makedirs(dirname(path), exist_ok=True)
            save_image(im_mt, path)

            im_tm = transform(im_m, **params)
            path = f"{out_dir}/{split_name}/{transform_dirname}/tm/{i}.png"
            makedirs(dirname(path), exist_ok=True)
            save_image(im_tm, path)
