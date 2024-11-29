import deepinv as dinv
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm

import numpy
import random
from datetime import datetime

import os

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

batch_size = 4

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

eval_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
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

loss_fn = nn.MSELoss()

# Training loop
epochs = 50
model.train()

# Training metrics
metrics = {}
metrics_fmt_map = {}
def metric_update(metric_name, value, fmt=None):
    # Register the metric once.
    if metric_name not in metrics:
        metrics[metric_name] = torchmetrics.MeanMetric()

    # Register its format once.
    if metric_name not in metrics_fmt_map:
        metrics_fmt_map[metric_name] = fmt
    # Catch attempts to change the format.
    else:
        assert metrics_fmt_map[metric_name] == fmt, "The format cannot change."

    if isinstance(value, torch.Tensor):
        value = value.item()
    metrics[metric_name].update(value)

for epoch in range(1, epochs+1):
    # Reset the metrics
    for metric in metrics.values():
        metric.reset()

    for i, (target, predictor) in enumerate(train_dataloader):
        target = target.to(device)
        predictor = predictor.to(device)

        optimizer.zero_grad()
        estimate = model(predictor)
        training_loss = loss_fn(estimate, target)
        training_loss.backward()
        optimizer.step()

        metric_update("Training loss", training_loss, fmt=".3e")

    for i, (target, predictor) in enumerate(eval_dataloader):
        target = target.to(device)
        predictor = predictor.to(device)

        with torch.no_grad():
            estimate = model(predictor)
            test_loss = loss_fn(estimate, target)

            metric_update("Eval loss", test_loss, fmt=".3e")

    if scheduler is not None:
        scheduler.step()

    epochs_ndigits = len(str(int(epochs)))
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    segments = [
        "",
        current_timestamp,
        f"[{epoch:{epochs_ndigits}d}/{epochs}]",
    ]

    for key, metric in metrics.items():
        fmt = metrics_fmt_map.get(key, None)
        if fmt is not None:
            value = metric.compute().item()
            segments.append(f"{key}: {value:{fmt}}")

    print("\t".join(segments))

# Save the final weights.
weights = model.state_dict()
path = f"{out_dir}/weights.pt"
os.makedirs(os.path.dirname(path), exist_ok=True)
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_image(im_m, path)

        for transform_dirname, transform in transforms_map.items():
            # Sample a random transform.
            params = transform.get_params(predictor)

            im_t = transform(predictor, **params)
            path = f"{out_dir}/{split_name}/{transform_dirname}/t/{i}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(im_t, path)

            im_mt = model(im_t)
            path = f"{out_dir}/{split_name}/{transform_dirname}/mt/{i}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(im_mt, path)

            im_tm = transform(im_m, **params)
            path = f"{out_dir}/{split_name}/{transform_dirname}/tm/{i}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_image(im_tm, path)
