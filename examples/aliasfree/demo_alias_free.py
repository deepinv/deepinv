r"""
Image Inpainting with Alias-Free UNets
====================================================================================================

This is a simple example showing how to use the Alias-Free UNet models and verify that they are equivariant, unlike the standard UNet models.

We use pre-trained weights on an inpainting task but the models can be trained on any other task. The pre-trained weights are available at https://huggingface.co/jscanvic/deepinv/tree/main/demo_alias_free.
"""

import deepinv as dinv
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import (
    Compose,
    ToTensor,
    CenterCrop,
    Resize,
    InterpolationMode,
)
from tqdm import tqdm

import numpy

import csv
from datetime import datetime
import random

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
dataset_root = "Urban100"
dataset_path = f"{dataset_root}/dinv_dataset0.h5"
# model_kind = "AliasFreeUNet"
model_kind = "UNet"
rotation_equivariant = False
# out_dir = "results/Inpainting_AliasFreeUNet"
out_dir = "results/Inpainting_UNet"
epochs = 500
# batch_size = 5
batch_size = 128

torch.manual_seed(0)
torch.cuda.manual_seed(0)
numpy.random.seed(0)
random.seed(0)


# %%
# Load the model
# ----------------------------------------------------------------------------------------
#

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

model.eval()
model.to(device)

print("Loading the pre-trained weights")

if isinstance(model, dinv.models.AliasFreeUNet):
    model_name = "AliasFreeUNet"
elif isinstance(model, dinv.models.UNet):
    model_name = "UNet"
else:
    raise ValueError(f"Unknown model: {model}")
weights_url = f"https://huggingface.co/jscanvic/deepinv/resolve/main/demo_alias_free/Inpainting_{model_name}.pt"
weights = torch.hub.load_state_dict_from_url(weights_url, map_location=device)
model.load_state_dict(weights)

# %%
# Performance evaluation
# ----------------------------------------------------------------------------------------
#

psnr_fn = torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0).to(device)

im_gt = dinv.utils.demo.load_url_image(
    "https://huggingface.co/jscanvic/deepinv/resolve/main/demo_alias_free/gt.png",
    device=device,
)
predictor = dinv.utils.demo.load_url_image(
    "https://huggingface.co/jscanvic/deepinv/resolve/main/demo_alias_free/predictor.png",
    device=device,
)

im_estimate = model(predictor)

psnr = psnr_fn(im_gt, im_estimate).item()
print(f"PSNR: {psnr:.1f} dB")

dinv.utils.plot(
    [im_gt, predictor, im_estimate],
    titles=["Ground Truth", "Input", "Output"],
    show=True,
)

# %%
# Equivariance evaluation
# ----------------------------------------------------------------------------------------
#


def eq_fn(model, predictor, transform, params=None):
    if params is None:
        params = transform.get_params(predictor)

    im_t = transform(predictor, **params)
    im_mt = model(im_t)
    im_estimate = model(predictor)
    im_tm = transform(im_estimate, **params)

    eq = psnr_fn(im_tm, im_mt).item()

    return eq, im_mt, im_tm


transform = dinv.transform.Shift()
eq_shift, im_tm_shift, im_mt_shift = eq_fn(model, predictor, transform)

transform = dinv.transform.Rotate(
    interpolation_mode=InterpolationMode.BILINEAR, padding="circular"
)
eq_rotation, im_tm_rotation, im_mt_rotation = eq_fn(model, predictor, transform)

print(f"Eq-Shift: {eq_shift:.1f} dB")
print(f"Eq-Rotation: {eq_rotation:.1f} dB")

dinv.utils.plot(
    [im_mt_shift, im_tm_shift, im_mt_rotation, im_tm_rotation],
    titles=["Shifted input", "Shifted output", "Rotated input", "Rotated output"],
    show=True,
)
