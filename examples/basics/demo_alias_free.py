r"""
Equivariant UNets for Image Inpainting
====================================================================================================

This is a simple example showing how to use equivariant UNets and verify that they are equivariant, unlike the standard UNet models.

We use pre-trained weights on an inpainting task but the models can be trained on any other task. The pre-trained weights are available at https://huggingface.co/jscanvic/deepinv/tree/main/demo_alias_free.
"""

import deepinv as dinv
import torch
import torchmetrics
from torchvision.transforms import InterpolationMode

import random

device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# %%
# Load the model
# ----------------------------------------------------------------------------------------
#
# We load pre-trained weights for the equivariant UNet and for the standard UNet as well.
#

models = {
    "AliasFreeUNet": dinv.models.EquivariantUNet(
        in_channels=3,
        out_channels=3,
        scales=5,
        rotation_equivariant=False,
    ),
    "UNet": dinv.models.UNet(
        in_channels=3,
        out_channels=3,
        scales=5,
        bias=False,
    ),
}

for model in models.values():
    model.eval()
    model.to(device)

print("Loading the pre-trained weights")

for model in models.values():
    if isinstance(model, dinv.models.EquivariantUNet):
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
# We evaluate the performance of the models on a test image in terms of PSNR.
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

for model_name, model in models.items():
    print(f"Evaluating {model_name}")

    with torch.no_grad():
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
# We evaluate the equivariance of the models to translations, rotations, and shifts, using the PSNR between images transformed before, and after the model.
# .. math::
#
#           \text{Eq} = \mathbb E_{g} \left\[ \text{PSNR}(f_\theta(T_g x), T_g f_\theta(x)) \right\]
#


def eq_fn(model, predictor, transform, params=None):
    if params is None:
        params = transform.get_params(predictor)

    with torch.no_grad():
        im_t = transform(predictor, **params)
        im_mt = model(im_t)
        im_estimate = model(predictor)
        im_tm = transform(im_estimate, **params)

    eq = psnr_fn(im_tm, im_mt).item()

    return eq, im_mt, im_tm


for model_name, model in models.items():
    print(f"Evaluating equivariance of {model_name}")

    transform = dinv.transform.Shift()
    eq_shift, im_tm_shift, im_mt_shift = eq_fn(model, predictor, transform)

    transform = dinv.transform.Translate(rng=torch.Generator(device=device))
    eq_translate, im_tm_translate, im_mt_translate = eq_fn(model, predictor, transform)

    transform = dinv.transform.Rotate(
        interpolation_mode=InterpolationMode.BILINEAR, padding="circular"
    )
    eq_rotation, im_tm_rotation, im_mt_rotation = eq_fn(model, predictor, transform)

    print(f"Eq-Shift: {eq_shift:.1f} dB")
    print(f"Eq-Translation: {eq_translate:.1f} dB")
    print(f"Eq-Rotation: {eq_rotation:.1f} dB")

    dinv.utils.plot(
        [
            im_mt_shift,
            im_tm_shift,
            im_mt_translate,
            im_tm_translate,
            im_mt_rotation,
            im_tm_rotation,
        ],
        titles=[
            "Shifted input",
            "Shifted output",
            "Translated input",
            "Translated output",
            "Rotated input",
            "Rotated output",
        ],
        show=True,
    )
