r"""
Fitting NIQE on a custom dataset
====================================================================================================

This example shows how to fit :class:`deepinv.loss.metric.NIQE` on a new dataset, and use it
to evaluate denoiser performance.

NIQE is a no-reference image quality metric that compares local image statistics
against those of pristine (distortion-free) images. Fitting NIQE on a domain-specific
dataset can better capture the expected image characteristics.

While DIV2K is also natural imaging data, the image quality and sharpness is better compared
to the dataset NIQE was originally fitted on. The fitting in this example can be done with
any dataset returning RGB or single channel tensors. Do set a reasonable denominator argument
relative to your data's pixel intensity range (e.g. for [0, 1], set denominator to 1/255)

We perform 5-fold cross-validation on the DIV2K validation set (80 fit / 20 test per fold)
and compare original NIQE weights against DIV2K-fitted weights at noise level σ=0.05.
A key finding is that the DIV2K-fitted NIQE assigns systematically higher (worse) scores
to over-smoothed outputs (e.g. large median filters), reflecting that it is more sensitive
to the loss of fine texture detail captured in the DIV2K prior.
"""

# %%
# Setup
# -----

import deepinv as dinv
from deepinv.utils import plot
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, CenterCrop, Lambda
from natsort import natsorted

device = dinv.utils.get_device()

# %%
# Define transforms and load DIV2K
# ---------------------------------
# We create two instances of DIV2K with different transforms:
# one that scales pixel values to [0, 255] for fitting NIQE weights,
# and one that keeps values in [0, 1] for denoising and PSNR evaluation.

crop_size = 1024

fit_transform = Compose(
    [
        ToTensor(),
        CenterCrop(crop_size),
        Lambda(lambda x: x * 255),
    ]
)

test_transform = Compose(
    [
        ToTensor(),
        CenterCrop(crop_size),
    ]
)

div2k_fit = dinv.datasets.DIV2K(
    root=dinv.utils.get_data_home(), mode="val", download=True, transform=fit_transform
)
div2k_fit.x_paths = natsorted(div2k_fit.x_paths)
div2k_test = dinv.datasets.DIV2K(
    root=dinv.utils.get_data_home(),
    mode="val",
    download=False,
    transform=test_transform,
)
div2k_test.x_paths = natsorted(div2k_test.x_paths)
n_images = len(div2k_fit)
all_indices = list(range(n_images))

# %%
# Define denoisers
# ----------------
# We wrap each denoiser to handle device placement and precision.
# We compare DRUNet against MedianFilter at two kernel sizes.


class Denoise:
    def __init__(self, denoiser: dinv.models.Denoiser):
        self.denoiser = denoiser

    def __call__(self, img: torch.Tensor, sigma: float) -> torch.Tensor:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                denoised = self.denoiser(img.to(device), sigma)
        return denoised.cpu()


denoisers = {
    "DRUNet": Denoise(dinv.models.DRUNet(pretrained="download", device=device)),
    "Median (k=6)": Denoise(dinv.models.MedianFilter(kernel_size=6)),
    "Median (k=9)": Denoise(dinv.models.MedianFilter(kernel_size=9)),
}

# %%
# Load original NIQE weights
# ---------------------------
# We load the original NIQE weights for comparison
# against our custom-fitted weights.

niqe_original = dinv.loss.metric.NIQE(device="cpu")

# %%
# Fit NIQE and save/load weights
# -------------------------------
# :meth:`deepinv.loss.metric.NIQE.create_weights` fits NIQE statistics from a pristine
# image dataset. Passing ``save_path`` persists the weights to disk so they can be
# reloaded in future sessions via the ``weights_path`` constructor argument.
# Here, ``save_path`` is not passed, meaning the niqe object will only be modified in-place.


def fit_niqe(fit_subset: Subset) -> dinv.loss.metric.NIQE:
    print(f"  Fitting NIQE on {len(fit_subset)} images...")
    niqe = dinv.loss.metric.NIQE(weights_path=None, device="cpu")
    niqe.create_weights(fit_subset)
    return niqe


# %%
# Run 5-fold cross-validation
# ----------------------------
# We split the 100 DIV2K validation images into 5 folds of 20 images each.
# For each fold, we fit NIQE on the remaining 80 images and evaluate on the
# held-out 20.

sigma = 0.05
fold_size = n_images // 5
results = {
    denoiser_name: {"original_niqe": [], "div2k_niqe": []}
    for denoiser_name in denoisers.keys()
}
torch.manual_seed(16 * 16)
for fold in range(5):
    print(f"Fold {fold + 1} / 5")

    test_indices = all_indices[fold * fold_size : (fold + 1) * fold_size]
    fit_indices = (
        all_indices[: fold * fold_size] + all_indices[(fold + 1) * fold_size :]
    )

    fit_subset = Subset(div2k_fit, fit_indices)
    test_subset = Subset(div2k_test, test_indices)

    niqe_fitted = fit_niqe(fit_subset)

    for i, img in enumerate(test_subset):
        img = img.unsqueeze(0)
        noisy = img + sigma * torch.randn_like(img)

        images = {}
        for name, denoiser in denoisers.items():
            images[name] = denoiser(noisy.to(device), sigma).cpu()

        for name, im in images.items():
            im_255 = im.to(torch.float32) * 255
            results[name]["original_niqe"].append(float(niqe_original(im_255)))
            results[name]["div2k_niqe"].append(float(niqe_fitted(im_255)))

# %%
# Scatter plot: original vs DIV2K-fitted NIQE
# --------------------------------------------
# Each point represents one test image, coloured by method. The x-axis shows the
# score under original weights and the y-axis shows the score under DIV2K-fitted weights.
# Points above the identity line are penalised *more* by the DIV2K prior.
#
# The median filters' NIQE score have a systematic upward shift: the DIV2K prior,
# fitted on higher-quality natural images, is more sensitive to over-smoothing and
# penalises the blurring introduced by large median filters more strongly than the
# original weights fitted on lower-quality natural images.
#
# DRUNet introduces less smoothing and has a less systematic shift.

fig, ax = plt.subplots(figsize=(9, 6))

all_orig, all_div2k = [], []
print(
    "Average relative change by utilizing DIV2K fitted NIQE instead of original NIQE:"
)
for name in denoisers.keys():
    x = np.array(results[name]["original_niqe"])
    y = np.array(results[name]["div2k_niqe"])
    avg_relative_shift = np.mean((y - x) / x)
    print(f"{name}: {float(avg_relative_shift) * 100} %")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    all_orig.append(x)
    all_div2k.append(y)
    ax.scatter(x, y, s=30, label=name, alpha=0.8)

all_orig = np.concatenate(all_orig)
all_div2k = np.concatenate(all_div2k)
lim_min = min(all_orig.min(), all_div2k.min())
lim_max = max(all_orig.max(), all_div2k.max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="identity")

ax.set_xlabel("NIQE — original weights")
ax.set_ylabel("NIQE — DIV2K-fitted weights")
ax.set_title(
    f"Per-image NIQE scores (σ = {sigma})\nPoints above the line are penalised more by the DIV2K prior"
)
ax.legend()
plt.tight_layout()
plt.show()

# %%
# Visual comparison between different denoisers
# --------------------------------------------
# Finally, we confirm visually significant blurring introduced by the median filters, not present in the ground-truth or DRUNet results

methods_all = ["gt", "noisy"] + list(denoisers.keys())
sample_img = div2k_test.__getitem__(5)[
    :, 512 - 128 : 512 + 128, 512 - 128 : 512 + 128
].unsqueeze(0)
sample_noisy = sample_img + sigma * torch.randn_like(sample_img)
images = {"gt": sample_img, "noisy": sample_noisy}
for name, denoiser in denoisers.items():
    images[name] = denoiser(sample_noisy.to(device), 0.05).cpu()
plot(
    [images[m] for m in methods_all],
    titles=methods_all,
    vmin=0,
    vmax=1,
    rescale_mode="clip",
)
