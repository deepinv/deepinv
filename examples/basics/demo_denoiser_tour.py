r"""
A tour of DeepInv's denoisers
===================================================

This example provides a tour of the denoisers in DeepInv.
A denoiser is a model that takes in a noisy image and outputs a denoised version of it.
Basically, it solves the following problem:

.. math::

    \underset{x}{\min}\|x -  \denoiser{x + \sigma \epsilon}{\sigma}\|_2^2.

The denoisers in DeepInv comes with different flavors, depending on whether they are derived from
analytical image processing techniques or learned from data.
This example will show how to use the different denoisers in DeepInv, compare their performances,
and highlights the different tradeoffs they offer.
"""

import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

import deepinv as dinv
from deepinv.utils.plotting import plot_inset
from deepinv.utils.demo import load_url_image, get_image_url


# %%
# Load test images
# ----------------
#
# First, let's load a test image to illustrate the denoisers.

dtype = torch.float32
device = "cpu"
img_size = (173, 125)

url = get_image_url("CBSD_0010.png")
image = load_url_image(
    url, grayscale=False, device=device, dtype=dtype, img_size=img_size
)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Finally, create a noisy version of the image with a fixed noise level sigma.
sigma = 0.2
noisy_image = image + sigma * torch.randn_like(image)

# Compute the PSNR of the noisy image
psnr_noisy = dinv.metric.cal_psnr(image, noisy_image)

# %%
# We are now ready to explore the different denoisers.
#
# Classical Denoisers
# -------------------
#
# DeepInv provides a set of classical denoisers such as :class:`deepinv.models.BM3D`,
# :class:`deepinv.models.TGVDenoiser`, or :class:`deepinv.models.WaveletDenoiser`.
#
# They can be easily used simply by instanciating their corresponding class,
# and calling them with the noisy image and the noise level.
#
bm3d = dinv.models.BM3D()
tgv = dinv.models.TGVDenoiser()
wavelet = dinv.models.WaveletDictDenoiser()

to_plot = {
    "Original": image,
    "Noisy": noisy_image,
    "BM3D": bm3d(noisy_image, sigma),
    "TGV": tgv(noisy_image, sigma),
    "Wavelet": wavelet(noisy_image, sigma),
}
plot(to_plot, suptitle=rf"Noise level $\sigma={sigma:.2f}$")

# %%
# Deep Denoisers
# --------------
#
# DeepInv also provides a set of deep denoisers.
# Most of these denoisers are available with pretrained weights, so they can be used readily.
# To instantiate them, you can simply call their corresponding class with default
# parameters and ``pretrained="download"`` to load their weights.
# You can then apply them by calling the model with the noisy image and the noise level.
dncnn = dinv.models.DnCNN()
drunet = dinv.models.DRUNet()
swinir = dinv.models.SwinIR()

to_plot = {
    "Original": image,
    "Noisy": noisy_image,
    "DnCNN": dncnn(noisy_image, sigma),
    "DRUNet": drunet(noisy_image, sigma),
    "SwinIR": swinir(noisy_image, sigma),
}
plot(to_plot, suptitle=rf"Noise level $\sigma={sigma:.2f}$")

# %%
# Comparing denoisers
# -------------------
#
# As we have seen, these denoisers don't have the same training or expected behavior depending on
# the noise level. Indeed, there are three classes of denoisers:
#
# - *Fixed-noise level denoisers:* Some denoisers are trained to be able to recover
#   images from noisy input with a fixed noise levels. Typically, this is the case
#   of :class:`deepinv.models.DnCNN` or :class:`deepinv.models.SwinIR`.
# - *Adaptive-level denoisers:* These denoisers are able to adapt to the noise level
#   of a given image. Basically, these denoisers' performance vary strognly with the
#   value ``sigma`` given as an input. This is typically the case for :class:`deepinv.models.BM3D`,
#   :class:`deepinv.models.SCUNet`, or :class:`deepinv.models.DRUNet`, but also for denoisers based on regularisations
#   like :class:`deepinv.models.WaveletDictDenoiser`.
#   A typical caveat of regularisation-based denoisers is that the second parameter doesn't
#   correspond to ``sigma`` but to a threshold value, which needs to be adapted to the noise level.
# - *Blind denoisers:* These denoisers estimate the level of noise in the input image
#   to output the cleanest image possible. Example of blind denoisers are :class:`deepinv.models.SCUNet`
#   or :class:`deepinv.models.Restformer`.
#
# Let us generate a set of noisy images with varying noise levels.

noise_levels = torch.logspace(-2, 0, 9)
noise = torch.randn((len(noise_levels), *image.shape[1:]))
noisy_images = image + noise_levels[:, None, None, None] * noise

# %%
# We first record the PSNR of the noisy images.

psnr = dinv.loss.metric.PSNR()
psnr_x = psnr(noisy_images, image)
res = [
    {"sigma": sigma.item(), "denoiser": "Noisy", "psnr": v.item(), "time": 0.0}
    for sigma, v in zip(noise_levels, psnr_x)
]

# %%
# Then, we evaluate the various denoisers with our set of varying noise level.
# Note that to minimize the computation time, we evaluate the performances in
# batch, by passing all the noisy images at once to the denoiser, with varying
# noise levels for each entry in the batch.
#
# We also store the runtime of each denoiser to evaluate the tradeoff between computation
# time and performances.

denoisers = {
    "DRUNet": dinv.models.DRUNet,
    # 'SwinIR': dinv.models.SwinIR, # SwinIR is slow for this example, skipping it in the doc
    "SCUNet": dinv.models.SCUNet,
    "DnCNN": dinv.models.DnCNN,
    "BM3D": dinv.models.BM3D,
    "Wavelet": dinv.models.WaveletDictDenoiser,
}

for name, cls in denoisers.items():
    print(f"Denoiser {name}...", end="", flush=True)
    d = cls()
    t_start = time.perf_counter()
    with torch.no_grad():
        clean_images = d(noisy_images, noise_levels)
        psnr_x = psnr(clean_images, image)
    runtime = time.perf_counter() - t_start
    res.extend(
        [
            {"sigma": sigma.item(), "denoiser": name, "psnr": v.item(), "time": runtime}
            for sigma, v in zip(noise_levels, psnr_x)
        ]
    )
    print(f"done ({runtime:.2f}s)")
df = pd.DataFrame(res)

# %%
# We can now compare the performances of the different denoisers.
# We plot the PSNR of the denoised images as a function of the noise level
# for each denoiser.

styles = {
    "Noisy": dict(ls="--", color="black"),
}
groups = df.groupby("denoiser")
_, ax = plt.subplots()
for name, g in groups:
    g.plot(x="sigma", y="psnr", label=name, ax=ax, **styles.get(name, {}))
ax.set_xscale("log")
plt.legend()


# %%
# We see that overall :class:`deepinv.models.DRUNet` achieves the best performances for all
# noise levels. It also achieves a good tradeoff between computation time and performances.
#
# Tuning regularisation-based denoisers
# -------------------------------------
#
# Note that the performances of denoisers that are based on regularisation,
# like :class:`deepinv.models.WaveletDictDenoiser`, are not well adapted to the noise level.
# Indeed, the second parameter of these denoisers is ``th``, which does not directly match the
# noise level ``sigma``. We will now show how to tune the threshold to match the noise level.
#
# First we start by evaluating the performances of the wavelet denoiser for a grid of threshold
# values on the noisy images.
wavelets = dinv.models.WaveletDictDenoiser()
thresholds = torch.logspace(-3, 1, 13)

res = []
for th in thresholds:
    t_start = time.perf_counter()
    clean_images = wavelets(noisy_images, th.item())
    runtime = time.perf_counter() - t_start
    res.extend(
        {
            "psnr": psnr(clean_img[None], image).item(),
            "sigma": sig.item(),
            "th": th.item(),
            "time": runtime,
        }
        for sig, clean_img in zip(noise_levels, clean_images)
    )
df_wavelet = pd.DataFrame(res)

# %%
# We can now display how the performances vary with the value of the threshold,
# and what is the best threshold for each noise level.
groups = df_wavelet.groupby("sigma")
best_th_psnr = groups.apply(lambda g: g.loc[g["psnr"].idxmax()])

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
cmap = plt.get_cmap("viridis")
norm = plt.cm.colors.LogNorm(
    vmin=df_wavelet["sigma"].min(), vmax=df_wavelet["sigma"].max()
)
for sigma, group in groups:
    group.plot(x="th", y="psnr", ax=axes[0], color=cmap(norm(sigma)), label=None)
axes[0].set_xscale("log")
axes[0].set_ylabel("Threshold")
axes[0].set_ylabel("PSNR")
axes[0].legend([])
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    label=r"$\sigma$",
    location="top",
    ax=axes[0],
)

axes[1].set_title("Best threshold for each noise level")
axes[1].loglog(best_th_psnr["sigma"], best_th_psnr["th"], marker="o")
axes[1].set_xlabel(r"$\sigma$")
axes[1].set_ylabel(r"Best threshold")


# %%
# Finally, we can update our comparison of the different denoisers to account for the performances of
# :class:`dinv.models.WaveletDictDenoiser` once the threshold have been tuned

merge_df = best_th_psnr.reset_index(drop=True).drop(columns="th")
merge_df["denoiser"] = "Tuned Wavelet"
merge_df = pd.concat([df.query("denoiser != 'Wavelet'"), merge_df])

styles = {
    "Noisy": dict(ls="--", color="black"),
}
_, ax = plt.subplots()
for name, g in merge_df.groupby("denoiser"):
    g.plot(x="sigma", y="psnr", label=name, ax=ax, **styles.get(name, {}))
ax.set_xscale("log")
plt.legend()


# %%
# We can also now compare the tradeoff between computation time and performances of the different denoisers.
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 2, height_ratios=[0.25, 0.75])
for i, sigma in enumerate(noise_levels[[0, 4]]):
    ax = fig.add_subplot(grid[1, i])
    to_plot = merge_df.query(f"sigma == {sigma}")
    handles = []
    for name, g in to_plot.groupby("denoiser"):
        handles.append(ax.scatter(g["time"], g["psnr"], label=name))
    ax.set_title(rf"$\sigma={sigma:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PSNR")

ax_legend = fig.add_subplot(grid[0, :])
ax_legend.legend(handles=handles, ncol=3, loc="center")
ax_legend.set_axis_off()
