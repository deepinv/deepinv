r"""
Benchmarking pretrained denoisers
===================================================

This example provides a tour of the denoisers in DeepInverse.
A denoiser is a model that takes in a noisy image and outputs a denoised version of it.
Basically, it solves the following problem:

.. math::

    \underset{x}{\min}\|x -  \denoiser{x + \sigma \epsilon}{\sigma}\|_2^2.

The denoisers in DeepInverse comes with different flavors, depending on whether they are derived from
analytical image processing techniques or learned from data.
This example will show how to use the different denoisers in DeepInverse, compare their performances,
and highlights the different tradeoffs they offer.
"""

import time

import torch
import pandas as pd
import matplotlib.pyplot as plt

import deepinv as dinv
from deepinv.utils.plotting import plot_inset
from deepinv.utils.demo import load_example
from deepinv.utils.compat import zip_strict

# %%
# Load test images
# ----------------
#
# First, let's load a test image to illustrate the denoisers.

dtype = torch.float32
device = "cpu"
img_size = (173, 125)

image = load_example(
    "CBSD_0010.png", grayscale=False, device=device, dtype=dtype, img_size=img_size
)

# Next, set the global random seed from pytorch to ensure reproducibility of the example.
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# Finally, create a noisy version of the image with a fixed noise level sigma.
sigma = 0.2
noisy_image = image + sigma * torch.randn_like(image)

# %%
# For this tour, we define an helper function to display comparison of various
# restored images, with their PSNR values and zoom-in on a region of interest.


def show_image_comparison(images, suptitle=None, ref=None):
    """Display various images restoration with PSNR and zoom-in"""

    titles = list(images.keys())
    if "Original" in images or ref is not None:
        # If the original image is in the dict, add PSNR in the titles.
        image = images["Original"] if "Original" in images else ref
        psnr = [dinv.metric.cal_psnr(image, im).item() for im in images.values()]
        titles = [
            f"{name} \n (PSNR: {psnr:.2f})" if name != "Original" else name
            for name, psnr in zip_strict(images.keys(), psnr)
        ]
    # Plot the images with zoom-in
    fig = plot_inset(
        list(images.values()),
        titles=titles,
        extract_size=0.2,
        extract_loc=(0.5, 0.0),
        inset_size=0.5,
        return_fig=True,
        show=False,
        figsize=(len(images) * 1.5, 2.5),
    )

    # Add a suptitle if it is provided
    if suptitle:
        plt.suptitle(suptitle, size=12)
        plt.tight_layout()
        fig.subplots_adjust(top=0.85, bottom=0.02, left=0.02, right=0.95)
        plt.show()


# %%
# We are now ready to explore the different denoisers.
#
# Classical Denoisers
# -------------------
#
# DeepInverse provides a set of classical denoisers such as :class:`deepinv.models.BM3D`,
# :class:`deepinv.models.TGVDenoiser`, or :class:`deepinv.models.WaveletDictDenoiser`.
#
# They can be easily used simply by instanciating their corresponding class,
# and calling them with the noisy image and the noise level.
#
bm3d = dinv.models.BM3D()
tgv = dinv.models.TGVDenoiser()
wavelet = dinv.models.WaveletDictDenoiser()

denoiser_results = {
    "Original": image,
    "Noisy": noisy_image,
    "BM3D": bm3d(noisy_image, sigma),
    "TGV": tgv(noisy_image, sigma),
    "Wavelet": wavelet(noisy_image, sigma),
}
show_image_comparison(denoiser_results, suptitle=rf"Noise level $\sigma={sigma:.2f}$")

# %%
# Deep Denoisers
# --------------
#
# DeepInverse also provides a set of deep denoisers.
# Most of these denoisers are available with pretrained weights, so they can be used readily.
# To instantiate them, you can simply call their corresponding class with default
# parameters and ``pretrained="download"`` to load their weights.
# You can then apply them by calling the model with the noisy image and the noise level.
dncnn = dinv.models.DnCNN()
drunet = dinv.models.DRUNet()
swinir = dinv.models.SwinIR()
scunet = dinv.models.SCUNet()

denoiser_results = {
    "Original": image,
    "Noisy": noisy_image,
    "DnCNN": dncnn(noisy_image, sigma),
    "DRUNet": drunet(noisy_image, sigma),
    "SCUNet": scunet(noisy_image, sigma),
    "SwinIR": swinir(noisy_image, sigma),
}
show_image_comparison(denoiser_results, suptitle=rf"Noise level $\sigma={sigma:.2f}$")

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
#   or :class:`deepinv.models.Restormer`.
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
    {"sigma": sig.item(), "denoiser": "Noisy", "psnr": v.item(), "time": 0.0}
    for sig, v in zip_strict(noise_levels, psnr_x)
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
    "DRUNet": drunet,
    # 'SwinIR': sinwir, # SwinIR is slow for this example, skipping it in the doc
    "SCUNet": scunet,
    "DnCNN": dncnn,
    "BM3D": bm3d,
    "Wavelet": wavelet,
}

for name, d in denoisers.items():
    print(f"Denoiser {name}...", end="", flush=True)
    t_start = time.perf_counter()
    with torch.no_grad():
        clean_images = d(noisy_images, noise_levels)
        psnr_x = psnr(clean_images, image)
    runtime = time.perf_counter() - t_start
    res.extend(
        [
            {"sigma": sig.item(), "denoiser": name, "psnr": v.item(), "time": runtime}
            for sig, v in zip_strict(noise_levels, psnr_x)
        ]
    )
    print(f" done ({runtime:.2f}s)")
df = pd.DataFrame(res)

# %%
# We can now compare the performances of the different denoisers.
# We plot the PSNR of the denoised images as a function of the noise level
# for each denoiser.

styles = {
    "Noisy": dict(ls="--", color="black"),
}
groups = df.groupby("denoiser")
_, ax = plt.subplots(figsize=(6, 4))
for name, g in groups:
    g.plot(x=r"sigma", y="psnr", label=name, ax=ax, **styles.get(name, {}))
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("PSNR")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()


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
        for sig, clean_img in zip_strict(noise_levels, clean_images)
    )
df_wavelet = pd.DataFrame(res)

# %%
# We can now display how the performances vary with the value of the threshold,
# and what is the best threshold for each noise level.
# sphinx_gallery_thumbnail_number = 3
groups = df_wavelet.groupby("sigma")[["sigma", "psnr", "th"]]
best_th_psnr = groups.apply(lambda g: g.loc[g["psnr"].idxmax()])

fig, axes = plt.subplots(1, 2, figsize=(15, 4))
cmap = plt.get_cmap("viridis")
norm = plt.cm.colors.LogNorm(
    vmin=df_wavelet["sigma"].min(), vmax=df_wavelet["sigma"].max()
)
for sig, group in groups:
    group.plot(x="th", y="psnr", ax=axes[0], color=cmap(norm(sig)), label=None)
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
plt.tight_layout()

# %%
# With this tuning, we can update our comparison of the different denoisers to account for
# the performances of :class:`deepinv.models.WaveletDictDenoiser` once the threshold have been tuned

merge_df = best_th_psnr.reset_index(drop=True).drop(columns="th")
merge_df["denoiser"] = "Wavelet (tuned)"
merge_df = pd.concat([df.query("denoiser != 'Wavelet'"), merge_df])

_, ax = plt.subplots(figsize=(6, 4))
for name, g in merge_df.groupby("denoiser"):
    g.plot(x=r"sigma", y="psnr", label=name, ax=ax, **styles.get(name, {}))
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("PSNR")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()

# %%#
# Adapting fixed-noise level denoisers
# ------------------------------------
#
# For fixed-noise level denoiser, we also see poor performances, since these models were trained
# for a given noise level which does not correspond to the noise level of the input image. See
# :ref:`pretrained-weights <pretrained-weights>` for more details on the chose noise level.
# A way to improve the performance of these models is to artificially rescale the input image
# to match the training noise level.
# We can define a wrapper that automatically applies this rescaling.


class AdaptedDenoiser:
    r"""
    This function rescales the input image to match the noise level of the model,
    applies the denoiser, and then rescales the output to the original noise level.
    """

    def __init__(self, model, sigma_train):
        self.model = model
        self.sigma_train = sigma_train

    def __call__(self, image, sigma):
        if isinstance(sigma, torch.Tensor):
            # If sigma is a tensor, we assume it is one value per element in the batch
            assert len(sigma) == image.shape[0]
            sigma = sigma[:, None, None, None]

        # Rescale the output to match the original noise level
        rescaled_image = image / sigma * self.sigma_train
        with torch.no_grad():
            output = self.model(rescaled_image, self.sigma_train)
        output = output * sigma / self.sigma_train
        return output


# Apply to DnCNN and SwinIR
sigma_train_dncnn = 2.0 / 255.0
adapted_dncnn = AdaptedDenoiser(dncnn, sigma_train_dncnn)

# Apply SwinIR
# sigma_train_swinir = 15.0 / 255.0
# adapted_swinir = AdaptedDenoiser(swinir, sigma_train_swinir)

# sphinx_gallery_multi_image = "single"
denoiser_results = {
    f"Original": image,
    f"Noisy": noisy_image,
    f"DnCNN": dncnn(noisy_image, sigma),
    f"DnCNN (adapted)": adapted_dncnn(noisy_image, sigma),
}
show_image_comparison(denoiser_results, suptitle=rf"Noise level $\sigma={sigma:.2f}$")

denoiser_results = {
    # Skipping SwinIR on CI due to high memory usage
    # f"SwinIR": swinir(noisy_image, sigma),
    # f"SwinIR (adapted)": adapted_swinir(noisy_image, sigma),
    f"DRUNet": drunet(noisy_image, sigma),
    f"SCUNet": scunet(noisy_image, sigma),
}
show_image_comparison(
    denoiser_results, ref=image, suptitle=rf"Noise level $\sigma={sigma:.2f}$"
)
# %%
# We can finally update our comparison with the adapted denoisers for DnCNN and SwinIR.

adapted_denoisers = {
    # "SwinIR": adapted_swinir, # SwinIR is slow for this example, skipping it in the doc
    "DnCNN (adapted)": adapted_dncnn,
}
res = []
for name, d in adapted_denoisers.items():
    print(f"Denoiser {name}...", end="", flush=True)
    t_start = time.perf_counter()
    with torch.no_grad():
        clean_images = d(noisy_images, noise_levels)
        psnr_x = psnr(clean_images, image)
    runtime = time.perf_counter() - t_start
    res.extend(
        [
            {"sigma": sig.item(), "denoiser": name, "psnr": v.item(), "time": runtime}
            for sig, v in zip_strict(noise_levels, psnr_x)
        ]
    )
    print(f" done ({runtime:.2f}s)")
df_adapted = pd.DataFrame(res)
merge_df = pd.concat(
    [merge_df.query("~denoiser.isin(['DnCNN', 'SwinIR'])"), df_adapted]
)

_, ax = plt.subplots(figsize=(6, 4))
for name, g in merge_df.groupby("denoiser"):
    g.plot(x=r"sigma", y="psnr", label=name, ax=ax, **styles.get(name, {}))
ax.set_xscale("log")
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel("PSNR")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout()
plt.show()

# %%
# We can see that the adapted denoisers achieve better performances than the original ones,
# but they are still not as good as DRUNet which is trained for a wide range of noise levels.
#
# Finally, we can also compare the tradeoff between computation time and performances of the different denoisers.
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 2, height_ratios=[0.25, 0.75])
for i, sig in enumerate(noise_levels[[0, 4]]):
    ax = fig.add_subplot(grid[1, i])
    to_plot = merge_df.query(f"sigma == {sig}")
    handles = []
    for name, g in to_plot.groupby("denoiser"):
        handles.append(ax.scatter(g["time"], g["psnr"], label=name))
    ax.set_title(rf"$\sigma={sig:.2f}$")
    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PSNR")

ax_legend = fig.add_subplot(grid[0, :])
ax_legend.legend(handles=handles, ncol=3, loc="center")
ax_legend.set_axis_off()

# %%
# We see that depending on the noise-level, the tadeoff between computation time
# and performances changes, with the deep denoisers performing the best
