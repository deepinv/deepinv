"""
2D and 3D tomography with ASTRA
===============================

This example simulates parallel-beam, fan-beam and cone-beam acquisitions of
a reconstructed X-ray CT volume and compares SIRT, SPS and OSTR. We also
show filtered backprojection (FBP) for parallel-beam and Feldkamp-Davis-Kress
(FDK) for cone-beam geometry. The geometries are illustrative; they are not
the geometry of the original scan.

It requires ``astra-toolbox``, ``nibabel``, and a CUDA-capable GPU.
"""

# %%
# sphinx_gallery_tags = ["Tomography"]

import torch
import matplotlib.pyplot as plt
import deepinv as dinv
from deepinv.datasets.utils import download_archive

device = dinv.utils.get_device()
if device.type != "cuda":
    raise RuntimeError("TomographyWithAstra requires a CUDA-capable GPU.")
torch.manual_seed(0)

# %%
# The reference is a 256³ reconstruction of Walnut 1 from the
# `Walnut CBCT dataset <https://zenodo.org/records/2686726>`_, published in a
# `preprocessed collection <https://huggingface.co/datasets/Zhentao-Liu/TMI2024_SVCT_dataset>`_.
# The volume download is about 35 MB. We normalize it to [0, 1] for this
# simulation; the resulting values are not calibrated attenuation coefficients.
# NIfTI axes are converted to the ``(depth, height, width)`` convention used
# by DeepInv.

path = dinv.utils.get_cache_home() / "astra_walnut.nii.gz"
if not path.exists():
    download_archive(
        "https://huggingface.co/datasets/Zhentao-Liu/TMI2024_SVCT_dataset/resolve/main/walnut/Walnut1/gt_volume.nii.gz?download=true",
        path,
    )
volume = dinv.utils.io.load_nifti(path).permute(2, 1, 0)[None, None].to(device)
volume = volume / volume.max()

# %%
# Reconstruction methods
# ----------------------
# We use 180 views and a detector with 384 cells per dimension. This 256³
# setup is intended to fit on an 8 GB GPU; the number of views, detector cells,
# and volume size can be reduced if necessary. We use the same noise model and
# reconstruction settings for each acquisition.
# FBP and FDK are direct methods, SIRT is an iterative weighted least-squares
# method, and SPS minimizes the log-Poisson likelihood. OSTR applies the SPS
# update to subsets of projections. These unregularized iterative methods can
# eventually fit noise; iteration counts are illustrative. The ASTRA forward
# and backprojectors are approximately, rather than exactly, matched.
#
# The log-Poisson model returns line integrals, not raw photon counts. Zero
# photon counts are clipped to one before taking their logarithm. We omit the
# direct reconstruction for fan-beam geometry because the current DeepInv FBP
# implementation does not apply fan-beam weighting.
# For SIRT, measurements along rays with negligible intersection with the image
# grid are set to zero. Their true line integrals are nearly zero, and dividing
# their noise by a near-zero ray length can otherwise dominate the update.
# PSNR and NRMSE are computed over the full image or volume; SSIM is computed
# on the displayed central axial slice in 3D. The convergence plots show PSNR,
# log-Poisson negative log-likelihood (NLL), and relative iterate change. The
# NLL omits terms independent of the reconstruction, and one OSTR epoch uses
# every subset once.

mu, N0 = 0.02, 4096.0
likelihood = dinv.optim.LogPoissonLikelihood(mu=mu, N0=N0)


def reconstruct(reference, geometry_type):
    is_3d = reference.ndim == 5
    physics = dinv.physics.TomographyWithAstra(
        img_size=reference.shape[2:],
        angles=180,
        angular_range=(0, 180) if geometry_type == "parallel" else (0, 360),
        n_detector_pixels=(384, 384) if is_3d else 384,
        geometry_type=geometry_type,
        geometry_parameters={"source_radius": 800.0, "detector_radius": 400.0},
        normalize=False,
        device=device,
    )
    with torch.no_grad():
        counts = torch.poisson(N0 * torch.exp(-mu * physics.A(reference)))
        y = -torch.log(counts.clamp_min(1) / N0) / mu
        images = [reference]
        titles = ["Reference"]
        if geometry_type != "fanbeam":
            images.append(physics.A_dagger(y, fbp=True))
            titles.append("FDK" if geometry_type == "conebeam" else "FBP")
        ray_lengths = physics.A(torch.ones_like(reference))
        y_sirt = torch.where(ray_lengths > 1e-3, y, torch.zeros_like(y))

        # Track the same log-Poisson likelihood for SIRT as for SPS and OSTR,
        # even though SIRT itself optimizes a weighted least-squares objective.
        def log_poisson_nll(history, x_prev, x):
            return (likelihood(x.unsqueeze(0), y, physics) / (N0 * y.numel())).item()

        models = {
            "SIRT": (
                dinv.optim.SIRT(max_iter=30, custom_metrics={"nll": log_poisson_nll}),
                y_sirt,
            ),
            "SPS": (dinv.optim.SPS(data_fidelity=likelihood, max_iter=30), y),
            "OSTR": (
                dinv.optim.SPS(data_fidelity=likelihood, num_subsets=4, max_iter=10),
                y,
            ),
        }
        histories = {}
        for name, (model, measurement) in models.items():
            image, history = model(
                measurement,
                physics,
                init=torch.zeros_like(reference),
                x_gt=reference,
                compute_metrics=True,
            )
            images.append(image)
            titles.append(name)
            histories[name] = history

        psnr = dinv.metric.PSNR(max_pixel=1.0)
        nrmse = dinv.metric.NRMSE()
        ssim = dinv.metric.SSIM(max_pixel=1.0)
        reference_slice = (
            reference[:, :, reference.shape[2] // 2] if is_3d else reference
        )
        subtitles = ["Reference"] + [
            f"PSNR: {psnr(image, reference).item():.2f} dB\n"
            f"NRMSE: {nrmse(image, reference).item():.3f}\n"
            f"SSIM: {ssim(image[:, :, image.shape[2] // 2] if is_3d else image, reference_slice).item():.3f}"
            for image in images[1:]
        ]
    print(
        f"{geometry_type} {reference.ndim - 2}D: "
        f"image {tuple(reference.shape)}, projections {tuple(y.shape)}"
    )
    for title, subtitle in zip(titles[1:], subtitles[1:], strict=True):
        summary = subtitle.replace("\n", ", ")
        print(f"  {title}: {summary}")
    return images, titles, subtitles, histories, N0 * y.numel()


def plot_histories(histories, normalization, label):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, history in histories.items():
        axes[0].plot(history["psnr"][0], label=name)
        nll = (
            history["nll"][0]
            if name == "SIRT"
            else [v / normalization for v in history["cost"][0]]
        )
        axes[1].plot(range(1, len(nll) + 1), nll, label=name)
        residual = history["residual"][0]
        axes[2].semilogy(range(1, len(residual) + 1), residual, label=name)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("Log-Poisson NLL / (N0 × measurements)")
    axes[2].set_ylabel("Relative iterate change")
    for axis in axes:
        axis.set_xlabel("Iteration / epoch")
        axis.legend()
        axis.grid(True, alpha=0.3)
    fig.suptitle(label)
    fig.tight_layout()
    plt.show()


def plot_volumes(images, titles, subtitles, label):
    fig, axes = plt.subplots(
        3, len(images), figsize=(3 * len(images), 8), squeeze=False
    )
    for column, image in enumerate(images):
        d, h, w = (size // 2 for size in image.shape[-3:])
        slices = (image[0, 0, d], image[0, 0, :, h], image[0, 0, :, :, w])
        for row, section in enumerate(slices):
            axes[row, column].imshow(section.cpu(), cmap="gray", vmin=0, vmax=1)
            axes[row, column].axis("off")
        axes[0, column].set_title(titles[column])
        axes[-1, column].text(
            0.5,
            -0.04,
            subtitles[column],
            transform=axes[-1, column].transAxes,
            ha="center",
            va="top",
        )
    fig.suptitle(label)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    plt.show()


# %%
# 2D parallel-beam and fan-beam
# -----------------------------
# We reconstruct the central axial slice. A 2D image has shape ``(B, C, H, W)``
# and its sinogram has shape ``(B, C, angles, detector columns)``.

slice_2d = volume[:, :, volume.shape[2] // 2]
for geometry, label in [("parallel", "Parallel-beam"), ("fanbeam", "Fan-beam")]:
    images, titles, subtitles, histories, normalization = reconstruct(
        slice_2d, geometry
    )
    dinv.utils.plot(
        images,
        titles=titles,
        subtitles=subtitles,
        suptitle=f"2D {label} reconstruction",
        rescale_mode=None,
        figsize=(16, 4),
        fontsize=9,
    )
    plot_histories(histories, normalization, f"2D {label}")
    del images

# %%
# 3D parallel-beam and cone-beam
# -----------------------------
# A 3D image has shape ``(B, C, D, H, W)`` and its projections have shape
# ``(B, C, detector rows, angles, detector columns)``. The plots show three
# orthogonal central slices through each reconstructed volume.

for geometry, label in [("parallel", "Parallel-beam"), ("conebeam", "Cone-beam")]:
    images, titles, subtitles, histories, normalization = reconstruct(volume, geometry)
    plot_volumes(images, titles, subtitles, f"3D {label} reconstruction")
    plot_histories(histories, normalization, f"3D {label}")
    del images
