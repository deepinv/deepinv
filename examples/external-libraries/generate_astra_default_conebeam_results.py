"""Generate result images for the default ASTRA cone-beam geometry.

This script requires CUDA and astra-toolbox. It creates simple 3D phantoms,
projects them with ``TomographyWithAstra(geometry_type="conebeam")`` using the
default parameters, reconstructs them with FDK, and saves summary PNG figures.

Run from the repository root:

    python examples/external-libraries/generate_astra_default_conebeam_results.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from deepinv.physics import TomographyWithAstra


def make_phantom(img_size: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, img_size[0], device=device),
        torch.linspace(-1, 1, img_size[1], device=device),
        torch.linspace(-1, 1, img_size[2], device=device),
        indexing="ij",
    )

    sphere = ((x**2 + y**2 + z**2) < 0.45**2).float()
    small_sphere = (
        ((x - 0.25) ** 2 + (y + 0.15) ** 2 + (z - 0.10) ** 2) < 0.18**2
    ).float()
    ellipsoid = (
        ((x + 0.20) / 0.20) ** 2 + ((y - 0.25) / 0.12) ** 2 + ((z + 0.05) / 0.26) ** 2
        < 1
    ).float()

    return (sphere + 0.6 * small_sphere + 0.35 * ellipsoid)[None, None]


def save_cbct_result(
    img_size: tuple[int, int, int],
    pixel_spacing: float | tuple[float, float, float],
    output_path: Path,
    angles: int = 90,
) -> None:
    device = torch.device("cuda")
    x = make_phantom(img_size, device)

    physics = TomographyWithAstra(
        img_size=img_size,
        angles=angles,
        geometry_type="conebeam",
        pixel_spacing=pixel_spacing,
        normalize=False,
        device=device,
    )

    y = physics(x)
    x_fdk = physics.A_dagger(y, fbp=True)

    slice_idx = img_size[0] // 2
    detector_row = y.shape[2] // 2

    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    axs[0].imshow(x[0, 0, slice_idx].detach().cpu(), cmap="gray")
    axs[0].set_title("Ground truth slice")
    axs[0].axis("off")

    axs[1].imshow(y[0, 0, detector_row].detach().cpu(), cmap="magma", aspect="auto")
    axs[1].set_title("Default CBCT projection")
    axs[1].axis("off")

    axs[2].imshow(x_fdk[0, 0, slice_idx].detach().cpu(), cmap="gray")
    axs[2].set_title("FDK reconstruction")
    axs[2].axis("off")

    n = img_size[-1]
    if isinstance(pixel_spacing, tuple):
        detector_spacing = (1.5 * pixel_spacing[2], 1.5 * pixel_spacing[0])
    else:
        detector_spacing = (1.5 * pixel_spacing, 1.5 * pixel_spacing)

    fig.suptitle(
        "Default conebeam setup: "
        f"img_size={img_size}, n_detector_pixels=({n}, {n}), "
        f"detector_spacing=({detector_spacing[0]:.2g}, {detector_spacing[1]:.2g}), "
        f"source_radius={4 * n}, detector_radius={n}, angular_range=(0, 360)"
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if importlib.util.find_spec("astra") is None:
        raise ModuleNotFoundError(
            "astra-toolbox is required. Install it with "
            "`conda install -c astra-toolbox -c nvidia astra-toolbox`."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable NVIDIA GPU.")

    output_dir = Path("results") / "astra_default_conebeam"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_cbct_result(
        img_size=(64, 64, 64),
        pixel_spacing=1.0,
        output_path=output_dir / "default_conebeam_isotropic.png",
    )
    save_cbct_result(
        img_size=(48, 64, 80),
        pixel_spacing=(1.0, 1.0, 1.5),
        output_path=output_dir / "default_conebeam_anisotropic.png",
    )

    print(f"Saved default CBCT result figures to {output_dir}")


if __name__ == "__main__":
    main()
