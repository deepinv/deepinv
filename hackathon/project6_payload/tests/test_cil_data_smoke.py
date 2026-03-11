from __future__ import annotations

import sys
from pathlib import Path

import deepinv as dinv
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hackathon.project6_payload.bridge.cil_data import (  # noqa: E402
    DeepInvDenoiserProximal,
    build_cil_parallel_beam_example,
)


def main():
    example = build_cil_parallel_beam_example(device="cpu")
    image = example.image_template.geometry.allocate(1.0)
    forward_projection = example.operator.direct(image)
    backprojection = example.operator.adjoint(example.acquisition_data)

    denoiser = dinv.models.TVDenoiser(ths=0.02, n_it_max=5)
    regulariser = DeepInvDenoiserProximal(denoiser, device="cpu")
    denoised = regulariser.proximal(backprojection, 0.02)

    backprojection_array = np.asarray(backprojection.as_array(), dtype=np.float32)
    denoised_array = np.asarray(denoised.as_array(), dtype=np.float32)
    diff_norm = float(
        np.linalg.norm((denoised_array - backprojection_array).reshape(-1))
    )
    if diff_norm <= 0.0:
        raise SystemExit(
            "CIL denoiser proximal produced no change in the backprojection."
        )

    print("Acquisition shape:", tuple(example.acquisition_data.shape))
    print("Forward projection shape:", tuple(forward_projection.shape))
    print("Backprojection shape:", tuple(backprojection.shape))
    print("Denoised shape:", tuple(denoised.shape))
    print("Difference norm:", diff_norm)


if __name__ == "__main__":
    main()
