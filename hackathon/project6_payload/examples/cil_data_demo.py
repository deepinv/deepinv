from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import deepinv as dinv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.sirf_deepinv_bridge.cil_data import (
    DeepInvDenoiserProximal,
    build_cil_parallel_beam_example,
    save_cil_image,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical-index", type=int, default=20)
    parser.add_argument("--angle-step", type=int, default=6)
    parser.add_argument("--tv-strength", type=float, default=0.02)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/project_06_cil_data"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    example = build_cil_parallel_beam_example(
        vertical_index=args.vertical_index,
        angle_step=args.angle_step,
        device="cpu",
    )

    uniform_image = example.image_template.geometry.allocate(1.0)
    forward_projection = example.operator.direct(uniform_image)
    backprojection = example.operator.adjoint(example.acquisition_data)

    denoiser = dinv.models.TVDenoiser(ths=args.tv_strength, n_it_max=10)
    regulariser = DeepInvDenoiserProximal(denoiser, device="cpu")
    denoised = regulariser.proximal(backprojection, args.tv_strength)

    backprojection_array = np.asarray(backprojection.as_array(), dtype=np.float32)
    denoised_array = np.asarray(denoised.as_array(), dtype=np.float32)
    summary = {
        "data_name": example.data_name,
        "vertical_index": args.vertical_index,
        "angle_step": args.angle_step,
        "acquisition_shape": tuple(int(v) for v in example.acquisition_data.shape),
        "image_shape": tuple(int(v) for v in example.image_template.shape),
        "forward_projection_shape": tuple(int(v) for v in forward_projection.shape),
        "backprojection_shape": tuple(int(v) for v in backprojection.shape),
        "denoised_shape": tuple(int(v) for v in denoised.shape),
        "backprojection_norm": float(np.linalg.norm(backprojection_array.reshape(-1))),
        "denoised_norm": float(np.linalg.norm(denoised_array.reshape(-1))),
        "difference_norm": float(np.linalg.norm((denoised_array - backprojection_array).reshape(-1))),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.skip_save:
        save_cil_image(
            backprojection,
            args.output_dir / "backprojection.png",
            title="Project 6: CIL Backprojection",
        )
        save_cil_image(
            denoised,
            args.output_dir / "denoised.png",
            title="Project 6: CIL TVDenoiser",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
