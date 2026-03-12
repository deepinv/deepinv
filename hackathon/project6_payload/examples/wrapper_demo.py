from __future__ import annotations

import argparse
import json
from pathlib import Path

from hackathon.project6_payload.bridge import (
    EmissionTomographyWithSIRF,
    adjointness_error,
    build_pet_ray_tracing_example,
    save_central_slices,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint-trials", type=int, default=3)
    parser.add_argument("--max-adjoint-error", type=float, default=5e-4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/project_06_wrapper"),
    )
    parser.add_argument("--skip-save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    example = build_pet_ray_tracing_example()
    physics = EmissionTomographyWithSIRF(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
    )
    measurement = physics.measurement_tensor_from_sirf(example.acquisition_data)
    backprojection = physics.A_adjoint(measurement)
    error = adjointness_error(physics, trials=args.adjoint_trials)
    raw_operator_norm = physics.estimate_operator_norm(max_iter=4, tol=1e-4, seed=0)
    summary = {
        "wrapper_class": "EmissionTomographyWithSIRF",
        "adjointness_error": float(error),
        "estimated_operator_norm": float(raw_operator_norm),
        "max_adjoint_error": float(args.max_adjoint_error),
        "measurement_shape": tuple(measurement.shape),
        "backprojection_shape": tuple(backprojection.shape),
        "data_path": example.data_path,
        "raw_data_file": example.raw_data_file,
    }
    print(f"Adjointness relative error: {error:.6e}")
    print("Measurement tensor shape:", tuple(measurement.shape))
    print("Backprojection tensor shape:", tuple(backprojection.shape))
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if not args.skip_save:
        save_central_slices(
            backprojection,
            args.output_dir / "backprojection.png",
            title="Project 6: Backprojection",
        )
    if error > args.max_adjoint_error:
        raise SystemExit(
            f"Adjointness error {error:.6e} exceeds threshold {args.max_adjoint_error:.6e}"
        )


if __name__ == "__main__":
    main()
