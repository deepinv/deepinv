from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepinv as dinv
import torch

from shared.sirf_deepinv_bridge import (
    EmissionTomographyWithSIRF,
    adjointness_error,
    build_pet_ray_tracing_example,
    run_deep_image_prior,
    save_central_slices,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adjoint-trials", type=int, default=2)
    parser.add_argument("--max-adjoint-error", type=float, default=5e-4)
    parser.add_argument("--norm-max-iter", type=int, default=6)
    parser.add_argument("--max-normalization-error", type=float, default=5e-4)
    parser.add_argument("--dip-iterations", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/project_06_quickcheck"),
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
    raw_operator_norm = physics.estimate_operator_norm(
        max_iter=args.norm_max_iter,
        tol=1e-4,
        seed=0,
    )
    normalized_physics = EmissionTomographyWithSIRF(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
        operator_norm=raw_operator_norm,
    )
    measurement = physics.measurement_tensor_from_sirf(example.acquisition_data)
    backprojection = physics.A_adjoint(measurement)
    error = adjointness_error(physics, trials=args.adjoint_trials)
    if error > args.max_adjoint_error:
        raise SystemExit(
            f"Adjointness error {error:.6e} exceeds threshold {args.max_adjoint_error:.6e}"
        )
    normalized_measurement = normalized_physics.A(physics.image_tensor_from_sirf(example.image_template))
    reference_measurement = physics.A_raw(physics.image_tensor_from_sirf(example.image_template))
    normalization_error = torch.linalg.vector_norm(
        (normalized_measurement - reference_measurement / raw_operator_norm).reshape(-1)
    ) / max(
        torch.linalg.vector_norm((reference_measurement / raw_operator_norm).reshape(-1)).item(),
        1e-12,
    )
    normalization_error = float(normalization_error)
    if normalization_error > args.max_normalization_error:
        raise SystemExit(
            "Normalisation error "
            f"{normalization_error:.6e} exceeds threshold {args.max_normalization_error:.6e}"
        )

    # Exercise a real DeepInverse model on the SIRF-backed tensor path.
    tv = dinv.models.TVDenoiser(ths=0.02, n_it_max=5)
    tv_output = tv(backprojection)

    reconstruction, _ = run_deep_image_prior(
        physics,
        measurement,
        iterations=args.dip_iterations,
        learning_rate=1e-2,
        in_size=(4, 4, 4),
        channels=32,
        layers=3,
        verbose=False,
    )

    summary = {
        "wrapper_class": "EmissionTomographyWithSIRF",
        "adjointness_error": float(error),
        "estimated_operator_norm": float(raw_operator_norm),
        "normalization_error": float(normalization_error),
        "measurement_shape": tuple(measurement.shape),
        "backprojection_shape": tuple(backprojection.shape),
        "tv_output_shape": tuple(tv_output.shape),
        "dip_output_shape": tuple(reconstruction.shape),
        "data_path": example.data_path,
        "raw_data_file": example.raw_data_file,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if not args.skip_save:
        save_central_slices(
            backprojection,
            args.output_dir / "backprojection.png",
            title="Project 6: Backprojection",
        )
        save_central_slices(
            reconstruction,
            args.output_dir / "dip_reconstruction.png",
            title="Project 6: DIP Quickcheck",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
