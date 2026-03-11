from __future__ import annotations

import torch

from shared.sirf_deepinv_bridge import (
    EmissionTomographyWithSIRF,
    adjointness_error,
    build_pet_ray_tracing_example,
)


def main():
    example = build_pet_ray_tracing_example()
    physics = EmissionTomographyWithSIRF(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
    )
    error = adjointness_error(physics, trials=2)
    if error > 5e-4:
        raise SystemExit(f"Adjointness error too large: {error}")
    operator_norm = physics.estimate_operator_norm(max_iter=4, tol=1e-4, seed=0)
    if operator_norm <= 0.0:
        raise SystemExit(f"Estimated operator norm must be positive, got {operator_norm}.")
    normalized = EmissionTomographyWithSIRF(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
        operator_norm=operator_norm,
    )
    x = physics.image_tensor_from_sirf(example.image_template)
    raw_forward = physics.A_raw(x)
    normalized_forward = normalized.A(x)
    scale_error = torch.linalg.vector_norm(
        (normalized_forward - raw_forward / operator_norm).reshape(-1)
    ) / max(torch.linalg.vector_norm((raw_forward / operator_norm).reshape(-1)).item(), 1e-12)
    scale_error = float(scale_error)
    if scale_error > 5e-4:
        raise SystemExit(f"Normalisation scaling error too large: {scale_error}")
    measurement = physics.measurement_tensor_from_sirf(example.acquisition_data)
    backprojection = physics.A_adjoint(measurement)
    print("Adjointness error:", error)
    print("Estimated operator norm:", operator_norm)
    print("Normalisation error:", scale_error)
    print("Backprojection shape:", tuple(backprojection.shape))


if __name__ == "__main__":
    main()
