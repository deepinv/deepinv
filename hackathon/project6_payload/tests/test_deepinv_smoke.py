from __future__ import annotations

import deepinv as dinv

from hackathon.project6_payload.bridge import (
    SIRFLinearPhysics,
    build_pet_ray_tracing_example,
    run_deep_image_prior,
)


def main():
    example = build_pet_ray_tracing_example()
    physics = SIRFLinearPhysics(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
    )
    y = physics.measurement_tensor_from_sirf(example.acquisition_data)
    tv = dinv.models.TVDenoiser(ths=0.02, n_it_max=5)
    _ = tv(physics.A_adjoint(y))
    reconstruction, _ = run_deep_image_prior(
        physics,
        y,
        iterations=1,
        learning_rate=1e-2,
        in_size=(4, 4, 4),
        channels=32,
        layers=3,
        verbose=False,
    )
    print("DIP smoke output shape:", tuple(reconstruction.shape))


if __name__ == "__main__":
    main()
