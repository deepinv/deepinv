from __future__ import annotations


def run_deep_image_prior(
    physics,
    y,
    *,
    iterations: int = 100,
    learning_rate: float = 1e-2,
    in_size=(4, 4, 4),
    channels: int = 64,
    layers: int = 5,
    verbose: bool = False,
):
    """Run DeepInverse DIP with a SIRF-backed linear physics operator."""
    import deepinv as dinv

    if len(in_size) != len(physics.image_tensor_shape) - 1:
        raise ValueError(
            f"in_size must have {len(physics.image_tensor_shape) - 1} spatial dimensions, got {in_size}."
        )

    generator = dinv.models.ConvDecoder(
        img_size=physics.image_tensor_shape,
        in_size=tuple(in_size),
        layers=layers,
        channels=channels,
    )
    reconstructor = dinv.models.DeepImagePrior(
        generator=generator,
        img_size=(channels,) + tuple(in_size),
        iterations=iterations,
        learning_rate=learning_rate,
        verbose=verbose,
    )
    reconstruction = reconstructor(y, physics=physics)
    return reconstruction.detach(), reconstructor
