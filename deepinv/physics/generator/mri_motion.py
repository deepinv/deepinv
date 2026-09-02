from __future__ import annotations

import torch

from deepinv.physics.generator.base import PhysicsGenerator


class BrownianMotionGenerator(PhysicsGenerator):
    r"""Generate bounded Brownian rigid-motion trajectories for 2D MRI.

    TODO

    :param int n_frames: number of temporal samples.
    :param float dt: time between samples in seconds.
    :param float rotation_sigma: rotational diffusion in degrees per square
        root second.
    :param float, tuple translation_sigma: translational diffusion in pixels
        per square root second, ordered as ``(x, y)``.
    :param float rotation_max: maximum absolute rotation in degrees.
    :param float, tuple translation_max: maximum absolute translation in
        pixels, ordered as ``(x, y)``.
    :param torch.Generator rng: random number generator.
    :param str, torch.device device: generation device.
    :param torch.dtype dtype: generated tensor dtype.
    """

    def __init__(
        self,
        n_frames: int,
        dt: float = 0.04,
        rotation_sigma: float = 0.3,
        translation_sigma: float | tuple[float, float] = 0.75,
        rotation_max: float = 1.0,
        translation_max: float | tuple[float, float] = 3.0,
        rng: torch.Generator = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(rng=rng, device=device, dtype=dtype)
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.n_frames = n_frames
        self.dt = dt
        self.rotation_sigma = self._pair(rotation_sigma, "rotation_sigma")[0]
        self.translation_sigma = self._pair(translation_sigma, "translation_sigma")
        self.rotation_max = self._pair(rotation_max, "rotation_max")[0]
        self.translation_max = self._pair(translation_max, "translation_max")
        if self.rotation_sigma < 0 or min(self.translation_sigma) < 0:
            raise ValueError("Motion sigmas must be nonnegative.")
        if self.rotation_max <= 0 or min(self.translation_max) <= 0:
            raise ValueError("Motion bounds must be positive.")

    @staticmethod
    def _pair(value: float | tuple[float, float], name: str) -> tuple[float, float]:
        if isinstance(value, (int, float)):
            return float(value), float(value)
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or a pair.")
        return float(value[0]), float(value[1])

    def _trajectory(
        self,
        batch_size: int,
        n_frames: int,
        sigma: float,
        bound: float,
    ) -> torch.Tensor:
        increments = (
            sigma
            * self.dt**0.5
            * torch.randn(
                batch_size,
                n_frames - 1,
                generator=self.rng,
                **self.factory_kwargs,
            )
        )
        trajectory = torch.cat(
            (
                torch.zeros(batch_size, 1, **self.factory_kwargs),
                increments,
            ),
            dim=1,
        ).cumsum(dim=1)
        trajectory = torch.remainder(trajectory + bound, 4 * bound)
        return bound - torch.abs(trajectory - 2 * bound)

    def step(
        self,
        batch_size: int = 1,
        seed: int | str = None,
        n_frames: int | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Generate a batch of motion trajectories."""
        self.rng_manual_seed(seed)
        n_frames = self.n_frames if n_frames is None else n_frames
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if n_frames < 1:
            raise ValueError("n_frames must be at least 1.")
        return {
            "theta": self._trajectory(
                batch_size, n_frames, self.rotation_sigma, self.rotation_max
            ),
            "x_shift": self._trajectory(
                batch_size,
                n_frames,
                self.translation_sigma[0],
                self.translation_max[0],
            ),
            "y_shift": self._trajectory(
                batch_size,
                n_frames,
                self.translation_sigma[1],
                self.translation_max[1],
            ),
        }
