from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor

from deepinv.transform import Transform


class MotionTrajectory(torch.nn.Module):
    r"""Container for time-varying MRI motion parameters.

    TODO

    :param params: mapping from parameter names to tensors of shape ``(B,T,...)``.
    """

    _prefix = "motion_param_"

    def __init__(self, params: Mapping[str, Tensor] | None = None):
        super().__init__()
        for name, value in ({} if params is None else params).items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"Motion parameter names must be valid identifiers, got {name!r}."
                )
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Motion parameter {name!r} must be a tensor, "
                    f"got {type(value).__name__}."
                )
            self.register_buffer(f"{self._prefix}{name}", value)

    def as_dict(self) -> dict[str, Tensor]:
        """Return the registered motion parameters as a dictionary."""
        return {
            name.removeprefix(self._prefix): value
            for name, value in self.named_buffers(recurse=False)
        }

    def __len__(self) -> int:
        return len(self._buffers)


class TimeVaryingMotion(torch.nn.Module, ABC):
    r"""Base class for deterministic time-varying MRI motion.

    TODO

    Implementations operate on dynamic images of shape ``(B,C,T,...)`` and
    must provide both the forward motion :math:`G_t` and its mathematical
    adjoint :math:`G_t^*`.
    """

    @staticmethod
    def check_params(
        params: Mapping[str, Tensor] | None,
        batch_size: int,
        time_size: int,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Validate and broadcast motion parameters to ``(B,T,...)``."""
        checked = {}
        for name, value in ({} if params is None else params).items():
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Motion parameter {name!r} must be a tensor, "
                    f"got {type(value).__name__}."
                )
            if value.ndim < 2:
                raise ValueError(
                    f"Motion parameter {name!r} must have leading dimensions "
                    f"(B,T), but got shape {tuple(value.shape)}."
                )
            if value.shape[0] not in (1, batch_size) or value.shape[1] not in (
                1,
                time_size,
            ):
                raise ValueError(
                    f"Motion parameter {name!r} with shape {tuple(value.shape)} "
                    f"is not broadcast-compatible with (B,T)=({batch_size},"
                    f"{time_size})."
                )
            value = value.to(device=device) if device is not None else value
            checked[name] = value.expand(batch_size, time_size, *value.shape[2:])
        return checked

    @abstractmethod
    def forward(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        """Apply the time-varying motion to ``x``."""

    @abstractmethod
    def adjoint(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        """Apply the adjoint time-varying motion to ``x``."""


class TransformMotion(TimeVaryingMotion):
    r"""Adapt a unitary DeepInv transform to time-varying MRI motion.

    The adapter applies exactly one parameter set to every ``(batch,time)``
    image. Existing transforms otherwise interpret multiple parameters as
    multiple transformations of the whole batch.

    TODO

    :param Transform transform: deterministic DeepInv transform with
        ``n_trans=1`` and constant output shape.
    :param str adjoint: adjoint implementation. Currently only ``"inverse"``
        is supported.
    """

    def __init__(self, transform: Transform, adjoint: str = "inverse"):
        super().__init__()
        if not isinstance(transform, Transform):
            raise TypeError(
                "transform must be an instance of deepinv.transform.Transform."
            )
        if transform.n_trans != 1:
            raise ValueError("TransformMotion requires transform.n_trans == 1.")
        if not transform.constant_shape:
            raise ValueError("TransformMotion requires a constant-shape transform.")
        if adjoint != "inverse":
            raise ValueError(
                'TransformMotion currently supports only adjoint="inverse".'
            )
        self.transform = transform
        self.adjoint_mode = adjoint

    def _apply_motion(
        self,
        x: Tensor,
        params: Mapping[str, Tensor] | None,
        inverse: bool,
    ) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "TransformMotion currently supports 2D dynamic images with "
                f"shape (B,C,T,H,W), but got {tuple(x.shape)}."
            )
        params = self.check_params(params, x.shape[0], x.shape[2], x.device)
        if not params:
            raise ValueError("TransformMotion requires non-empty motion parameters.")

        output = torch.empty_like(x)
        for b in range(x.shape[0]):
            for t in range(x.shape[2]):
                frame_params = {
                    name: value[b, t].reshape(1, *value.shape[2:])
                    for name, value in params.items()
                }
                if inverse:
                    frame_params = self.transform.invert_params(frame_params)
                transformed = self.transform.transform(
                    x[b : b + 1, :, t], **frame_params
                )
                if transformed.shape != x[b : b + 1, :, t].shape:
                    raise RuntimeError(
                        "The wrapped transform changed the image shape from "
                        f"{tuple(x[b : b + 1, :, t].shape)} to "
                        f"{tuple(transformed.shape)}."
                    )
                output[b : b + 1, :, t] = transformed
        return output

    def forward(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        return self._apply_motion(x, params, inverse=False)

    def adjoint(self, x: Tensor, params: Mapping[str, Tensor] | None = None) -> Tensor:
        return self._apply_motion(x, params, inverse=True)
