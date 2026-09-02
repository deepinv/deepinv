from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from deepinv.physics.forward import LinearPhysics
from deepinv.transform import Transform


class TimeVaryingMotion(LinearPhysics):
    r"""Apply a DeepInv transform with different parameters at each time step.

    The operator acts on 2D dynamic images of shape ``(B,C,T,H,W)`` and
    applies one transform parameter set to each ``(batch,time)`` image.
    Existing transforms otherwise interpret multiple parameters as multiple
    transformations of the complete batch.

    Motion parameters may be stored at construction, changed persistently
    using :meth:`update`, or overridden for one call to :meth:`A` or
    :meth:`A_adjoint`.

    :param Transform transform: deterministic DeepInv transform with
        ``n_trans=1`` and constant output shape.
    :param motion_params: optional mapping of parameters with leading
        dimensions ``(B,T)``.
    :param str adjoint: adjoint implementation. Currently only ``"inverse"``
        is supported, so the wrapped transform's inverse must be its
        mathematical adjoint when exact adjointness is required.
    :param torch.device, str device: operator device.
    """

    _motion_param_prefix = "_motion_param_"

    def __init__(
        self,
        transform: Transform,
        motion_params: Mapping[str, Tensor] | None = None,
        adjoint: str = "inverse",
        device: torch.device | str = "cpu",
    ):
        super().__init__(device=device)
        if not isinstance(transform, Transform):
            raise TypeError(
                "transform must be an instance of deepinv.transform.Transform."
            )
        if transform.n_trans != 1:
            raise ValueError("TimeVaryingMotion requires transform.n_trans == 1.")
        if not transform.constant_shape:
            raise ValueError("TimeVaryingMotion requires a constant-shape transform.")
        if adjoint != "inverse":
            raise ValueError(
                'TimeVaryingMotion currently supports only adjoint="inverse".'
            )
        self.transform = transform
        self.adjoint_mode = adjoint
        if motion_params is not None:
            self.update_parameters(motion_params=motion_params)
        self.to(device)

    @staticmethod
    def check_params(
        params: Mapping[str, Tensor] | None,
        batch_size: int | None = None,
        time_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Validate motion parameters and optionally broadcast to ``(B,T,...)``."""
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("motion_params must be a mapping from names to tensors.")
        if (batch_size is None) != (time_size is None):
            raise ValueError("batch_size and time_size must be provided together.")

        checked = {}
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
            if value.ndim < 2:
                raise ValueError(
                    f"Motion parameter {name!r} must have leading dimensions "
                    f"(B,T), but got shape {tuple(value.shape)}."
                )
            if batch_size is not None and (
                value.shape[0] not in (1, batch_size)
                or value.shape[1] not in (1, time_size)
            ):
                raise ValueError(
                    f"Motion parameter {name!r} with shape {tuple(value.shape)} "
                    f"is not broadcast-compatible with (B,T)=({batch_size},"
                    f"{time_size})."
                )
            value = value.to(device=device) if device is not None else value
            checked[name] = (
                value.expand(batch_size, time_size, *value.shape[2:])
                if batch_size is not None
                else value
            )
        return checked

    def update_parameters(
        self,
        motion_params: Mapping[str, Tensor] | None = None,
        **kwargs,
    ) -> None:
        """Update the stored motion parameters."""
        super().update_parameters(**kwargs)
        if motion_params is not None:
            checked = self.check_params(
                motion_params, device=self._device_holder.device
            )

            for buffer_name in list(self._buffers):
                if buffer_name.startswith(self._motion_param_prefix):
                    delattr(self, buffer_name)

            for name, value in checked.items():
                self.register_buffer(
                    f"{self._motion_param_prefix}{name}",
                    value,
                )

    def _apply_motion(
        self,
        x: Tensor,
        motion_params: Mapping[str, Tensor] | None,
        inverse: bool,
    ) -> Tensor:
        if x.ndim != 5:
            raise ValueError(
                "TimeVaryingMotion currently supports 2D dynamic images with "
                f"shape (B,C,T,H,W), but got {tuple(x.shape)}."
            )
        if motion_params is None:
            motion_params = {
                name.removeprefix(self._motion_param_prefix): value
                for name, value in self.named_buffers(recurse=False)
                if name.startswith(self._motion_param_prefix)
            }
        params = self.check_params(
            motion_params, x.shape[0], x.shape[2], x.device
        )
        if not params:
            raise ValueError("TimeVaryingMotion requires non-empty motion parameters.")

        output = torch.empty_like(x)
        for t in range(x.shape[2]):
            frame_params = {name: value[:, t] for name, value in params.items()}
            if inverse:
                frame_params = self.transform.invert_params(frame_params)
            transformed = self.transform.transform(
                x[:, :, t], batchwise=False, **frame_params
            )
            output[:, :, t] = transformed
        return output

    def A(
        self,
        x: Tensor,
        motion_params: Mapping[str, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        """Apply the time-varying transform."""
        return self._apply_motion(x, motion_params, inverse=False)

    def A_adjoint(
        self,
        x: Tensor,
        motion_params: Mapping[str, Tensor] | None = None,
        **kwargs,
    ) -> Tensor:
        """Apply the adjoint of the time-varying transform."""
        return self._apply_motion(x, motion_params, inverse=True)
