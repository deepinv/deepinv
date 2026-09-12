from __future__ import annotations

from collections.abc import Mapping

import torch

from deepinv.physics.forward import LinearPhysics
from deepinv.transform import Transform


class TimeVaryingMotion(LinearPhysics):
    r"""Apply a deepinv transform with different parameters at each time step.

    The operator acts on 2D dynamic images of shape ``(B,C,T,H,W)`` and
    applies one transform parameter set to each ``(batch,time)`` image.

    More precisely, given a tensor :math:`x` of shape ``(B,C,T,H,W)`` and a transform :math:`T(\cdot, \theta)`
    parametrized by :math:`\theta`, this class applies

    .. math::
        y_{b, t} = T(x_{b, t}, \theta_{b, t}).

    Motion parameters may be stored at construction, changed persistently
    using ``update``, or overridden for one call to :meth:`A` or
    :meth:`A_adjoint`.

    .. note::
        This is a new functionality that is not yet supported by all transforms. Please do not hesitate to raise an
        issue in case of an unexpected behavior.

    :param Transform transform: deterministic DeepInv transform with
        ``n_trans=1`` and constant output shape.
    :param motion_params: optional mapping of parameters with leading
        dimensions ``(B,T)``.
    :param torch.device, str device: operator device.

    |sep|

    :Example:

    >>> import torch
    >>> from deepinv.physics import TimeVaryingMotion
    >>> from deepinv.transform import Shift
    >>> x = torch.zeros(1, 1, 3, 5, 6)  # (B,C,T,H,W)
    >>> x[0, 0, :, 2, 1] = 1  # Same impulse at (y,x)=(2,1) in every frame
    >>> params = {
    ...     "x_shift": torch.tensor([[0, 1, 2]]),
    ...     "y_shift": torch.tensor([[0, 0, -1]]),
    ... }
    >>> motion = TimeVaryingMotion(Shift(), motion_params=params)
    >>> motion(x)[0, 0].nonzero().tolist()  # Coordinates are (t,y,x)
    [[0, 2, 1], [1, 2, 2], [2, 1, 3]]
    """

    _motion_param_prefix = "_motion_param_"

    def __init__(
        self,
        transform: Transform,
        motion_params: Mapping[str, torch.Tensor] | None = None,
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
        self.transform = transform
        if motion_params is not None:
            self.update_parameters(motion_params=motion_params)
        self.to(device)

    @staticmethod
    def check_params(
        params: Mapping[str, torch.Tensor] | None,
        batch_size: int | None = None,
        time_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""Validate and optionally broadcast motion parameters.

        :param Mapping[str, torch.Tensor] params: parameter tensors with leading
            batch and time dimensions ``(B,T)``.
        :param int batch_size: target batch size. Must be provided together with
            ``time_size``.
        :param int time_size: target number of time steps. Must be provided
            together with ``batch_size``.
        :param torch.device, str device: optional device for returned tensors.
        :return: Validated parameters, broadcast to ``(batch_size,time_size,...)``
            when target dimensions are provided.
        """
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
            if not isinstance(value, torch.Tensor):
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
        motion_params: Mapping[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        """Update motion parameters stored as operator buffers.

        Existing motion-parameter buffers are replaced when ``motion_params``
        is provided.

        :param Mapping[str, torch.Tensor] motion_params: parameter tensors with
            leading batch and time dimensions ``(B,T)``.
        """
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
        x: torch.Tensor,
        motion_params: Mapping[str, torch.Tensor] | None,
        inverse: bool,
    ) -> torch.Tensor:
        r"""
        Applies motion parameters to ``x``.

        :param torch.Tensor x: input tensor with shape ``(B,C,T,H,W)``.
        :param Mapping[str, torch.Tensor] motion_params: parameter tensors with
        leading batch and time dimensions ``(B,T)``.
        :param bool inverse: whether to invert the transform
        :return: transformed tensor with shape ``(B,C,T,H,W)``.
        """
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
        params = self.check_params(motion_params, x.shape[0], x.shape[2], x.device)
        if not params:
            raise ValueError("TimeVaryingMotion requires non-empty motion parameters.")

        output = torch.empty_like(x)
        # note: batching over t dimension should be supported - but this requires careful check of transforms, would be worth splitting in another PR
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
        x: torch.Tensor,
        motion_params: Mapping[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Apply the time-varying transform.

        :param torch.Tensor x: dynamic image with shape ``(B,C,T,H,W)``.
        :param Mapping[str, torch.Tensor] motion_params: optional per-call
            parameter override with leading dimensions ``(B,T)``.
        :return: Transformed dynamic image with the same shape as ``x``.
        """
        return self._apply_motion(x, motion_params, inverse=False)

    def A_adjoint(
        self,
        x: torch.Tensor,
        motion_params: Mapping[str, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""Apply the adjoint (reverse) time-varying transform.

        :param torch.Tensor x: dynamic image with shape ``(B,C,T,H,W)``.
        :param Mapping[str, torch.Tensor] motion_params: optional per-call
            parameter override with leading dimensions ``(B,T)``.
        :return: Adjoint-transformed dynamic image with the same shape as ``x``.
        """
        return self._apply_motion(x, motion_params, inverse=True)
