from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def sirf_to_numpy(container, copy: bool = True) -> np.ndarray:
    """Return a NumPy view/copy from a SIRF container."""
    if hasattr(container, "asarray"):
        try:
            array = container.asarray(copy=copy)
            return np.asarray(array)
        except TypeError:
            pass
        except Exception:
            if not copy:
                raise
    array = container.as_array()
    if copy:
        return np.array(array, copy=True)
    return np.asarray(array)


def infer_tensor_shape(template) -> tuple[int, ...]:
    """Return a channel-first tensor shape for a SIRF template."""
    return (1,) + tuple(int(v) for v in sirf_to_numpy(template, copy=True).shape)


def sirf_to_torch(
    container,
    *,
    device: str | torch.device | None = None,
    copy: bool = True,
    add_batch: bool = False,
    add_channel: bool = True,
) -> torch.Tensor:
    """Convert SIRF data into a torch tensor."""
    array = np.ascontiguousarray(sirf_to_numpy(container, copy=copy))
    tensor = torch.from_numpy(array)
    if add_channel:
        tensor = tensor.unsqueeze(0)
    if add_batch:
        tensor = tensor.unsqueeze(0)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _squeeze_singleton_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 1 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def tensor_to_sirf_like(
    template, tensor: torch.Tensor, *, squeeze_channel: bool = True
):
    """Create a SIRF container like ``template`` filled with tensor values."""
    value = tensor.detach().cpu()
    if squeeze_channel:
        value = _squeeze_singleton_channel(value)
    array = np.ascontiguousarray(value.numpy())
    out = template.clone()
    out.fill(array)
    return out


def ensure_batch_channel(
    tensor: torch.Tensor,
    raw_shape: Sequence[int],
) -> torch.Tensor:
    """Normalise to [B, C, ...] where C is usually 1."""
    expected_ndim = len(raw_shape)
    if tensor.ndim == expected_ndim:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == expected_ndim + 1:
        if tensor.shape[0] == 1:
            return tensor.unsqueeze(0)
        return tensor.unsqueeze(1)
    if tensor.ndim == expected_ndim + 2:
        return tensor
    raise ValueError(
        f"Tensor of shape {tuple(tensor.shape)} is incompatible with raw shape {tuple(raw_shape)}."
    )
