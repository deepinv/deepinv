from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def volume_to_numpy(volume) -> np.ndarray:
    if isinstance(volume, torch.Tensor):
        array = volume.detach().cpu().numpy()
    else:
        array = np.asarray(volume)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def central_slices(volume) -> list[np.ndarray]:
    array = volume_to_numpy(volume)
    if array.ndim == 2:
        return [array]
    if array.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D data after squeezing, got shape {array.shape}."
        )
    z, y, x = array.shape
    return [array[z // 2], array[:, y // 2, :], array[:, :, x // 2]]


def save_central_slices(volume, output_path: str | Path, *, title: str = "") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    slices = central_slices(volume)
    rendered = []
    for slc in slices:
        arr = np.abs(np.asarray(slc, dtype=np.float32))
        arr -= arr.min()
        arr /= max(arr.max(), 1e-8)
        rendered.append(Image.fromarray((255.0 * arr).astype(np.uint8), mode="L"))

    max_height = max(image.height for image in rendered)
    total_width = sum(image.width for image in rendered)
    canvas = Image.new("L", (total_width, max_height), color=0)
    x_offset = 0
    for image in rendered:
        y_offset = (max_height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width
    canvas.save(output_path)
    return output_path
