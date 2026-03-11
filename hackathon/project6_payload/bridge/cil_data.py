from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from cil.optimisation.functions import Function
from cil.plugins.astra import ProjectionOperator
from cil.processors import Slicer, TransmissionAbsorptionConverter
from cil.utilities.dataexample import SYNCHROTRON_PARALLEL_BEAM_DATA

from .utils import save_central_slices


@dataclass
class CILExample2D:
    acquisition_data: object
    operator: object
    image_template: object
    data_name: str
    vertical_index: int


def build_cil_parallel_beam_example(
    *,
    vertical_index: int = 20,
    angle_step: int = 6,
    horizontal_crop: tuple[int, int, int] = (20, 140, 1),
    device: str = "cpu",
) -> CILExample2D:
    data_sync = SYNCHROTRON_PARALLEL_BEAM_DATA.get()
    scale = data_sync.get_slice(vertical=vertical_index).mean()
    data_sync = data_sync / scale
    data_sync = TransmissionAbsorptionConverter()(data_sync)
    acquisition_data = data_sync.get_slice(vertical=vertical_index)
    acquisition_data = Slicer(
        roi={
            "angle": (0, 90, angle_step),
            "horizontal": horizontal_crop,
        }
    )(acquisition_data)
    acquisition_data.reorder(order="astra")
    image_geometry = acquisition_data.geometry.get_ImageGeometry()
    operator = ProjectionOperator(image_geometry, acquisition_data.geometry, device=device)
    image_template = image_geometry.allocate(0.0)
    return CILExample2D(
        acquisition_data=acquisition_data,
        operator=operator,
        image_template=image_template,
        data_name="SYNCHROTRON_PARALLEL_BEAM_DATA",
        vertical_index=vertical_index,
    )


def _call_torch_denoiser(denoiser, x_torch: torch.Tensor, tau: float) -> torch.Tensor:
    for kwargs in ({"ths": float(tau)}, {"sigma": float(tau)}, {}):
        try:
            return denoiser(x_torch, **kwargs)
        except TypeError:
            continue
    return denoiser(x_torch)


class DeepInvDenoiserProximal(Function):
    """Wrap a torch/DeepInverse denoiser as a CIL proximal operator."""

    def __init__(self, denoiser, device: str = "cpu"):
        self.device = torch.device(device)
        self.denoiser = denoiser
        super().__init__()

    def __call__(self, x):
        return 0

    def cil_to_torch(self, x) -> torch.Tensor:
        array = np.asarray(x.as_array(), dtype=np.float32)
        tensor = torch.from_numpy(array).to(self.device)
        if tensor.ndim == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        raise ValueError(f"Unsupported CIL container shape for torch conversion: {array.shape}")

    def torch_to_cil(self, x_torch: torch.Tensor, out):
        array = x_torch.detach().cpu().numpy()
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        out.fill(np.asarray(array, dtype=np.float32))

    def proximal(self, x, tau, out=None):
        if out is None:
            out = x.geometry.allocate(None)
        with torch.no_grad():
            x_torch = self.cil_to_torch(x)
            denoised = _call_torch_denoiser(self.denoiser, x_torch, tau)
            self.torch_to_cil(denoised, out)
        return out


def save_cil_image(image, output_path: str | Path, *, title: str = "") -> Path:
    return save_central_slices(np.asarray(image.as_array()), output_path, title=title)
