from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cil.framework import ImageGeometry, VectorGeometry
from cil.optimisation.operators import LinearOperator

from .containers import sirf_to_numpy


def cil_geometry_from_shape(shape, *, dtype=np.float32):
    """Create a simple CIL geometry matching a NumPy/SIRF array shape."""
    shape = tuple(int(v) for v in shape)
    if len(shape) == 1:
        return VectorGeometry(shape[0], dtype=dtype)
    if len(shape) == 2:
        geometry = ImageGeometry(voxel_num_x=shape[1], voxel_num_y=shape[0])
    elif len(shape) == 3:
        geometry = ImageGeometry(
            voxel_num_x=shape[2],
            voxel_num_y=shape[1],
            voxel_num_z=shape[0],
        )
    elif len(shape) == 4:
        geometry = ImageGeometry(
            voxel_num_x=shape[3],
            voxel_num_y=shape[2],
            voxel_num_z=shape[1],
            channels=shape[0],
        )
    else:
        raise ValueError(f"Unsupported shape for automatic CIL geometry: {shape}")
    geometry.dtype = dtype
    return geometry


def cil_to_numpy(container) -> np.ndarray:
    return np.asarray(container.as_array())


def _expand_to_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim < len(target_shape):
        out = np.expand_dims(out, axis=0)
    if out.shape != target_shape:
        raise ValueError(f"Cannot expand shape {array.shape} to match target shape {target_shape}.")
    return out


def _squeeze_to_shape(array: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(array)
    while out.ndim > len(target_shape) and out.shape[0] == 1:
        out = out[0]
    if out.shape != target_shape:
        raise ValueError(f"Cannot squeeze shape {array.shape} to match target shape {target_shape}.")
    return out


def cil_to_sirf_like(template, container):
    template_shape = tuple(int(v) for v in sirf_to_numpy(template, copy=True).shape)
    array = np.asarray(cil_to_numpy(container), dtype=np.float32)
    array = _expand_to_shape(array, template_shape)
    out = template.clone()
    out.fill(np.ascontiguousarray(array))
    return out


def cil_from_array(array, geometry, out=None):
    if out is None:
        out = geometry.allocate(None)
    target_shape = tuple(int(v) for v in out.shape)
    array = np.asarray(array, dtype=np.float32)
    array = _squeeze_to_shape(array, target_shape)
    out.fill(array)
    return out


def sirf_to_cil(container, geometry=None, out=None):
    array = np.asarray(sirf_to_numpy(container, copy=True), dtype=np.float32)
    if geometry is None:
        geometry = cil_geometry_from_shape(array.shape, dtype=array.dtype)
    return cil_from_array(array, geometry, out=out)


@dataclass
class SIRFCILShapes:
    domain_shape: tuple[int, ...]
    range_shape: tuple[int, ...]


class SIRFLinearOperatorCIL(LinearOperator):
    """Expose a SIRF linear operator through the CIL LinearOperator API."""

    def __init__(
        self,
        operator,
        domain_template,
        range_template,
        *,
        domain_geometry=None,
        range_geometry=None,
        dtype=np.float32,
    ):
        self.operator = operator
        self.domain_template = domain_template
        self.range_template = range_template
        self.shapes = SIRFCILShapes(
            domain_shape=tuple(int(v) for v in sirf_to_numpy(domain_template, copy=True).shape),
            range_shape=tuple(int(v) for v in sirf_to_numpy(range_template, copy=True).shape),
        )
        if domain_geometry is None:
            domain_geometry = cil_geometry_from_shape(self.shapes.domain_shape, dtype=dtype)
        if range_geometry is None:
            range_geometry = cil_geometry_from_shape(self.shapes.range_shape, dtype=dtype)
        super().__init__(domain_geometry, range_geometry=range_geometry)

    def domain_from_sirf(self, container, out=None):
        return sirf_to_cil(container, self.domain_geometry(), out=out)

    def range_from_sirf(self, container, out=None):
        return sirf_to_cil(container, self.range_geometry(), out=out)

    def domain_to_sirf(self, container):
        return cil_to_sirf_like(self.domain_template, container)

    def range_to_sirf(self, container):
        return cil_to_sirf_like(self.range_template, container)

    def direct(self, x, out=None):
        sirf_x = self.domain_to_sirf(x)
        sirf_y = self.operator.forward(sirf_x)
        return self.range_from_sirf(sirf_y, out=out)

    def adjoint(self, x, out=None):
        sirf_y = self.range_to_sirf(x)
        sirf_x = self.operator.backward(sirf_y)
        return self.domain_from_sirf(sirf_x, out=out)


def relative_dot_error(operator, x, y) -> float:
    lhs = operator.direct(x).dot(y)
    rhs = x.dot(operator.adjoint(y))
    denom = max(abs(lhs), abs(rhs), 1e-12)
    return float(abs(lhs - rhs) / denom)
