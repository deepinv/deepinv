"""Reusable SIRF/DeepInverse bridge code for the hackathon workspace."""

from .containers import infer_tensor_shape, sirf_to_torch, tensor_to_sirf_like
from .dip import run_deep_image_prior
from .examples import build_mr_cartesian_example, build_pet_ray_tracing_example
from .metrics import compute_basic_metrics, prepare_array, save_metrics
from .physics import EmissionTomographyWithSIRF, SIRFLinearPhysics, adjointness_error
from .pnp import pnp_reconstruct
from .utils import save_central_slices

try:
    from .cil_operator import (
        SIRFLinearOperatorCIL,
        cil_geometry_from_shape,
        relative_dot_error,
        sirf_to_cil,
    )
except Exception:  # pragma: no cover - CIL is optional in the SIRF-only runtime
    SIRFLinearOperatorCIL = None
    cil_geometry_from_shape = None
    relative_dot_error = None
    sirf_to_cil = None
