from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def prepare_array(array, *, complex_mode: str = "magnitude") -> np.ndarray:
    """Normalise arrays for metrics and visualisation."""
    array = np.asarray(array)
    if np.iscomplexobj(array):
        if complex_mode == "real":
            array = array.real
        elif complex_mode == "imag":
            array = array.imag
        else:
            array = np.abs(array)
    return np.asarray(array, dtype=np.float32)


def compute_basic_metrics(reference, candidate) -> dict[str, float]:
    """Compute compact regression-style image metrics."""
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    diff = cand - ref
    ref_flat = ref.reshape(-1)
    cand_flat = cand.reshape(-1)
    diff_flat = diff.reshape(-1)
    ref_norm = float(np.linalg.norm(ref_flat))
    rmse = float(np.sqrt(np.mean(diff_flat**2)))
    ref_range = float(ref.max() - ref.min())
    covariance = float(
        np.mean((ref_flat - ref_flat.mean()) * (cand_flat - cand_flat.mean()))
    )
    denom = float(ref_flat.std() * cand_flat.std())
    correlation = covariance / max(denom, 1e-12)
    return {
        "mae": float(np.mean(np.abs(diff_flat))),
        "rmse": rmse,
        "nrmse_range": rmse / max(ref_range, 1e-12),
        "relative_l2": float(np.linalg.norm(diff_flat)) / max(ref_norm, 1e-12),
        "bias": float(np.mean(diff_flat)),
        "correlation": correlation,
    }


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return output_path
