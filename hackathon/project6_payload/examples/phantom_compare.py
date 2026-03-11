from __future__ import annotations

import argparse
import json
from pathlib import Path

import deepinv as dinv
import numpy as np
import sirf.STIR as pet
from PIL import Image

from shared.sirf_deepinv_bridge import (
    EmissionTomographyWithSIRF,
    compute_basic_metrics,
    pnp_reconstruct,
    prepare_array,
    save_central_slices,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("cg", "gradient"), default="cg")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--tv-strength", type=float, default=0.0)
    parser.add_argument("--tv-iters", type=int, default=10)
    parser.add_argument("--solver", choices=("CG", "lsqr"), default="CG")
    parser.add_argument("--positivity", action="store_true", default=True)
    parser.add_argument("--no-positivity", dest="positivity", action="store_false")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/project_06_phantom_compare"),
    )
    return parser.parse_args()


def _normalize(array) -> np.ndarray:
    out = np.asarray(array, dtype=np.float32)
    out = out - out.min()
    out /= max(float(out.max()), 1e-8)
    return out


def _save_comparison_strip(image_paths: list[Path], output_path: Path) -> Path:
    images = [Image.open(path).convert("L") for path in image_paths]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images)
    canvas = Image.new("L", (total_width, max_height), color=0)
    x_offset = 0
    for image in images:
        y_offset = (max_height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _save_axial_slice(array, output_path: Path) -> Path:
    volume = np.asarray(array, dtype=np.float32)
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume for axial slice export, got shape {volume.shape}."
        )
    slc = _normalize(volume[volume.shape[0] // 2])
    image = Image.fromarray((255.0 * slc).astype(np.uint8), mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _save_axial_triptych(
    reference, reconstruction, abs_error, output_path: Path
) -> Path:
    paths = [
        _save_axial_slice(reference, output_path.parent / "phantom_axial.png"),
        _save_axial_slice(
            reconstruction, output_path.parent / "reconstruction_axial.png"
        ),
        _save_axial_slice(abs_error, output_path.parent / "abs_error_axial.png"),
    ]
    return _save_comparison_strip(paths, output_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(pet.examples_data_path("PET"))
    phantom_file = data_path / "test_image_PM_QP_6.hv"
    acquisition_file = data_path / "Utahscat600k_ca_seg4.hs"

    phantom = pet.ImageData(str(phantom_file))
    acquisition_template = pet.AcquisitionData(str(acquisition_file))
    acquisition_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acquisition_model.set_up(acquisition_template, phantom)

    physics = EmissionTomographyWithSIRF(
        acquisition_model,
        phantom,
        acquisition_template,
        normalize=True,
        norm_max_iter=6,
        norm_tol=1e-4,
        norm_seed=0,
    )
    operator_norm = physics.operator_norm
    if args.step_size is None:
        step_size = 1.0 / max(operator_norm**2, 1e-12)
    else:
        step_size = args.step_size
    reference = physics.image_tensor_from_sirf(phantom)
    simulated_measurement = physics.A(reference)
    backprojection = physics.A_adjoint(simulated_measurement)

    x0 = backprojection / backprojection.amax().clamp_min(1e-6)
    history = []
    if args.method == "cg":
        reconstruction = physics.A_dagger(
            simulated_measurement,
            solver=args.solver,
            max_iter=args.iterations,
            tol=1e-6,
            verbose=False,
        )
        if args.positivity:
            reconstruction = reconstruction.clamp_min(0)
        history.append(
            {
                "iteration": args.iterations,
                "solver": args.solver,
                "estimate_norm": float(reconstruction.norm().item()),
            }
        )
    else:
        denoiser = None
        if args.tv_strength > 0.0:
            denoiser = dinv.models.TVDenoiser(
                ths=args.tv_strength, n_it_max=args.tv_iters
            )
        reconstruction, history = pnp_reconstruct(
            physics,
            simulated_measurement,
            x0=x0,
            denoiser=denoiser,
            num_iterations=args.iterations,
            step_size=step_size,
            denoiser_strength=args.tv_strength,
            data_fidelity="least_squares",
        )

    phantom_array = prepare_array(phantom.as_array())
    backprojection_array = prepare_array(
        backprojection.detach().cpu().numpy()
    ).squeeze()
    reconstruction_array = prepare_array(
        reconstruction.detach().cpu().numpy()
    ).squeeze()
    abs_error_array = np.abs(reconstruction_array - phantom_array)

    phantom_path = save_central_slices(phantom_array, args.output_dir / "phantom.png")
    backprojection_path = save_central_slices(
        backprojection_array, args.output_dir / "backprojection.png"
    )
    reconstruction_path = save_central_slices(
        reconstruction_array, args.output_dir / "reconstruction.png"
    )
    abs_error_path = save_central_slices(
        abs_error_array, args.output_dir / "abs_error.png"
    )
    strip_path = _save_comparison_strip(
        [phantom_path, reconstruction_path, abs_error_path],
        args.output_dir / "phantom_reconstruction_error_strip.png",
    )
    axial_strip_path = _save_axial_triptych(
        phantom_array,
        reconstruction_array,
        abs_error_array,
        args.output_dir / "phantom_reconstruction_error_axial_strip.png",
    )

    normalized_phantom = _normalize(phantom_array)
    metrics = {
        "backprojection": compute_basic_metrics(
            normalized_phantom,
            _normalize(backprojection_array),
        ),
        "reconstruction": compute_basic_metrics(
            normalized_phantom,
            _normalize(reconstruction_array),
        ),
        "last_iteration": history[-1],
        "estimated_operator_norm": operator_norm,
        "step_size": float(step_size),
        "method": args.method,
        "solver": args.solver if args.method == "cg" else None,
        "positivity": bool(args.positivity),
        "phantom_file": str(phantom_file),
        "acquisition_file": str(acquisition_file),
        "phantom_path": str(phantom_path),
        "backprojection_path": str(backprojection_path),
        "reconstruction_path": str(reconstruction_path),
        "abs_error_path": str(abs_error_path),
        "comparison_strip_path": str(strip_path),
        "axial_strip_path": str(axial_strip_path),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
