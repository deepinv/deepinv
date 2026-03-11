from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.sirf_deepinv_bridge import (  # noqa: E402
    SIRFLinearOperatorCIL,
    build_pet_ray_tracing_example,
    relative_dot_error,
)


def main():
    example = build_pet_ray_tracing_example()
    operator = SIRFLinearOperatorCIL(
        example.acquisition_model,
        example.image_template,
        example.acquisition_data,
    )

    x = operator.domain_geometry().allocate("random", seed=1)
    y = operator.range_geometry().allocate("random", seed=11)
    direct = operator.direct(x)
    adjoint = operator.adjoint(y)
    error = relative_dot_error(operator, x, y)

    summary = {
        "domain_shape": tuple(int(v) for v in operator.domain_geometry().shape),
        "range_shape": tuple(int(v) for v in operator.range_geometry().shape),
        "direct_shape": tuple(int(v) for v in direct.shape),
        "adjoint_shape": tuple(int(v) for v in adjoint.shape),
        "relative_dot_error": float(error),
    }
    output_dir = Path("outputs/project_06_cil_operator")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
