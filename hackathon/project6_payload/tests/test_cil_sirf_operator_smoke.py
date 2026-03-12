from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hackathon.project6_payload.bridge import (  # noqa: E402
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
    if error > 5e-4:
        raise SystemExit(f"Relative CIL/SIRF dot error too large: {error}")

    print("Domain shape:", tuple(operator.domain_geometry().shape))
    print("Range shape:", tuple(operator.range_geometry().shape))
    print("Direct shape:", tuple(direct.shape))
    print("Adjoint shape:", tuple(adjoint.shape))
    print("Relative dot error:", error)


if __name__ == "__main__":
    main()
