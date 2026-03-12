## Summary

This contribution adds a prototype bridge from SIRF acquisition operators to:

- a DeepInverse-compatible linear physics interface
- a CIL-compatible linear operator interface

The current local implementation focuses on the PET/STIR path and provides:

- conversion utilities between SIRF containers and torch tensors
- conversion utilities between SIRF containers and CIL data containers
- an `EmissionTomographyWithSIRF` wrapper with forward and adjoint calls
- a `SIRFLinearPhysics` compatibility alias for the same wrapper
- explicit operator norm estimation and normalisation support
- a `SIRFLinearOperatorCIL` wrapper exposing a SIRF acquisition model as a CIL `LinearOperator`
- an autograd-compatible forward path for differentiable use cases
- validation utilities including adjointness checking and operator scaling checks
- runnable examples that support the hackathon Project 3 and Project 4 demos

## Current scope

- validated locally on PET/STIR
- CPU-focused
- DeepInverse integration demonstrated
- CIL operator integration demonstrated

## Not yet included

- broad CIL backend coverage
- parallelproj backend coverage
- MR/Gadgetron support
- upstream test-suite integration

## Validation

Locally validated with:

- `project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py`
- `project_06_deepinverse_wrapper_sirf_cil/quickcheck.py`
- `project_06_deepinverse_wrapper_sirf_cil/cil_operator_demo.py`
- `tests/test_pet_bridge_smoke.py`
- `tests/test_deepinv_smoke.py`
- `tests/test_cil_sirf_operator_smoke.py`

The current bridge passes local adjointness checks, supports explicit operator normalisation, and supports a one-iteration Deep Image Prior path on top of the PET/STIR forward model.
