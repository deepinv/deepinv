# Project 6: DeepInverse Wrapper for SIRF/CIL Physics Operators

## Goal

Create a DeepInverse `LinearPhysics` bridge so SIRF emission-tomography operators can be used natively by DeepInverse code.

## What is here

- [wrapper_demo.py](/home/fotis/hackathon/project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py)
- [quickcheck.py](/home/fotis/hackathon/project_06_deepinverse_wrapper_sirf_cil/quickcheck.py)
- [phantom_compare.py](/home/fotis/hackathon/prior_10.3.26/project_06_deepinverse_wrapper_sirf_cil/phantom_compare.py)
- [cil_data_demo.py](/home/fotis/hackathon/project_06_deepinverse_wrapper_sirf_cil/cil_data_demo.py)
- [cil_operator_demo.py](/home/fotis/hackathon/prior_10.3.26/project_06_deepinverse_wrapper_sirf_cil/cil_operator_demo.py)
- Shared bridge code:
  - [containers.py](/home/fotis/hackathon/shared/sirf_deepinv_bridge/containers.py)
  - [cil_operator.py](/home/fotis/hackathon/prior_10.3.26/shared/sirf_deepinv_bridge/cil_operator.py)
  - [physics.py](/home/fotis/hackathon/shared/sirf_deepinv_bridge/physics.py)
  - [examples.py](/home/fotis/hackathon/shared/sirf_deepinv_bridge/examples.py)
  - [cil_data.py](/home/fotis/hackathon/prior_10.3.26/shared/sirf_deepinv_bridge/cil_data.py)

## Status

This is the core deliverable in the workspace. It is the most reusable part and directly supports Projects 3 and 4.

Current validated scope:

- PET/STIR path
- CPU execution
- real SIRF-backed forward/adjoint operator
- `EmissionTomographyWithSIRF`, with `SIRFLinearPhysics` kept as a compatibility alias
- explicit operator normalisation support for DeepInverse-facing use
- real CIL `LinearOperator` wrapper around a SIRF acquisition model
- DeepInverse model compatibility
- CIL built-in data demo path

Not yet hardened as a core-SIRF PR:

- no upstream test integration yet
- no MR/Gadgetron path in the current environment
- no CIL/parallelproj backend coverage yet

## Fast commands

Run the main demo:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py
```

Run the quick validation check:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python project_06_deepinverse_wrapper_sirf_cil/quickcheck.py
```

Run a synthetic phantom-vs-reconstruction comparison with metrics:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python project_06_deepinverse_wrapper_sirf_cil/phantom_compare.py
```

This comparison now defaults to a stronger setting:

- DeepInverse-native `A_dagger` with `CG`
- 60 solver iterations by default
- positivity clipping after the solve
- operator normalisation built into the wrapper
- additional axial-only comparison strip for easier visual inspection

You can still force the older gradient path with:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python \
  project_06_deepinverse_wrapper_sirf_cil/phantom_compare.py \
  --method gradient --iterations 40
```

Run the CIL-data demo:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_cil_env.sh project_06_deepinverse_wrapper_sirf_cil/cil_data_demo.py
```

Run the CIL-operator demo:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_cil_sirf_deepinv_env.sh project_06_deepinverse_wrapper_sirf_cil/cil_operator_demo.py
```

Run Project 6 in the unified `cil + sirf + deepinv` runtime:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_cil_sirf_deepinv_env.sh project_06_deepinverse_wrapper_sirf_cil/quickcheck.py --skip-save
```

Run the existing smoke scripts directly:

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python tests/test_pet_bridge_smoke.py
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python tests/test_deepinv_smoke.py
bash /home/fotis/hackathon/live_work/use_cil_env.sh tests/test_cil_data_smoke.py
bash /home/fotis/hackathon/live_work/use_cil_sirf_deepinv_env.sh tests/test_cil_sirf_operator_smoke.py
```

## What the quickcheck verifies

- SIRF PET example data loads
- the bridge constructs an `EmissionTomographyWithSIRF` operator
- forward/adjoint tensor flow works
- adjointness error is below a fixed threshold
- operator norm estimation succeeds
- explicit operator normalisation matches the raw operator scaling
- a DeepInverse denoiser runs on the reconstructed tensor
- a one-iteration DIP path runs successfully
- a synthetic PET phantom can be forward-projected, reconstructed, and compared against a known reference with saved metrics
- built-in CIL example data loads
- a CIL ASTRA projection operator runs direct and adjoint on CPU
- a DeepInverse TV denoiser runs through a custom CIL proximal wrapper
- `cil`, `torch`, `deepinv`, and `sirf.STIR` now also run together in one local runtime via the CIL env plus a rebuilt local SIRF overlay
- a SIRF PET acquisition model is now exposed as a CIL `LinearOperator`

## Why this matters

Project 6 is the easiest live project because it gives a reusable base. Once this path is stable, Projects 3 and 4 become application demos instead of separate infrastructure work.
