# Project 6 Submission Bundle

This folder packages the current state of Project 6 in a form that is easier to explain and hand over during the hackathon.

## Project

Project 6: DeepInverse wrapper for SIRF/CIL physics operators

## Current local status

- Working PET/STIR path: yes
- Working DeepInverse bridge: yes
- DeepInverse wrapper class aligned with the brief: yes (`EmissionTomographyWithSIRF`)
- Explicit operator normalisation support: yes
- Working CPU validation path: yes
- CIL installed separately: yes
- CIL + torch + deepinv environment: yes
- Working CIL built-in data demo: yes
- Practical unified `cil + torch + deepinv + sirf` runtime: yes, via local SIRF overlay

## What is already validated

- `project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py`
- `project_06_deepinverse_wrapper_sirf_cil/quickcheck.py`
- `project_06_deepinverse_wrapper_sirf_cil/cil_data_demo.py`
- `project_06_deepinverse_wrapper_sirf_cil/cil_operator_demo.py`
- `tests/test_pet_bridge_smoke.py`
- `tests/test_deepinv_smoke.py`
- `tests/test_cil_data_smoke.py`
- `tests/test_cil_sirf_operator_smoke.py`

## Current best live message

We have a working PET/STIR bridge from SIRF to DeepInverse, validated locally with adjointness checks, operator-norm estimation, and smoke tests. The main DeepInverse-facing class is now `EmissionTomographyWithSIRF`, which matches the project brief more closely. This already supports the Project 3 and Project 4 demos. We now also have a working CIL built-in-data demo that exercises a CIL ASTRA operator plus a DeepInverse denoiser wrapper in the separate CIL environment.

We also now have a practical unified runtime where `cil`, `torch`, `deepinv`, and `sirf.STIR` import together and Project 6 quickcheck runs successfully.

## CIL alignment note

The separate CIL environment now includes:

- `cil`
- `torch`
- `deepinv`
- `astra-toolbox`
- `ccpi-regulariser`

This is enough for the import-level path of the `CIL-User-Showcase/016_cil_torch_fista_pnp` example and for built-in CIL data loading.

The main remaining boundary is that `sirf` is available there via a rebuilt local overlay rather than as a clean package installed into the conda env itself. However, Project 6 now has both a real CIL-data artifact and a practical unified runtime.

## Main code paths

- `/home/fotis/hackathon/prior_10.3.26/project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py`
- `/home/fotis/hackathon/prior_10.3.26/project_06_deepinverse_wrapper_sirf_cil/quickcheck.py`
- `/home/fotis/hackathon/prior_10.3.26/shared/sirf_deepinv_bridge/physics.py`
- `/home/fotis/hackathon/prior_10.3.26/shared/sirf_deepinv_bridge/containers.py`
- `/home/fotis/hackathon/prior_10.3.26/shared/sirf_deepinv_bridge/examples.py`

## Fast run commands

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python project_06_deepinverse_wrapper_sirf_cil/quickcheck.py
```

```bash
cd /home/fotis/hackathon/prior_10.3.26
bash /home/fotis/hackathon/live_work/use_sirf_deepinv_cpu.sh python project_06_deepinverse_wrapper_sirf_cil/wrapper_demo.py
```
