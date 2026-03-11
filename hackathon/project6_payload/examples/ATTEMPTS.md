# Attempts

1. Verified the official DeepInverse `LinearPhysics` constructor and method names from the upstream source/docs.
2. Verified the SIRF PET acquisition model API from local SIRF source snapshots and tests.
3. Implemented a reusable torch bridge with:
   - tensor/SIRF container conversion
   - batch handling
   - differentiable forward pass for linear operators
   - adjointness test helper
4. Scoped the current bridge to CPU data movement because project 8 is the explicit CUDA-managed-memory follow-up.

