# Review: LSMR solver + `optim/linear` refactor

**Scope reviewed:** the three "Add files via upload" commits (`2c859372`, `79d24e3f`, `7adefa6f`) on top of base `87a738f4`.
**Files touched:** `deepinv/optim/linear/{__init__,lsmr,lsqr,minres,bicgstab,conjugate_gradient,least_squares,utils}.py`, `deepinv/tests/test_optim.py`, `docs/source/changelog.rst`.
**Environment:** verified on an RTX 4070 Ti SUPER, torch 2.8.0+cu126, double precision.

**Overall:** The core LSMR algorithm is numerically correct (matches SciPy/`lstsq`/LSQR to machine precision on plain tensors), and the direction of the refactor is good — a unified residual-based convergence criterion, the batched `_sym_ortho` in `utils.py`, and adding a stagnation fallback are all sensible. But the PR as it stands has **several blocking correctness bugs/regressions** (including the new stagnation criterion, which needs a precision-dependent threshold), an accidental **test-file downgrade**, and documentation gaps. Details below, ordered by severity.

Every finding below was reproduced empirically on the GPU (double precision unless noted); each lists the concrete symptom, the reason, and a fix. "Regression" means the behavior worked on the base commit `87a738f4` and this PR breaks it.

---

## Re-audit status (after the author's fix commits)
Re-verified each finding against the updated code (HEAD `7c3c3298`); targeted tests pass (`44 + 64` linear/solver tests, incl. the restored `nonleaf_buffer_grad`).

**Resolved & verified fixed (16):**
- **B1** `g_det` restored — backward runs, grad is a tensor; the restored test passes.
- **B2** all 4 deleted tests restored; `physics.functional.gaussian_blur` API back; `lsmr` in `solvers`.
- **B3** `least_squares.py:164` now gated on `not gamma_provided` — all solvers match the regularized closed form (~5e-9) on a square op.
- **B4** `.real` restored **and** stagnation vector fixed to `alpha*y + omega*z` — complex bicgstab works (~4e-10).
- **B5** `stagtol` default is now `8·finfo(dtype).eps` — LSMR reaches 9.4e-13 at `tol=1e-12` (was 3.8e-4).
- **C6** `normf` is now the true L2 norm — multi-block `TensorList` gives ~2.5e-16 (was ~4e-3).
- **DOC 1/2/3/5/6/7** bib entry, API rst, user guide, backtick, minres param, lsmr typos — all fixed.
- **EFF-1/2/3/4** `Aty` reused and moved out of the lsqr/lsmr path; `lsmr` now skips `A(x)`/reuses `xt` at init when `x0` is zero.

**Not resolved (3):**
- **C7 — still broken.** The early return was changed `torch.any`→`torch.all`, but the upstream `torch.all(s.beta>0)` guard in `_reset_state` still sets `alpha=0` for the whole batch when one element has `b=0`, which then makes `arnorm` all-zero → early return. **Re-reproduced:** heterogeneous batch → `lsqr`/`lsmr` still return rel err = 1.0 for the solvable elements. Fix must make the `beta>0`/`alpha>0` guards **per-element**, not just the early return.
- **DOC 4 — changelog placeholders remain** (8 × `:gh:`…`` / `by `…`_`).
- **EFF-6** (minres `.item()` ×2) and the other per-iteration EFF items (5/7/8) — untouched (non-blocking).

**⚠ New regression introduced by the B5 fix (BLOCKER):**
- **`stagtol = 8.0 * torch.finfo(b.dtype).eps` crashes on `TensorList` inputs** — `TensorList` has no `.dtype`. **Re-reproduced:** `lsqr`, `lsmr`, `CG`, `minres` all raise `AttributeError: 'TensorList' object has no attribute 'dtype'` on a `TensorList` `b` (these previously ran). This also makes the C6 fix unreachable by default (the crash precedes `normf`). **Fix:** derive the dtype via `b[0].dtype` for `TensorList` (a small helper), as the solvers already do for `device`. Not covered by any test (no solver test uses `TensorList`).

---

## What was verified to work
- LSMR vs `torch.linalg.lstsq`/LSQR: ~1e-13 (over/under/square, plain tensors); damped vs closed form ~1e-9; batched `eta` ~1e-12; complex ~1e-11; `x0` warm-start consistent ~1e-12.
- **`_sym_ortho` (utils.py) is correct** — the batched rewrite matches a scalar reference exactly across all edge cases (zeros, sign combinations, tiny/huge magnitudes, `a=b=0`); `c²+s²=1` to machine precision.
- **LSMR `restart` is correct** — restart cycles of 5/10/20 match the no-restart solution.
- **`minres` handles symmetric indefinite systems** (its purpose) to ~1e-15 where `CG` diverges (`1e13`) — the solver zoo is justified.
- **`conlim` works** — `lsqr`/`lsmr` stop with the condition-number message on an ill-conditioned system.
- **`parallel_dim` with multiple batch dims** (e.g. `[0,1]`) solves correctly for `CG` and `lsmr`.
- `pytest -k "linear_system or condition_number"` → 42 passed; `test_least_square_solvers -k "CG and inpainting"` → 16 passed; `test_least_squares_implicit_backward` → passed. (Note: these pass despite the bugs above — see BLOCKER 3 / OBS 5 for why the suite misses them.)

## Scope note
The PR is intended as a cleanup of the **whole** `optim/linear` module, so pre-existing issues surfaced below (the `normf` L2-norm bug, the `minres`/`BiCGStab` `gamma` gap) are treated as in-scope fixes, not out-of-scope observations.

---

## BLOCKER 1 — Typo `g.det` crashes the implicit backward pass
**File:** `least_squares.py`, lines **367** & **369** (`LeastSquaresSolver.backward`).
```python
g_det = g.detach()          # line 362  (defined, correct, but never used)
...
if p.grad is None:
    p.grad = g.det          # line 367  <-- bound method Tensor.det, should be g_det
else:
    p.grad = p.grad + g.det # line 369  <-- should be g_det
```
**Why this is wrong:** `g_det` is a tensor (the detached gradient of a physics parameter). `g.det` is *attribute access* on that tensor, which returns the bound **method** `torch.Tensor.det` (the determinant function) — not a tensor. Assigning a method object to `p.grad` is rejected by PyTorch. In Python, `x.foo` and the local variable `x_foo` are completely unrelated; the `_` was silently turned into a `.`.

**When this code runs:** `LeastSquaresSolver.backward` optionally accumulates gradients into the `physics`'s trainable buffers (the `if params:` block). It is reached only when the forward physics has a buffer with `requires_grad=True` — e.g. learning the blur kernel / operator itself. Standard reconstruction (grad w.r.t. `y`, `z`, `gamma` only) never hits it, which is why most tests stay green.

**Reproduced:** any `least_squares_implicit_backward` whose `physics` has a trainable (`requires_grad`) buffer raises
`TypeError: assigned grad expected to be a Tensor or None but got grad of type builtin_function_or_method`.
**Fix:** `g.det` → `g_det` on both lines.
*Currently invisible to CI because the test that hit this path was deleted in this same PR (Blocker 2).*

---

## BLOCKER 2 — `test_optim.py` was overwritten with a stale copy
The uploaded `test_optim.py` is an **older revision** (net −198 lines) that silently reverts unrelated upstream work instead of only adding the LSMR test. It:
- **Deletes tests that exist on `main`:** `test_MLEM`, `test_sirt` (#985), `test_correct_global_phase` (#1074), and `test_least_squares_implicit_backward_nonleaf_buffer_grad` (#1146) — **the exact test that catches Blocker 1.**
- **Reverts an API rename:** `dinv.physics.functional.gaussian_blur` → `dinv.physics.blur.gaussian_blur` in 4 tests (stale path; still resolves today so it won't fail loudly).
- Re-adds a `FrEIA` `importorskip` to `test_patch_prior`.

Looks like a GitHub web-UI upload from an out-of-date checkout. **Fix:** rebase onto current `main`; keep only the LSMR additions (`"lsmr"` in `solvers`, the `elif solver == "lsmr"` branch in `test_linear_system`); restore all deleted tests.

---

## BLOCKER 3 — Regularization (`gamma`, `z`) silently dropped for square operators with CG
**File:** `least_squares.py`, line **160**.
```python
if complete and solver in {"BiCGStab", "minres", "CG"}:   # base had only {"BiCGStab", "minres"}
    H = lambda x: A(x)
    b = y
```
**Background.** `least_squares` solves one of two problems depending on `gamma`:
- `gamma=None` → plain least squares `min_x ‖Ax − y‖²`.
- `gamma` given → *regularized* least squares `min_x ‖Ax − y‖² + (1/γ)‖x − z‖²`, whose solution is `x = (AᵀA + I/γ)⁻¹(Aᵀy + z/γ)`.

`CG`/`minres`/`BiCGStab` only solve *symmetric/square* systems `Hx = b`, so for the regularized problem the code is supposed to set `H = AᵀA + I/γ`, `b = Aᵀy + z/γ` (this `H` is symmetric positive-definite → CG is valid). "`complete`" here means `A` is already square (`Aᵀy` has the same shape as `y`).

**The bug:** the short-circuit `H = A, b = y` (solve `Ax = y` directly) is taken **whenever `A` is square, regardless of whether `gamma` was provided.** Solving `Ax = y` is the answer to the *unregularized* problem; it throws away the `(1/γ)‖x−z‖²` term and the prior point `z` entirely. So a user who asked for a regularized/prox solve on a square operator silently gets the unregularized inverse instead (and for a non-SPD square `A`, CG on `A` directly may not even converge).

- The base restricted this branch to `BiCGStab`/`minres`; **this PR adds `CG`, the default solver.**
- **Reproduced** by calling `least_squares` directly on a square operator (random `z`, `gamma`): `solver="CG"` → err vs regularized closed form **2.4e-01** (SPD op) / **6e+07** on a general square op; `minres`/`BiCGStab` similarly wrong; `lsqr`/`lsmr` correct (~1e-8).
- **Affected class of physics:** a **non-decomposable, square `LinearPhysics`** solved through `least_squares`/`prox_l2` with `CG` (the default) and a `gamma` — e.g. `deepinv.physics.Blur` with circular padding. (`DecomposablePhysics` — `Inpainting`, `BlurFFT`, `Denoising`, … — overrides `prox_l2`/`A_dagger` with a closed-form SVD and never calls `least_squares`, so it is unaffected.)
- **Why CI misses it:** the only *square* physics in `least_squares_physics` is `inpainting`, which is decomposable and bypasses `least_squares` entirely; the other three are rectangular, where `CG` correctly uses the normal equations. So no test exercises the square-`least_squares`-`CG` path. (Compounding this, `prox_l2` passes `init=z` and the test sets `z=x`, seeding the solver at the answer — see the test-coverage note below.)

**This also exposes a pre-existing latent bug:** `minres`/`BiCGStab` already dropped `gamma` for square operators (they were in the branch on `main`). Rather than making `CG` match them, the branch should honor `gamma` for all three:
```python
if complete and not gamma_provided and solver in {"BiCGStab", "minres", "CG"}:
```
That is consistent **and** correct: the efficient direct `Ax=y` path is used only in the genuinely unregularized square case; whenever `gamma` is supplied, all solvers form the SPD normal-equations system. (See design note at the end.)

---

## BLOCKER 4 — `bicgstab` crashes on complex inputs (`.real` dropped)
**File:** `bicgstab.py`, convergence block.
```python
if torch.all(dot(r, r, dim=dim) <= tol):            # base: dot(r, r, dim=dim).real < tol
...
search_update_norm = dot(search_update, search_update, dim=dim)   # no .real
xnorm = dot(x, x, dim=dim)                                         # no .real
elif torch.all(search_update_norm <= stagtol * xnorm):
```
**Why this is wrong:** `dot(a, b) = Σ conj(a)·b`. Even though `dot(r, r) = Σ|r|²` is mathematically real, when `r` is complex PyTorch keeps the result in a **complex dtype** (imaginary part `0`). Ordered comparisons (`<`, `<=`) are undefined for complex numbers, so `complex_tensor <= tol` throws. The base code wrote `.real` to drop the (zero) imaginary part first; this PR removed it. **Reproduced** on a Hermitian positive-definite complex system:
`NotImplementedError: "compare_cuda" not implemented for 'ComplexDouble'`.
CG and MINRES kept `.real` and work on the same input. The base `bicgstab` worked on complex inputs; **this is a regression.** Complex systems are common (e.g. MRI), so this matters.
**Fix:** take `.real` on `dot(r,r)`, `search_update_norm`, and `xnorm` in the comparisons (these are real-valued quantities).

**Second, separate bug in the same added block — the stagnation vector is the wrong quantity.** The stagnation test wants "how much did `x` change this iteration?", i.e. `‖Δx‖`. From the code, the new iterate is `x_new = h + omega*z = x + alpha*y + omega*z` (lines 78, 94), so the true step is `Δx = alpha*y + omega*z`. But the PR computes `search_update = alpha*v - omega*z` (line 97), where `v = A(y)` lives in the *measurement* space and the sign on `omega*z` is flipped — it is not the change in `x` at all. (LSQR/LSMR/CG each correctly use their real `x`-increment; only BiCGStab is wrong.) In the tested case this quantity happened to be ~50× larger than the true step, so it did not cause an *early* stop, but it is still the wrong metric and could mis-fire on other problems. **Fix:** `search_update = alpha*y + omega*z`.

---

## BLOCKER 5 — Stagnation stopping criterion is not precision-dependent → caps accuracy far below the requested `tol`
**Files:** all solvers (the `stagtol` default and the `‖Δx‖ ≤ stagtol·‖x‖` test are new in this PR for `CG`/`bicgstab`/`lsqr`; `lsmr` is new; `minres` reworked).

**Background.** There are two reasons to stop an iterative solver: (a) *converged* — the residual `‖Ax−y‖` is below `tol`; (b) *stagnated* — the iterate has stopped moving, so continuing is pointless. The stagnation test here is `‖Δx‖ ≤ stagtol·‖x‖` (the relative size of the last step). The intent of (b) is to catch the case where finite-precision arithmetic prevents any further progress — that floor is set by the machine epsilon of the dtype (`eps ≈ 2.2e-16` for float64, `≈ 1.2e-7` for float32).

**Why the fixed `1e-6` is wrong.** A converging Krylov method takes *progressively smaller* steps `Δx`. So `‖Δx‖/‖x‖` passes `1e-6` while the method is still happily converging — long before the residual reaches a tight `tol`. The stagnation branch then fires and silently overrides `tol`. Worse, `1e-6` is **dtype-blind**: it is ~10·eps for float32 (roughly OK), but ~4×10⁹·eps for float64, so in double precision it stops ~10 orders of magnitude too early. A stagnation floor should scale with `torch.finfo(dtype).eps`, not be a hardcoded constant.

**Reproduced** (SPD system, cond≈300, `max_iter=5000`; verbose confirms "stagnated"):

| requested `tol` | CG (default) | CG (stagtol=0) | LSMR (default) | LSMR (stagtol=0) |
|---|---|---|---|---|
| 1e-8 | 1.6e-5 | 1.9e-7 | **3.8e-4** | 6.5e-9 |
| 1e-12 | 1.6e-5 | 1.9e-7 | **3.8e-4** | 2.6e-14 |

LSMR is worst-hit: **even at the default `tol=1e-6` it returns residual 3.8e-4** here, and the value is identical across all tighter `tol` (stagnation, not `tol`, is controlling). Precision-scaling the threshold fixes it — with `stagtol≈eps` (float64), LSMR reaches **2.2e-14**:

| dtype (eps) | stagtol=1e-6 | stagtol≈eps | stagtol≈√eps |
|---|---|---|---|
| float32 (1.2e-7) | LSMR 1.3e-5 | 1.3e-5 | 2.8e-2 |
| float64 (2.2e-16) | LSMR 3.8e-4 | **2.2e-14** | 9.2e-8 |

**Fix:** make the `stagtol` default precision-dependent (e.g. a small multiple of `torch.finfo(b.dtype).eps`, so ~1e-15 for float64 and ~1e-6 for float32), and/or only trigger when the step actually stops decreasing rather than merely being small. This is a regression: base `CG`/`lsqr`/`bicgstab` had no stagnation criterion and reached the requested `tol`.

---

## CORRECTNESS 6 — `normf` sums component norms instead of the L2 norm (multi-block TensorList)
**Files:** `lsmr.py` and `lsqr.py` (identical `normf`).
```python
total = 0.0
for k in range(len(u)):
    total += torch.linalg.vector_norm(u[k], dim=dims[k])   # Σ‖u_k‖ , not sqrt(Σ‖u_k‖²)
return total
```
**Background.** A `TensorList` `u = [u₁, …, u_K]` represents one long vector formed by stacking the blocks (deepinv uses it for stacked physics / multiple measurement operators). The Euclidean norm of that stacked vector is `‖u‖ = sqrt(‖u₁‖² + … + ‖u_K‖²)`. LSQR/LSMR are built on Golub–Kahan bidiagonalization, which repeatedly does `beta = ‖u‖; u = u/beta` to build an **orthonormal** basis — this is only correct with the true Euclidean norm.

**The bug:** `normf` returns `‖u₁‖ + … + ‖u_K‖` (sum of block norms) instead of `sqrt(Σ‖u_k‖²)`. These agree only when there is a single block, so plain tensors are fine, but for ≥2 blocks the normalization is off by up to `sqrt(K)` and the basis is no longer orthonormal — the solution is wrong.
- **Reproduced** (overdetermined, unique LS solution): single-block → ~1e-16 (correct); **two-block → LSMR 4.4e-3, LSQR 2.9e-3** vs dense `lstsq` (both wrong; underdetermined min-norm case similar).
- Pre-existing in `lsqr`; **inherited by the new `lsmr`.** Single-tensor inputs are unaffected. Worth fixing in the shared helper (`sqrt` of the sum of squared block norms); at minimum flag that multi-block `TensorList` is unsupported/untested.

---

## CORRECTNESS 7 — `lsqr`/`lsmr` return all-zeros for the **whole batch** if any one element is trivial
**Files:** `lsqr.py` (line ~160) and `lsmr.py` (line ~177).
```python
arnorm = alpha * beta
if torch.any(arnorm == 0):
    return x, acond      # <-- returns x (all zeros) for the ENTIRE batch
```
**Background.** These solvers process a batch of independent systems at once. `arnorm = ‖Aᵀr₀‖` per batch element is the natural "already solved" signal — it is `0` when that element's `b` is zero, or when `b` is orthogonal to the range of `A`. The guard is meant to return early for a solved system.

**The bug:** the check is `torch.any(...)` over the batch and the return is unconditional, so if **one** element is trivial, the function returns the initial `x` (all zeros) for **every** element — silently discarding the other, perfectly solvable systems. The related in-loop guards `if torch.all(beta > 0)` / `if torch.all(alpha > 0)` have the same all-or-nothing flaw: if one element converges mid-run, the whole batch skips its update.

**Reproduced** (batch of 3, element 0 has `b=0`): `CG`/`bicgstab`/`minres` solve elements 1 and 2 correctly (rel err ≤ 2.5e-6, element 0 → 0), but **`lsqr` and `lsmr` return rel err = 1.0 for elements 1 and 2** (they were zeroed). Pre-existing in `lsqr`, inherited by `lsmr`.
**Fix:** make the early return and the normalization guards **per-element** (`torch.where` / masking), not batch-collapsing — return early only for the elements that are actually done.

---

## Efficiency (cleanup opportunities — the PR is a module cleanup, so these are in scope)

The dominant cost in every solver is the operator applications `A`/`AT` (matvecs); then per-iteration reductions/norms; then GPU→CPU synchronizations. Measured on GPU (double precision). **Context:** the per-iteration items below only matter when the operator is *cheap* (diagonal, FFT, small conv) or many small solves are run — for an expensive physics forward operator the matvecs dominate and this overhead is negligible.

**Redundant operator applications (one-time per call; verified by counting `A`/`AT` invocations):**
- **EFF-1 — `least_squares` recomputes `AT(y)`.** `Aty = AT(y)` (line 106) is computed, then line 170 does `b = AT(y) + 1/gamma*z` — a **second** `AT(y)`. Reuse: `b = Aty + 1/gamma*z`. (Overcomplete CG+gamma measured `A=4, AT=6`; one of those `AT`s is this recompute.)
- **EFF-2 — `least_squares` computes `Aty` even when the solver never uses it.** For `lsqr`/`lsmr` with scalar `gamma`, `Aty` (line 106) is dead (those branches use `y`/`AT` internally). Move the `Aty = AT(y)` computation into the `else` (CG/minres/bicgstab) branch, and derive the batched-`gamma` shape from `z`/`xt` instead. Saves one adjoint per `lsqr`/`lsmr` call.
- **EFF-3 — `lsmr` applies `A(x)` at init even when `x0 is None`.** `_reset_state` always does `b - A(x)` (line 102); with `x0=None`, `x=0`, so this is `A(0)`. `lsqr` special-cases this (`beta = bnorm`, no matvec). Measured: direct `lsmr` `A=4` vs `lsqr` `A=3` for the same 3-iteration rectangular solve. Skip the matvec when `x` is zero.
- **EFF-4 — `lsqr`/`lsmr` compute `xt = AT(b)` only for shape, then recompute `AT(b)`.** `xt = AT(b)` (line 44/46) is used only via `zeros_like(xt)`; the first Lanczos step then computes `AT(u) = AT(b)/beta` again. Reuse `xt` (scaled) as the first `v`, or take the shape from `x0`/`z`. Saves one adjoint per call.

**Per-iteration overhead (multiplies by iteration count):**
- **EFF-5 — the new stagnation check adds a full-vector norm every iteration.** `lsmr` computes `normf(x)` and `normf(search_update)`; `minres` computes `vector_norm(solution)` and `vector_norm(search_update)`; `CG`/`bicgstab` add `dot(x,x)` and `dot(search_update,·)`. Taking a full norm of the whole solution every iteration is exactly what LSMR/LSQR are designed to avoid (SciPy estimates `xnorm` with a scalar recurrence). Measured: removing `lsmr`'s per-iteration stagnation norms saved ~4% on a cheap operator. Recommendation: update these via a recurrence, compute them every *K* iterations, or make the stagnation test opt-in.
- **EFF-6 — GPU synchronization every iteration.** All stopping tests force a device→host sync each iteration: `torch.all(...)` (CG/bicgstab/lsqr/lsmr) and especially **`.max().item()` in `minres` (twice per iteration)**. These serialize the GPU and dominate when many small/cheap solves run. Consider checking convergence every *K* iterations, or keeping the flags on-device. (`minres`'s `.item()` is the worst offender — the others at least return a 0-dim tensor.)
- **EFF-7 — `bicgstab` recomputes `left_precon(s)`.** Line 84 recomputes `left_precon(s)` already produced inside `z = right_precon(left_precon(s))` (line 80). Redundant only when a non-identity `left_precon` is supplied.
- **EFF-8 — `lsqr` allocates a full-vector temp `dk` per iteration for the condition estimate.** `dk = scalar(w, 1/rho)` exists only for `ddnorm += normf(dk)**2`; since `normf(dk) = normf(w)/|rho|` this can skip the allocation, and the whole `acond` machinery is only needed for the `conlim` stop (default `1e8`, rarely hit) and the returned value — could be computed cheaply or made optional.

**Measured per-iteration wall time** (same diagonal operator, 200 iters, so this is pure per-update overhead): `CG 0.12 ms`, `minres 0.20 ms` (1.7× CG), `lsqr 0.46 ms`, `lsmr 0.97 ms` (2.1× lsqr).

**Note — is `lsmr`'s ~2× per-iteration cost over `lsqr` expected? (and why is `minres` vs `CG` a smaller gap?)** Largely **yes, and it is mostly algorithmic**: LSMR minimizes `‖Aᵀr‖` by running an *extra* layer of Givens rotations on top of the same Golub–Kahan bidiagonalization that LSQR uses, so it carries roughly twice the scalar recurrences (`rhobar/cbar/sbar`, `betadd/betad`, `thetatilde/rhotilde0/tautilde0`, plus the `maxrbar/minrbar` condition tracking). LSQR needs a single rotation and far fewer scalars. The gap is **amplified on GPU** because each of those ~30 scalar recurrences is a separate 0-dim CUDA tensor op (kernel-launch-bound), and with a cheap operator the launches, not the FLOPs, set the wall time. The stagnation norms this PR added are only ~4% of it — not the main driver. `minres` vs `CG` is a smaller ratio (1.7×) because `minres` adds only one Givens rotation and a handful of scalars over CG's very lean loop. So: the ordering is expected; the actionable part is trimming the *implementation* overhead (EFF-4…EFF-8) and, if desired, keeping the scalar recurrences as Python floats / fusing them rather than as 0-dim tensors. For a real (expensive) forward operator this overhead is washed out by the matvecs.

---

## Documentation issues
- **DOC 1 (build-breaking):** `lsmr.py` cites `:cite:t:`fong2011lsmr`` but `docs/source/refs.bib` has no `fong2011lsmr` entry (only `paige1982lsqr`). Unresolved citation → Sphinx build error/warning. **Add the BibTeX entry.**
- **DOC 2:** `docs/source/api/deepinv.optim.rst` lists `lsqr/bicgstab/minres/conjugate_gradient/least_squares` but **not `deepinv.optim.linear.lsmr`.** Add it.
- **DOC 3:** `docs/source/user_guide/reconstruction/least-squares.rst` (≈ lines 32–35) lists CG/LSQR/MINRES/BiCGStab but not LSMR. Add it.
- **DOC 4:** `changelog.rst` has literal placeholders `(:gh:`...` by `...`_)` on the LSMR / convergence / `_sym_ortho` / lsqr entries. Fill in PR number + author.
- **DOC 5:** `least_squares.py` docstring line 61 — malformed RST `` and 'lsmr'` `` (missing opening backtick on `'lsmr'`) and a stray double space in `` `'minres'`,  and ``.
- **DOC 6:** `minres.py` docstring documents `stagtol` as a second `:param float tol:` (copy-paste; should be `:param float stagtol:`).
- **DOC 7:** `lsmr.py` docstring typos: `:param torch. b:` / `:param float, torch. eta:` (truncated `torch.Tensor`), `:math:`eta`` (should be `\eta`), `:retrun:` → `:return:`.

---

## Minor / logic observations (non-blocking)
- **OBS 1 — LSMR condition-number return is degenerate on fast convergence.** `lsmr` returns `maxrbar / minrbar` (line 283). `minrbar` starts at `+inf` and is only updated for `itn > 0`, so immediate convergence returns **0.0** (verified: identity matrix → `cond = 0.0`, true value 1.0). Separately, the loop computes a proper SciPy-style estimate in local `acond` (used for the `conlim` test) but returns the cruder `maxrbar/minrbar`. Recommend returning `acond` (as `lsqr` returns `acond.sqrt()`) and handling the `minrbar == inf` case. Low impact since `least_squares` discards it (`x, _ = lsmr(...)`), but the docstring advertises it.
- **OBS 2 — `restart` option:** converges but with noticeably reduced accuracy (tol 1e-12 → ~1e-6 result) and is untested/undocumented beyond the one-line param. Worth a test or a note.
- **OBS 3 — `minres` docstring** says `tol` is "absolute" but `b` is rescaled to unit norm, so it is effectively a *relative* residual (consistent with the others). Wording only.
- **OBS 4 — early-return type inconsistency:** `lsmr`/`lsqr` return a Python `float` `acond=1.0` on the `arnorm==0` early exit but a tensor otherwise. Cosmetic.
- **OBS 5 — `test_least_square_solvers` coverage (pre-existing, inherited).** The `least_squares_physics` list and this test are unchanged from `main`; the PR only adds `"lsmr"` to `solvers`. Two long-standing weaknesses are worth fixing while here, since they are the reason Blocker 3 slips through:
  1. `inpainting` is a `DecomposablePhysics`; its `prox_l2`/`A_dagger` use a closed-form SVD and swallow `solver`/`tol`/`max_iter`, so the `solver` axis is a **no-op** for it — every solver yields the identical SVD result. Decomposable physics should not be listed as least-squares *solver* tests (the PR now runs `lsmr` against it too, equally vacuously).
  2. No **non-decomposable square** physics (e.g. `Blur` circular) is tested, and `z=x`/`init=z` seeds `prox_l2` at the exact solution. Together these mean the regularized square-operator dispatch is never actually validated. Add such a case with `z ≠ x` and a `gamma`-honoring reference.
- **OBS 6 — checked and OK (no action):** `b=0` returns exactly `0` for all five solvers; implicit-backward gradients w.r.t. `y` and `gamma` match finite differences (~1e-6). The core implicit differentiation is sound — the `g.det` bug is isolated to the physics-buffer gradient path.

---

## Checklist for the author (consolidated, in priority order)

**Correctness — must fix before merge**
- [ ] **B1** `least_squares.py:367,369` — `g.det` → `g_det` (crashes backward with trainable physics buffers).
- [ ] **B3** `least_squares.py:160` — gate the square short-circuit on `not gamma_provided` so `CG`/`minres`/`BiCGStab` honor `gamma` for square operators (fixes the CG regression *and* the pre-existing minres/BiCGStab gap).
- [ ] **B4** `bicgstab.py` — restore `.real` in the convergence/stagnation comparisons (complex inputs currently crash); fix the stagnation vector to the true step `alpha*y + omega*z` (not `alpha*v - omega*z`).
- [ ] **B5** all solvers — make the `stagtol` default precision-dependent (≈`finfo(b.dtype).eps`); the current fixed `1e-6` silently caps float64 accuracy far below the requested `tol` (LSMR/CG worst).
- [ ] **C6** `lsmr.py`/`lsqr.py` — fix `normf` to the true L2 norm `sqrt(Σ‖u_k‖²)` (currently `Σ‖u_k‖`; wrong for multi-block `TensorList`), or document multi-block `TensorList` as unsupported.
- [ ] **C7** `lsqr.py`/`lsmr.py` — make the `arnorm==0` early return and the `beta>0`/`alpha>0` guards **per-element** (not `torch.any`/`torch.all` over the whole batch), so a single trivial element doesn't zero the rest of the batch.

**Tests**
- [ ] **B2** Do **not** replace `test_optim.py` wholesale — rebase onto `main`, keep only the LSMR additions, and restore the 4 deleted tests (esp. `test_least_squares_implicit_backward_nonleaf_buffer_grad`, which catches B1).
- [ ] Strengthen `test_least_square_solvers` (OBS 5): drop decomposable physics from the *solver* list, add a non-decomposable **square** physics with `z ≠ x` and a `gamma`-honoring reference, and add a **high-accuracy float64** case (would catch B3 and B5). Add **complex** (B4) and **multi-block TensorList** (C6) cases.

**Documentation**
- [ ] Add the `fong2011lsmr` entry to `refs.bib` (missing citation → Sphinx build fails).
- [ ] Register `lsmr` in `api/deepinv.optim.rst` and the least-squares user guide; fill the `:gh:`/author placeholders in `changelog.rst`.
- [ ] Docstring fixes: `least_squares.py:61` malformed RST; `minres.py` duplicate `:param float tol:` (→ `stagtol`); `lsmr.py` truncated types / `:math:`\eta`` / `:retrun:`.

**Efficiency (cleanup opportunities; mostly non-blocking)**
- [ ] **EFF-1** `least_squares.py:170` — reuse `Aty` instead of recomputing `AT(y)`.
- [ ] **EFF-2** `least_squares.py:106` — don't compute `Aty` on the `lsqr`/`lsmr` path (move into the CG/minres/bicgstab branch).
- [ ] **EFF-3** `lsmr.py` — skip `A(x)` at init when `x0 is None` (as `lsqr` does).
- [ ] **EFF-4** `lsqr.py`/`lsmr.py` — avoid computing `xt = AT(b)` just for shape then recomputing `AT(b)`.
- [ ] **EFF-5** — update the stagnation norms via a recurrence / every *K* iterations rather than a full-vector norm each iteration (ties in with the B5 stagnation rework).
- [ ] **EFF-6** — reduce per-iteration device→host syncs (esp. `minres`'s `.max().item()` ×2).
- [ ] **EFF-7/8** — `bicgstab` reuse `left_precon(s)`; `lsqr` avoid the per-iteration `dk` allocation / make `acond` cheaper.

**Optional / polish**
- [ ] `lsmr` return the SciPy-style `acond` instead of `maxrbar/minrbar` (currently 0.0 on fast convergence); handle `minrbar == inf`.
- [ ] Add a `restart` test; unify the early-return type (`float` vs tensor `acond`).

---

## Design note (re: regularizing the square case)
The regularization is part of the *requested problem*: passing `gamma` asks for `min ‖Ax−y‖² + (1/γ)‖x−z‖²`, which is well-defined and should be honored whether or not `A` is square. The direct `Ax=y` short-circuit solves the *unregularized* problem and only makes sense when `gamma` is `None`. So the fix is **not** to make CG match the (already-wrong) minres/BiCGStab behavior, but to gate the short-circuit on `not gamma_provided`, so every solver honors `gamma` when it is given and only the truly unregularized square case takes the fast direct path. This is both internally consistent and mathematically correct, and it repairs the pre-existing minres/BiCGStab gap at the same time.

---

## Remaining aspects not yet audited
Flagged for completeness — worth a look before merge, but not yet checked:
- **API consistency across the five solvers** (a cleanup-PR concern): `init` vs `x0`; `A,b` vs `A,AT,b`; `precon` vs `left_precon`/`right_precon`; return `x` vs `(x, cond)`; `max_iter` default `1e2` (a float) vs `100`; `parallel_dim` typed `int` in `minres` but `None|int|list` elsewhere. Unifying these would be a natural part of the cleanup.
- **The in-loop `torch.all(beta>0)`/`torch.all(alpha>0)` guards** (same root cause as C7) — I confirmed the init early-return zeros the batch; the in-loop guards likely stall a heterogeneous batch too, but I demonstrated only the early-return path.
- **Differentiating *through* the plain solvers** (autograd across the CG/LSQR loop, as in unrolled networks) — only the `LeastSquaresSolver` implicit path was checked, and only its `y`/`gamma` grads (not `z`/`init`, and not the buffer path which is B1).
- **The undercomplete min-norm path** (`least_squares` `gamma=None`, `x = AT(x)` at line 225 + the new warning) — the warning fires, but I did not numerically verify the returned min-norm solution.
- **Preconditioners** in `minres`/`bicgstab` (docstrings say "not tested") and **breakdown recovery** (do `bicgstab`'s `torch.where` safeguards actually recover, or just avoid NaN?).
- **Test-suite coverage in general** — there are currently no tests exercising complex inputs per solver, `TensorList` inputs, batched `eta`/`gamma`, `conlim`, `restart`, preconditioners, heterogeneous batches, or high-accuracy float64. Notably, **every bug found in this review lives in one of those untested areas** — so the green suite is giving false confidence.

## Overall assessment
**State: promising core, not yet mergeable — needs one more iteration.**

- **The LSMR contribution itself is sound.** The algorithm is a faithful, correct port (machine-precision agreement with SciPy/`lstsq` on plain tensors across over/under/square, damped, complex, batched `eta`, `x0`, `restart`), and `_sym_ortho`, `minres`-on-indefinite, `conlim`, `parallel_dim`, `b=0`, and the implicit `y`/`gamma` gradients all check out. The refactor's direction — a shared `utils`, a unified residual criterion, a stagnation fallback — is the right idea.
- **But the PR is blocked by a cluster of correctness bugs, several of them regressions:** a crash typo (B1), regularization silently dropped for square `CG` (B3), a `bicgstab` complex-input crash + wrong stagnation vector (B4), a precision-blind stagnation threshold that caps float64 accuracy (B5), the multi-block-`TensorList` norm bug (C6), and the batch-collapsing early return (C7). Individually the fixes are small and well-understood; collectively they mean the module returns silently-wrong or lower-accuracy answers in several realistic settings (complex/MRI, stacked physics, tight tolerances, batched-with-a-trivial-element, regularized square operators).
- **The biggest process concern is the test file (B2):** it was overwritten with a stale copy that *deletes* unrelated tests — including the one that would have caught B1 — and the existing suite doesn't cover the areas where the bugs live. So the tests must be restored *and* extended; that is the largest remaining task, larger than the code fixes.
- **Docs and efficiency** are secondary: one build-breaking missing citation, a few missing entries/typos, and a set of minor redundant-op / per-iteration cleanups.

**Bottom line:** roughly one focused revision away from merge. The hard, valuable part (a correct batched/differentiable LSMR) is done; what remains is fixing ~7 mostly-small correctness issues, restoring and broadening the tests to the currently-uncovered regimes, and finishing the docs. I would not merge as-is, but I would encourage the author — the foundation is good.
