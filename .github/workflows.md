# GitHub Actions Workflows

This document describes all CI/CD workflows of the library.

---

## Testing

### `test_cpu.yml` — Test CPU
**Trigger:** Pull requests targeting `main`.

Runs the full test suite on every pull request. Installs `deepinv` using [`pixi`](https://pixi.sh).
We use pixi to manage both `conda` and `pip` dependencies in a single workflow.
Tests are executed in parallel (`-n 2`) on both **Ubuntu** and **Windows** without and with all optional dependencies.
Uses [pytest-testmon](https://github.com/tarpas/pytest-testmon) to skip tests unaffected by the changes, speeding up CI.
Coverage reports are uploaded to [Codecov](https://codecov.io). Also runs doctests in `.py` files from inline module docstrings (`--doctest-modules`).


---

### `test_gpu.yml` — Test GPU
**Triggers:** 

- PRs: Maintainer writing `gpu-tests` on a PR comment (via `gpu_trigger.yml`), or manual dispatch specifying a PR number.
- `main`: pushes, daily schedule (02:30 UTC)

Runs the full test suite on a **self-hosted GPU runner**. Uses testmon caching, saves cache only on `main`. Posts a success/failure comment back to the PR when triggered for a specific PR number.

---


## Documentation

### `docs_cpu.yml` — Build Docs (CPU)
**Trigger:** Pushes to `main` and PRs.

Builds the Sphinx documentation. For PRs, it intelligently determines which gallery examples need to be rebuilt by
running a diff script (`.github/scripts/diff_sphinx_gallery.py`) against the base branch — unless a maintainer has
posted a `test-examples` comment on the PR, in which case all examples are rebuilt. Also:
- Converts all Python examples in `examples/` to Jupyter notebooks.
- Uploads the built docs as a GitHub Actions artifact.
- Runs doctests on `.rst` files under `docs/`.

---

### `docs_gpu.yml` — GPU Docs
**Trigger:** 

- `main`: pushes, daily schedule (02:30 UTC).
- PRs: manual dispatch specifying a PR number or maintainer writing `gpu-tests` on a PR comment (via `gpu_trigger.yml`).

Builds the full documentation on a **self-hosted GPU runner**, ensuring that all examples run correctly in a GPU environment. Follows the same smart diff logic as `docs_cpu.yml` for selective example rebuilding. Additionally:
- Runs Sphinx doctests (with support to Colab examples).
- Deploys the built docs to **GitHub Pages** (`gh-pages` branch) on pushes to `main`.
- Posts a success/failure comment with artifact and run links back to the PR when triggered for a specific PR.

---

## Linting & Code Quality

### `lint.yml` — Lint
**Trigger:** Pushes to `main` and pull requests targeting `main`.

Enforces code style and quality:
- **[black](https://black.readthedocs.io):** Checks code formatting.
- **[ruff](https://docs.astral.sh/ruff/):** Fast Python linter for common errors and style issues.

---

## Install & Import Checks

### `test_install.yml` — Test Install Methods
**Trigger:** Manual dispatch or monthly schedule.

Verifies that `deepinv` installs correctly across the following combinations:
- **Package managers:** `pip`, `uv`, `pixi`, `conda`
- **Sources:** PyPI and the Git repository
- **Dependency sets:** core and full (`dataset`, `denoisers`, `physics` extras)
- **Platforms:** Ubuntu, Windows, macOS
- **Python versions:** 3.10–3.13

Each combination installs the package and verifies a successful `import deepinv`.

---

### `import_time.yml` — Import Time vs main
**Trigger:** Pull requests targeting `main`.

Benchmarks the `import deepinv` time on the PR branch versus `main` using [Hyperfine](https://github.com/sharkdp/hyperfine). Runs for two dependency configurations: all optional dependencies and none. Fails if the PR introduces a **>5% regression** in import time. Also measures `torch + torchvision` import time as a baseline reference.

---

## Release & Deployment

### `release.yml` — Auto Publish to PyPI
**Trigger:** Manual dispatch by one of the maintainers, selecting a bump type (`patch`, `minor`, `major`, or `dev`).

Automates the release process to PyPI:
1. Updates the changelog from `docs/source/changelog.rst`.
2. Bumps the version in `pyproject.toml` using the selected bump type (e.g., `patch` increments the patch version, `minor` increments the minor version, etc.).
3. Opens a pull request (`release-branch` → `main`) titled "Release vX.Y.Z" with the version and changelog changes.
4. Creates a GitHub Release with the changelog as release notes.

The `dev` bump type skips the PR and GitHub Release steps (for debugging).

---

### `upload_pypi.yml` — Deploy to PyPI
**Trigger:** Closure of a pull request where the source branch is `release-branch` and the target is `main`, and the PR was merged.

Publishes the package to PyPI after a release PR is merged:
1. Builds the distribution with `python -m build`.
2. Uploads to **TestPyPI** and verifies the install.
3. Uploads to **PyPI**.


---

## Triggers & Utilities

### `gpu_trigger.yml` — PR Comment Trigger
**Trigger:** Comments on issues/PRs (filtered by content), or manual dispatch.

A dispatcher workflow that lets maintainers trigger expensive GPU workflows from PR comments, without requiring direct access to `workflow_dispatch`. Two comment commands are supported:

- **`/gpu-tests`** — Triggers `test_gpu.yml` and `docs_gpu.yml` for the commented PR, then posts a confirmation comment. Only users with `write` or `admin` repository permissions can trigger this.
- **`/test-examples`** — Triggers `docs_cpu.yml` to rebuild all documentation examples for the PR.

Adds a 🚀 reaction to the triggering comment as acknowledgement.

---

### `check_citation.yml` — CITATION.cff
**Trigger:** Pushes that modify `CITATION.cff`, or manual dispatch.

Validates the `CITATION.cff` file using the [`cff-validator`](https://github.com/dieghernan/cff-validator) action to ensure the citation metadata is well-formed and complies with the [Citation File Format](https://citation-file-format.github.io/) spec.

