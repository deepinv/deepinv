# GitHub Actions Workflows

This document describes all CI/CD workflows of the library.

---

## Testing

### `test_pr.yml` — Test PRs
**Trigger:** Pull requests targeting `main`.

Runs the full test suite on every pull request using [Pixi](https://pixi.sh) environments. Tests are executed in parallel (`-n 2`) on both **Ubuntu** and **Windows** with Python 3.12 and all optional dependencies. Uses [pytest-testmon](https://github.com/tarpas/pytest-testmon) to skip tests unaffected by the changes, speeding up CI. Coverage reports are uploaded to [Codecov](https://codecov.io). Also runs doctests from inline module docstrings (`--doctest-modules`).

---

### `test_recurrent_main.yml` — Test main
**Trigger:** Daily schedule (03:30 UTC) and every push to `main`.

Nightly full test suite run on the `main` branch. Covers three configurations:
- Ubuntu with all optional dependencies
- Ubuntu with no optional dependencies
- Windows with all optional dependencies

Uses testmon caching to accelerate repeated runs, and uploads coverage + test results to Codecov. Only runs on the `deepinv/deepinv` repository (not forks).

---

### `test_gpu.yml` — Test GPU
**Trigger:** Daily schedule (02:30 UTC), pushes to `main`, or manual dispatch specifying a PR number.

Runs the full test suite on a **self-hosted GPU runner**. The PR to test can be specified via `workflow_dispatch` (e.g., triggered by `gpu_trigger.yml`). Uses testmon caching, saves cache only on `main`. Posts a success/failure comment back to the PR when triggered for a specific PR number.

---


## Documentation

### `documentation.yml` — Build Docs (CPU)
**Trigger:** Pushes to `main` and pull requests targeting `main`.

Builds the Sphinx documentation using the `docs` Pixi environment. For pull requests, it intelligently determines which gallery examples need to be rebuilt by running a diff script (`.github/scripts/diff_sphinx_gallery.py`) against the base branch — unless a maintainer has posted a `test-examples` comment on the PR, in which case all examples are rebuilt. Also:
- Converts all Python examples in `examples/` to Jupyter notebooks.
- Uploads the built docs as a GitHub Actions artifact.
- Runs doctests on `.rst` files under `docs/`.

---

### `gpu_docs.yml` — GPU Docs
**Trigger:** Daily schedule (02:30 UTC), pushes to `main`, or manual dispatch specifying a PR number.

Builds the full documentation on a **self-hosted GPU runner**, ensuring that all examples run correctly in a GPU environment. Follows the same smart diff logic as `documentation.yml` for selective example rebuilding. Additionally:
- Runs Sphinx doctests.
- Converts examples to notebooks.
- Deploys the built docs to **GitHub Pages** (`gh-pages` branch) on pushes to `main`.
- Posts a success/failure comment with artifact and run links back to the PR when triggered for a specific PR.

---

## Linting & Code Quality

### `lint.yml` — Lint
**Trigger:** Pushes to `main` and pull requests targeting `main`.

Enforces code style and quality:
- **[Black](https://black.readthedocs.io):** Checks code formatting.
- **[Ruff](https://docs.astral.sh/ruff/):** Fast Python linter for common errors and style issues.

---

## Install & Import Checks

### `test_install.yml` — Test Install Methods
**Trigger:** Manual dispatch or quarterly schedule (1st of every 4th month).

Verifies that `deepinv` installs correctly across a broad matrix of:
- **Package managers:** `pip`, `uv`, `pixi`, `conda`
- **Sources:** PyPI and the Git repository
- **Dependency sets:** core and full (`dataset`, `denoisers`, `physics` extras)
- **Platforms:** Ubuntu, Windows, macOS
- **Python versions:** 3.10–3.14

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

- **`/gpu-tests`** — Triggers `test_gpu.yml` and `gpu_docs.yml` for the commented PR, then posts a confirmation comment. Only users with `write` or `admin` repository permissions can trigger this.
- **`/test-examples`** — Triggers `documentation.yml` to rebuild all documentation examples for the PR.

Adds a 🚀 reaction to the triggering comment as acknowledgement.

---

### `check_citation.yml` — CITATION.cff
**Trigger:** Pushes that modify `CITATION.cff`, or manual dispatch.

Validates the `CITATION.cff` file using the [`cff-validator`](https://github.com/dieghernan/cff-validator) action to ensure the citation metadata is well-formed and complies with the [Citation File Format](https://citation-file-format.github.io/) spec.

