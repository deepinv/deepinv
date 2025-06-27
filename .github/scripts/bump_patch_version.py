import re
from pathlib import Path


def increment_version(version):
    """Increment the patch version number (last digit)."""
    parts = version.split(".")

    try:
        major, minor, patch = map(int, parts)
    except ValueError:
        raise ValueError("Version components must be integers")

    patch += 1
    return f"{major}.{minor}.{patch}"


file_path = "pyproject.toml"

path = Path(file_path)
if not path.exists():
    raise FileNotFoundError(f"{file_path} does not exist")

# Read the file content while preserving line endings
with open(path, "r", newline="") as f:
    lines = f.readlines()

# More precise pattern that matches the whole version line
version_pattern = re.compile(
    r'^(\s*version\s*=\s*")(\d+\.\d+\.\d+)("\s*(?:#.*)?$)', re.IGNORECASE
)

version_line_index = None
old_version = None

# Find and update the version line
for i, line in enumerate(lines):
    match = version_pattern.match(line)
    if match:
        version_line_index = i
        prefix, old_version, suffix = match.groups()
        new_version = increment_version(old_version)
        lines[i] = f"{prefix}{new_version}{suffix}\n"
        break

if version_line_index is None:
    raise ValueError(
        "Could not find version in pyproject.toml\n"
        'Expected format: version = "X.Y.Z" (with optional spaces/comments)'
    )

# Write back to file only if we found and modified the version
with open(path, "w", newline="") as f:
    f.writelines(lines)

print(f"Updated version from {old_version} to {new_version}")
