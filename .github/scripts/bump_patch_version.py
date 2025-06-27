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

version_pattern = re.compile(r'^\s*version\s*=\s*"(\d+\.\d+\.\d+)"')
version_line_index = None
old_version = None

# Find the version line
for i, line in enumerate(lines):
    match = version_pattern.match(line)
    if match:
        version_line_index = i
        old_version = match.group(1)
        break

if version_line_index is None:
    raise ValueError(
        'Could not find version in pyproject.toml (expected format: version = "X.Y.Z")'
    )

new_version = increment_version(old_version)

# Replace only the version number while preserving the rest of the line
old_line = lines[version_line_index]
new_line = re.sub(r'(")(\d+\.\d+\.\d+)(")', f"\\g<1>{new_version}\\g<3>", old_line)

lines[version_line_index] = new_line

# Write back to file with original line endings
with open(path, "w", newline="") as f:
    f.writelines(lines)

print(f"Updated version from {old_version} to {new_version}")
