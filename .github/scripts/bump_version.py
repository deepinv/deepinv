from pathlib import Path
import tomlkit


def increment_version(version, increment="patch"):
    """Increment the version number.
    :param str version: The current version string in the format "major.minor.patch".
    :param str increment: The part of the version to increment, can be "major", "minor", or "patch".
    """
    parts = version.split(".")

    try:
        major, minor, patch = map(int, parts)
    except ValueError:
        raise ValueError("Version components must be integers")

    if increment == "major":
        major += 1
    elif increment == "minor":
        minor += 1
    elif increment == "patch" or increment == "dev":
        patch += 1
    else:
        raise ValueError("Increment must be 'major', 'minor', 'patch' or 'dev'")
    return (
        f"{major}.{minor}.{patch}"
        if increment != "dev"
        else f"{major}.{minor}.{patch}.dev"
    )


def bump_patch_version(file_path, increment="patch"):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    # Read the file content while preserving line endings
    with open(file_path, "r", encoding="utf-8") as f:
        # Load the TOML file
        toml_content = f.read()
        metadata = tomlkit.parse(toml_content)

    current_version = metadata["project"].get("version")
    if not current_version:
        raise ValueError("Version not found in the provided file")
    new_version = increment_version(current_version, increment=increment)
    metadata["project"]["version"] = new_version
    # Write the updated version back to the file
    tomlkit.dump(metadata, path.open("w", encoding="utf-8"))

    return new_version


if __name__ == "__main__":
    import sys

    if len(sys.argv) not in [2, 3]:
        print("Usage: python bump_version.py path/to/pyproject.toml patch")
        sys.exit(1)
    file_path = sys.argv[1]
    increment = sys.argv[2] if len(sys.argv) > 2 else "patch"
    new_version = bump_patch_version(file_path, increment)

    print(f"Updated version to {new_version}")
