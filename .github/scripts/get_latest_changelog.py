import re
import sys
from get_version import get_version_from_toml
from bump_version import increment_version


def rst_to_md(rst_text):
    lines = rst_text.strip().splitlines()
    md_lines = []
    skip_next = False

    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue

        # Convert section headers
        if line.strip() in {"New Features", "Changed", "Fixed"}:
            md_lines.append(f"### {line.strip()}")
            skip_next = True  # Skip the underline in the next line
            continue

        # Remove underline-only lines (that slipped through)
        if re.match(r"^[-=^~]{3,}$", line.strip()):
            continue

        # Convert references
        line = re.sub(
            r":gh:`(\d+)`", r"[#\1](https://github.com/deepinv/deepinv/pull/\1)", line
        )
        line = re.sub(r"`([^`]+)`_", r"\1", line)
        line = re.sub(r":class:`([^`]+)`", r"`\1`", line)

        md_lines.append(line)

    return "\n".join(md_lines).strip()


def update_changelog(changelog_path, pyproject_path, increment="patch"):
    r"""
    Function for automatic release changelog generation.
    """

    with open(changelog_path, "r", encoding="utf-8") as f:
        changelog = f.read()

    # Extract the "Current" section
    match = re.search(
        r"(?P<header>^Current\s*-+\s*\n)(?P<body>.*?)(?=^v?\d+\.\d+\.\d+\s*-+\s*\n)",
        changelog,
        re.DOTALL | re.MULTILINE,
    )

    if not match:
        print("Could not find a 'Current' section.")
        sys.exit(1)

    header = match.group("header")
    body = match.group("body").rstrip()

    # Extract current version from pyproject.toml or bump script
    version = get_version_from_toml(pyproject_path)
    print("current version: ", version)
    version = increment_version(version, increment=increment)
    print("new version: ", version)
    version_header = f"v{version}\n{'-' * (len(version)+1)}\n"

    # Write extracted changes to changelog.txt
    with open("changelog.txt", "w", encoding="utf-8") as f:
        f.write(rst_to_md(body).strip())

    # Prepare a new empty "Current" section
    new_current = (
        "Current\n"
        "-------\n\n"
        "New Features\n^^^^^^^^^^^^\n\n"
        "Changed\n^^^^^^^\n\n"
        "Fixed\n^^^^^\n\n"
    )

    # Replace the old "Current" section with the empty one and insert the versioned section below it
    updated_changelog = changelog.replace(
        header + body, new_current + "\n" + version_header + body.strip() + "\n\n", 1
    )

    with open(changelog_path, "w", encoding="utf-8") as f:
        f.write(updated_changelog)


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print(
            "Usage: python get_latest_changelog.py path/to/changelog.rst pyproject_path  patch"
        )
        sys.exit(1)
    file_path = sys.argv[1]
    increment = sys.argv[2] if len(sys.argv) > 2 else "patch"

    update_changelog(sys.argv[1], increment)
    print("changelog.rst updated.")
