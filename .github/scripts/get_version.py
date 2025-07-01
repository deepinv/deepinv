import tomlkit


def get_version_from_toml(filename):
    with open(filename, "r", encoding="utf-8") as f:
        # Load the TOML file
        toml_content = f.read()
        metadata = tomlkit.parse(toml_content)["project"]
    return metadata["version"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python get_version.py path/to/pyproject.toml")
        sys.exit(1)

    file_path = sys.argv[1]
    version = get_version_from_toml(file_path)
    print(version)
