try:
    import tomllib
except ImportError:
    import tomli as tomllib


def get_version_from_toml(filename):
    with open(filename, "rb") as f:
        metadata = tomllib.load(f)["project"]
    return metadata["version"]

if __name__ == "__main__":
    version = get_version_from_toml("../../pyproject.toml")
    print(version)
