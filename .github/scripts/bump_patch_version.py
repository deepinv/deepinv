try:
    import tomllib
except ImportError:
    import tomli as tomllib

with open("pyproject.toml", "rb") as f:
    data = tomllib.load(f)

version = data["project"]["version"]
major, minor, patch = map(int, version.split("."))
patch += 1
new_version = f"{major}.{minor}.{patch}"
data["project"]["version"] = new_version
with open(file_path, "w") as f:
    toml.dump(data, f)
print(f"Bumped version to {new_version}")
