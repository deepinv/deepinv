try:
    import tomllib
except ImportError:
    import tomli as tomllib

with open("pyproject.toml", "rb") as f:
    metadata = tomllib.load(f)["project"]

print(metadata["version"])
