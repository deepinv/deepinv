try:
    import tomllib
except ImportError:
    import tomli as tomllib
    
data = tomllib.load("pyproject.toml")
print(data["project"]["version"])
