import tomlkit

with open("../pyproject.toml", "rb") as f:
    metadata = tomlkit.load(f)["project"]

__all__ = [
    "__title__",
    "__summary__",
    "__url__",
    "__version__",
    "__author__",
    "__license__",
]

__title__ = metadata["name"]
__summary__ = metadata["description"]
__version__ = metadata["version"]
__author__ = ", ".join(auth["name"] for auth in metadata["authors"])
__license__ = metadata["license"]["text"]
__url__ = metadata["urls"]["Homepage"]
