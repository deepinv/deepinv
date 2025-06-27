try:
    import tomllib
except ImportError:
    import tomli as tomllib

with open("pyproject.toml", "rb") as f:
    metadata = tomllib.load(f)["project"]

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
authors = ""
for author in metadata["authors"]:
    if authors:
        authors += ", "
    authors += author["name"]
__author__ = authors
__license__ = metadata["license"]["text"]
__url__ = metadata["urls"]["Homepage"]
