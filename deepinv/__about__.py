from importlib.metadata import metadata as importlib_metadata

metadata = importlib_metadata("deepinv")
__title__ = metadata["Name"]
__summary__ = metadata["Summary"]
__version__ = metadata["Version"]
__author__ = metadata["Author"]
__license__ = metadata.get(
    "License", metadata.get("License-Expression", "BSD-3-Clause")
)
__url__ = metadata["Project-URL"]

__all__ = [
    "__title__",
    "__summary__",
    "__version__",
    "__author__",
    "__license__",
    "__url__",
]
