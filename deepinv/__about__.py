from importlib.metadata import metadata as importlib_metadata

metadata = importlib_metadata("deepinv")
__title__ = metadata["Name"]
__summary__ = metadata["Summary"]
__version__ = metadata["Version"]
__author__ = metadata["Author"]
__license__ = metadata["License"]
__url__ = metadata["Project-URL"]
