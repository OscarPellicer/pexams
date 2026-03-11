from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pexams")
except PackageNotFoundError:
    __version__ = "unknown"
