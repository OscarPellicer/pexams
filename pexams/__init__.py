from importlib.metadata import PackageNotFoundError, version

from .env import load_project_dotenv

load_project_dotenv()

try:
    __version__ = version("pexams")
except PackageNotFoundError:
    __version__ = "unknown"
