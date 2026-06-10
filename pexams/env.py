from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv() -> None:
    """Load .env files from the current workspace and package parents."""
    seen: set[Path] = set()
    starts = [Path.cwd(), Path(__file__).resolve()]
    for start in starts:
        folder = start if start.is_dir() else start.parent
        for candidate in (folder, *folder.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            load_dotenv(candidate / ".env")
