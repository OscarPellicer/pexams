from pathlib import Path

from dotenv import load_dotenv


def load_project_dotenv(*extra_starts) -> None:
    """Load .env files from the current folder, this package and ``extra_starts`` (files or
    folders, e.g. another package's ``__file__``), and all their parents."""
    seen: set[Path] = set()
    starts = [Path.cwd(), Path(__file__).resolve(), *(Path(p).resolve() for p in extra_starts)]
    for start in starts:
        folder = start if start.is_dir() else start.parent
        for candidate in (folder, *folder.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            load_dotenv(candidate / ".env")
