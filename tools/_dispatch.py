from __future__ import annotations

import importlib
import sys
from pathlib import Path


def run(module_name: str, entrypoint: str = "main") -> None:
    """Load a packaged tool entrypoint from `python/zantetsu_tools`."""
    repo_root = Path(__file__).resolve().parents[1]
    python_root = repo_root / "python"
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    module = importlib.import_module(module_name)
    getattr(module, entrypoint)()
