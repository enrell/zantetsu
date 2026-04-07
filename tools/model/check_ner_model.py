#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from tools._dispatch import run

    run("zantetsu_tools.model.check_ner_model")


if __name__ == "__main__":
    main()
