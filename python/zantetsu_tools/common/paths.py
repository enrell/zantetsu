from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "models"
TARGET_RELEASE_DIR = REPO_ROOT / "target" / "release"
DEFAULT_ANIMEDB_BASE = "http://localhost:8081"
