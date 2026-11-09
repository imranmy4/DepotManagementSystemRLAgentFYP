"""Repo-root-anchored paths, each overridable by an environment variable."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("DEPOT_MODELS_DIR", ROOT / "models"))
LOGS_DIR = Path(os.environ.get("DEPOT_LOGS_DIR", ROOT / "logs"))
CONFIG_INI = Path(os.environ.get("DEPOT_CONFIG_INI", ROOT / "configs" / "config.ini"))
DEPLOY_MODEL_DIR = Path(os.environ.get("DEPOT_DEPLOY_MODEL_DIR", ROOT / "deploy" / "model"))
