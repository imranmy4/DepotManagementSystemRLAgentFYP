"""Repo-root-anchored paths, so scripts work from any working directory.

Each can be overridden by an environment variable (e.g. inside a Docker container
where models are mounted elsewhere).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("DEPOT_MODELS_DIR", ROOT / "models"))
LOGS_DIR = Path(os.environ.get("DEPOT_LOGS_DIR", ROOT / "logs"))
CONFIG_INI = Path(os.environ.get("DEPOT_CONFIG_INI", ROOT / "configs" / "config.ini"))
