from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------
# If you later publish to PyPI, you can switch to importlib.metadata.version
__version__ = "0.1.0"

# ---------------------------------------------------------------------
# Paths helper
# ---------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
# With a src/ layout, project root is the parent of the package directory
_PROJECT_ROOT = _PACKAGE_DIR.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DEFAULT_CONFIG_PATH = _CONFIG_DIR / "config.yaml"
_DATA_DIR = _PROJECT_ROOT / "data"
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
_NOTEBOOKS_DIR = _PROJECT_ROOT / "notebooks"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
_TESTS_DIR = _PROJECT_ROOT / "tests"


def paths() -> Dict[str, Path]:
    """Return important project paths as a dict."""
    return {
        "package_dir": _PACKAGE_DIR,
        "project_root": _PROJECT_ROOT,
        "config_dir": _CONFIG_DIR,
        "default_config": _DEFAULT_CONFIG_PATH,
        "data_dir": _DATA_DIR,
        "artifacts_dir": _ARTIFACTS_DIR,
        "notebooks_dir": _NOTEBOOKS_DIR,
        "scripts_dir": _SCRIPTS_DIR,
        "tests_dir": _TESTS_DIR,
    }


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
# Attach a null handler by default to avoid "No handler found" warnings
logging.getLogger(__name__).addHandler(logging.NullHandler())


def get_logger(name: Optional[str] = None, level: Optional[int] = None) -> logging.Logger:
    """
    Create/get a logger with sensible defaults.
    - name=None returns the package logger.
    - level, if provided, sets the logger level (e.g., logging.INFO).
    """
    logger_name = f"{__name__}" if name in (None, "", __name__) else f"{__name__}.{name}"
    logger = logging.getLogger(logger_name)
    if level is not None:
        logger.setLevel(level)

    # If no handlers attached (e.g., first call), add a simple stream handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load YAML config.
    If `path` is None, it tries `config/config.yaml` at the project root.
    Returns an empty dict if the file is not found.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        get_logger("config").warning("Config file not found: %s (returning empty dict)", cfg_path)
        return {}

    try:
        import yaml  # PyYAML
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to load configuration. Please install `pyyaml`."
        ) from e

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# ---------------------------------------------------------------------
# Lazy imports of submodules to keep top-level import fast & clean
# ---------------------------------------------------------------------
def __getattr__(name: str):
    """
    Lazily import submodules so `import package; package.model` works
    without importing everything at once.
    """
    if name in {"data_loader", "model", "train"}:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ["data_loader", "model", "train"])


# What we publicly expose at package level
__all__ = [
    "__version__",
    "paths",
    "get_logger",
    "load_config",
    # submodules (via lazy import)
    "data_loader",
    "model",
    "train",
]
