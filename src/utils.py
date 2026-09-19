"""Cross-track utilities. Stable API: do not change signatures without checking every track."""

from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config and enforce the one non-negotiable field."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"FATAL: config not found: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"FATAL: config must be a mapping: {config_path}")
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    return config


def init_directory(output_dir: str | Path, overwrite: bool = False) -> Path:
    """Create the output directory. Refuse to clobber an existing one unless --overwrite.

    On overwrite, a ``downstream/`` subdirectory is PRESERVED: it holds runs that depend on
    this directory's outputs (the downstream pattern in docs/repo_usage.md), and rebuilding an
    upstream must never destroy hours of downstream compute. Everything else is removed.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"FATAL: output_dir exists: {output_dir}. Pass --overwrite to replace it.")
        downstream = output_dir / "downstream"
        for child in output_dir.iterdir():
            if child == downstream:
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        if downstream.exists():
            print(f"init_directory: overwrote {output_dir} but preserved {downstream}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def copy_config(config_path: str | Path, output_dir: str | Path) -> None:
    """Record the exact config used for a run."""
    shutil.copy(config_path, Path(output_dir) / "config.yaml")
