"""Reproduction of the quirk plane search in Knapen et al., arXiv:1708.02243."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"

__all__ = ["PROJECT_ROOT", "CONFIG_DIR", "RESULTS_DIR"]
