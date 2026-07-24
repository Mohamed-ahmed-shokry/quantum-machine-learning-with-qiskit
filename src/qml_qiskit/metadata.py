"""Reproducibility metadata for serialized experiment results."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

ARTIFACT_SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "qml-qiskit",
    "qiskit",
    "qiskit-machine-learning",
    "numpy",
    "scikit-learn",
)


def runtime_metadata() -> dict[str, object]:
    """Describe the interpreter, platform, and core dependency versions."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: _package_version(package) for package in TRACKED_PACKAGES},
    }


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
