"""Reproducibility metadata for serialized experiment results."""

from __future__ import annotations

import json
import platform
from importlib import resources
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


def load_artifact_schema() -> dict[str, object]:
    """Load the JSON Schema matching :data:`ARTIFACT_SCHEMA_VERSION`."""

    schema_file = resources.files("qml_qiskit.schemas").joinpath("result-v1.schema.json")
    return json.loads(schema_file.read_text(encoding="utf-8"))


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
