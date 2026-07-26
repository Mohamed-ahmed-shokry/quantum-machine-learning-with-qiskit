"""Reproducibility metadata for serialized experiment results."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from typing import cast

ARTIFACT_SCHEMA_VERSION = 1
TRACKED_PACKAGES = (
    "qml-qiskit",
    "qiskit",
    "qiskit-machine-learning",
    "numpy",
    "scikit-learn",
)


def artifact_identifier(payload: object) -> str:
    """Return a stable SHA-256 identifier for JSON-serializable artifact content."""

    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


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
    return cast(dict[str, object], json.loads(schema_file.read_text(encoding="utf-8")))


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"
