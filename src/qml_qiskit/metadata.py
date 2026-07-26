"""Reproducibility metadata for serialized experiment results."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from hashlib import sha256
from hmac import compare_digest
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


def verify_artifact_identifier(payload: Mapping[str, object]) -> bool:
    """Return whether an artifact ID matches its measured, non-runtime content."""

    claimed_identifier = payload.get("artifact_id")
    if not isinstance(claimed_identifier, str):
        return False

    content = dict(payload)
    content.pop("artifact_id")
    content.pop("schema_version", None)
    content.pop("runtime", None)
    benchmarks = content.get("benchmarks")
    if benchmarks is not None and (
        not isinstance(benchmarks, list)
        or not all(
            isinstance(benchmark, Mapping) and verify_artifact_identifier(benchmark)
            for benchmark in benchmarks
        )
    ):
        return False

    try:
        expected_identifier = artifact_identifier(content)
    except (TypeError, ValueError):
        return False
    return compare_digest(claimed_identifier, expected_identifier)


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
