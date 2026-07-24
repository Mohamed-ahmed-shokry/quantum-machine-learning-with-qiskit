"""Tests for serialized experiment environment metadata."""

from importlib.metadata import PackageNotFoundError

import qml_qiskit.metadata as metadata


def test_runtime_metadata_tracks_reproducibility_dependencies() -> None:
    result = metadata.runtime_metadata()
    packages = result["packages"]

    assert metadata.ARTIFACT_SCHEMA_VERSION == 1
    assert result["python"]
    assert result["platform"]
    assert set(packages) == set(metadata.TRACKED_PACKAGES)
    assert packages["qml-qiskit"] == "1.0.0"
    assert all(version != "not-installed" for version in packages.values())


def test_runtime_metadata_marks_missing_packages(monkeypatch) -> None:
    def missing_version(_package: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing_version)

    assert metadata._package_version("missing-example") == "not-installed"
