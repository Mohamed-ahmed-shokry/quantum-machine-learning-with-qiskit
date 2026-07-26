"""Tests for serialized experiment environment metadata."""

from importlib.metadata import PackageNotFoundError

import qml_qiskit.metadata as metadata


def test_artifact_identifier_is_stable_and_content_derived() -> None:
    first = metadata.artifact_identifier({"beta": [2, 3], "alpha": 1})
    reordered = metadata.artifact_identifier({"alpha": 1, "beta": [2, 3]})
    changed = metadata.artifact_identifier({"alpha": 1, "beta": [2, 4]})

    assert first == reordered
    assert first != changed
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


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
