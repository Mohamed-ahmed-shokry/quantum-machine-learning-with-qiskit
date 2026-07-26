"""Tests for loading and reconstructing saved experiment artifacts."""

import json

import pytest

from qml_qiskit import (
    ArtifactLoadError,
    load_artifact,
    make_moons_split,
    run_benchmark,
    run_study,
)
from qml_qiskit.metadata import artifact_identifier
from qml_qiskit.models import BenchmarkResult
from qml_qiskit.study import StudyResult


def test_load_benchmark_preserves_result_and_saved_runtime(tmp_path) -> None:
    result = run_benchmark(make_moons_split(samples=20, seed=7), feature_map_reps=1)
    payload = result.as_dict()
    payload["runtime"]["python"] = "saved-python"
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_artifact(path)
    report = loaded.render_html()

    assert isinstance(loaded.result, BenchmarkResult)
    assert loaded.result == result
    assert loaded.artifact_id == payload["artifact_id"]
    assert loaded.schema_version == 1
    assert loaded.runtime["python"] == "saved-python"
    assert loaded.artifact_id in report
    assert "Python saved-python" in report


def test_load_study_reconstructs_nested_benchmarks(tmp_path) -> None:
    result = run_study(samples=20, base_seed=7, runs=2, feature_map_reps=1)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(result.as_dict()), encoding="utf-8")

    loaded = load_artifact(path)

    assert isinstance(loaded.result, StudyResult)
    assert loaded.result == result
    assert loaded.result.benchmarks == result.benchmarks
    assert loaded.result.sign_test_pvalue == result.sign_test_pvalue


def test_load_study_backfills_sign_test_for_earlier_v1_artifact(tmp_path) -> None:
    result = run_study(samples=20, base_seed=7, runs=2, feature_map_reps=1)
    payload = result.as_dict()
    payload.pop("sign_test_pvalue")
    content = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_id", "runtime", "schema_version"}
    }
    payload["artifact_id"] = artifact_identifier(content)
    path = tmp_path / "earlier-study.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_artifact(path)

    assert isinstance(loaded.result, StudyResult)
    assert 0 <= loaded.result.sign_test_pvalue <= 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload.update(samples=7), "schema validation failed"),
        (lambda payload: payload.update(samples=21), "artifact_id does not match"),
        (lambda payload: payload.pop("artifact_id"), "missing artifact_id"),
    ],
)
def test_load_artifact_rejects_invalid_content(tmp_path, change, message) -> None:
    result = run_benchmark(make_moons_split(samples=20), feature_map_reps=1)
    payload = result.as_dict()
    change(payload)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match=message):
        load_artifact(path)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not JSON", "invalid JSON"),
        ("[]", "must contain a JSON object"),
    ],
)
def test_load_artifact_rejects_malformed_files(tmp_path, content, message) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match=message):
        load_artifact(path)


def test_load_artifact_reports_read_failure(tmp_path) -> None:
    with pytest.raises(ArtifactLoadError, match="could not read artifact"):
        load_artifact(tmp_path / "missing.json")
