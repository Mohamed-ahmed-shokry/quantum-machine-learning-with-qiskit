"""Loading and reconstruction of saved experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

from jsonschema import Draft202012Validator, ValidationError

from qml_qiskit.metadata import load_artifact_schema, verify_artifact_identifier
from qml_qiskit.models import BenchmarkResult, ModelMetrics
from qml_qiskit.report import render_html_report
from qml_qiskit.study import (
    MetricSummary,
    StudyResult,
    _summarize,
    _two_sided_sign_test_pvalue,
)

SEMANTIC_TOLERANCE = 1e-12


class ArtifactLoadError(ValueError):
    """Raised when a saved experiment artifact cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """A reconstructed result with its saved runtime provenance."""

    result: BenchmarkResult | StudyResult
    runtime: dict[str, object]
    artifact_id: str
    schema_version: int

    def render_html(self) -> str:
        """Render a report using the artifact's original runtime metadata."""

        return render_html_report(
            self.result,
            runtime=self.runtime,
            artifact_id=self.artifact_id,
        )


def load_artifact(path: str | Path) -> LoadedArtifact:
    """Load, validate, verify, and reconstruct a saved JSON artifact."""

    artifact_path = Path(path)
    try:
        serialized = artifact_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactLoadError(f"could not read artifact {artifact_path}: {error}") from error
    try:
        loaded = json.loads(serialized)
    except json.JSONDecodeError as error:
        raise ArtifactLoadError(f"invalid JSON in artifact {artifact_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ArtifactLoadError(f"artifact {artifact_path} must contain a JSON object")

    payload = cast(dict[str, Any], loaded)
    try:
        Draft202012Validator(load_artifact_schema()).validate(payload)
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        suffix = f" at {location}" if location else ""
        raise ArtifactLoadError(
            f"artifact schema validation failed{suffix}: {error.message}"
        ) from error

    artifact_id = payload.get("artifact_id")
    if not isinstance(artifact_id, str):
        raise ArtifactLoadError("artifact is missing artifact_id")
    if not verify_artifact_identifier(payload):
        raise ArtifactLoadError("artifact_id does not match measured content")

    result = _load_study(payload) if "benchmarks" in payload else _load_benchmark(payload)
    _validate_semantics(payload, result)
    return LoadedArtifact(
        result=result,
        runtime=cast(dict[str, object], payload["runtime"]),
        artifact_id=artifact_id,
        schema_version=cast(int, payload["schema_version"]),
    )


def _validate_semantics(
    payload: dict[str, Any],
    result: BenchmarkResult | StudyResult,
) -> None:
    if isinstance(result, StudyResult):
        _validate_study_semantics(payload, result)
    else:
        _validate_benchmark_semantics(payload, result, "benchmark")


def _validate_benchmark_semantics(
    payload: dict[str, Any],
    result: BenchmarkResult,
    path: str,
) -> None:
    _require_close(
        float(payload["quantum_advantage"]),
        result.quantum_advantage,
        f"{path}.quantum_advantage",
    )


def _validate_study_semantics(payload: dict[str, Any], result: StudyResult) -> None:
    payload_benchmarks = cast(list[dict[str, Any]], payload["benchmarks"])
    if len(result.seeds) != len(result.benchmarks):
        _semantic_error("seeds and benchmarks must have the same length")

    for index, (payload_benchmark, benchmark) in enumerate(
        zip(payload_benchmarks, result.benchmarks, strict=True)
    ):
        path = f"benchmarks[{index}]"
        _validate_benchmark_semantics(payload_benchmark, benchmark, path)
        if benchmark.seed != result.seeds[index]:
            _semantic_error(f"{path}.seed does not match seeds[{index}]")
        for field_name in ("samples", "features", "feature_map_reps"):
            if getattr(benchmark, field_name) != getattr(result, field_name):
                _semantic_error(f"{path}.{field_name} does not match the study")
        _require_optional_close(benchmark.noise, result.noise, f"{path}.noise")
        _require_optional_close(benchmark.test_size, result.test_size, f"{path}.test_size")

    advantages = tuple(benchmark.quantum_advantage for benchmark in result.benchmarks)
    quantum_wins = sum(
        advantage > 0 and not isclose(advantage, 0, abs_tol=SEMANTIC_TOLERANCE)
        for advantage in advantages
    )
    classical_wins = sum(
        advantage < 0 and not isclose(advantage, 0, abs_tol=SEMANTIC_TOLERANCE)
        for advantage in advantages
    )
    ties = len(advantages) - quantum_wins - classical_wins
    for name, actual, expected in (
        ("quantum_wins", result.quantum_wins, quantum_wins),
        ("ties", result.ties, ties),
        ("classical_wins", result.classical_wins, classical_wins),
    ):
        if actual != expected:
            _semantic_error(f"{name} does not match the paired benchmark outcomes")

    advantage_mean = fmean(advantages)
    if isclose(advantage_mean, 0, abs_tol=SEMANTIC_TOLERANCE):
        advantage_mean = 0.0
    _require_close(
        result.quantum_advantage_mean,
        advantage_mean,
        "quantum_advantage_mean",
    )
    _require_close(
        result.quantum_advantage_std,
        pstdev(advantages),
        "quantum_advantage_std",
    )
    _validate_metric_summary(
        result.classical,
        _summarize(tuple(benchmark.classical for benchmark in result.benchmarks)),
        "classical",
    )
    _validate_metric_summary(
        result.quantum,
        _summarize(tuple(benchmark.quantum for benchmark in result.benchmarks)),
        "quantum",
    )
    _require_close(
        result.sign_test_pvalue,
        _two_sided_sign_test_pvalue(quantum_wins, classical_wins),
        "sign_test_pvalue",
    )


def _validate_metric_summary(
    actual: MetricSummary,
    expected: MetricSummary,
    path: str,
) -> None:
    if actual.name != expected.name:
        _semantic_error(f"{path}.name does not match the paired benchmarks")
    if actual.runs != expected.runs:
        _semantic_error(f"{path}.runs does not match the number of paired benchmarks")
    for field_name in (
        "train_accuracy_mean",
        "train_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "fit_seconds_mean",
        "fit_seconds_std",
        "support_vectors_mean",
        "support_vectors_std",
    ):
        _require_close(
            float(getattr(actual, field_name)),
            float(getattr(expected, field_name)),
            f"{path}.{field_name}",
        )


def _require_optional_close(
    actual: float | None,
    expected: float | None,
    path: str,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            _semantic_error(f"{path} does not match the study")
        return
    _require_close(actual, expected, path)


def _require_close(actual: float, expected: float, path: str) -> None:
    if not isclose(actual, expected, rel_tol=SEMANTIC_TOLERANCE, abs_tol=SEMANTIC_TOLERANCE):
        _semantic_error(f"{path} is inconsistent with the underlying benchmarks")


def _semantic_error(message: str) -> None:
    raise ArtifactLoadError(f"artifact semantic validation failed: {message}")


def _load_model_metrics(payload: dict[str, Any]) -> ModelMetrics:
    return ModelMetrics(
        name=cast(str, payload["name"]),
        train_accuracy=float(payload["train_accuracy"]),
        test_accuracy=float(payload["test_accuracy"]),
        fit_seconds=float(payload["fit_seconds"]),
        support_vectors=cast(int, payload["support_vectors"]),
    )


def _load_benchmark(payload: dict[str, Any]) -> BenchmarkResult:
    noise = payload.get("noise")
    test_size = payload.get("test_size")
    return BenchmarkResult(
        samples=cast(int, payload["samples"]),
        features=cast(int, payload["features"]),
        seed=cast(int, payload["seed"]),
        feature_map_reps=cast(int, payload["feature_map_reps"]),
        classical=_load_model_metrics(cast(dict[str, Any], payload["classical"])),
        quantum=_load_model_metrics(cast(dict[str, Any], payload["quantum"])),
        noise=None if noise is None else float(noise),
        test_size=None if test_size is None else float(test_size),
    )


def _load_metric_summary(payload: dict[str, Any]) -> MetricSummary:
    return MetricSummary(
        name=cast(str, payload["name"]),
        runs=cast(int, payload["runs"]),
        train_accuracy_mean=float(payload["train_accuracy_mean"]),
        train_accuracy_std=float(payload["train_accuracy_std"]),
        test_accuracy_mean=float(payload["test_accuracy_mean"]),
        test_accuracy_std=float(payload["test_accuracy_std"]),
        fit_seconds_mean=float(payload["fit_seconds_mean"]),
        fit_seconds_std=float(payload["fit_seconds_std"]),
        support_vectors_mean=float(payload["support_vectors_mean"]),
        support_vectors_std=float(payload["support_vectors_std"]),
    )


def _load_study(payload: dict[str, Any]) -> StudyResult:
    benchmarks = tuple(
        _load_benchmark(cast(dict[str, Any], benchmark))
        for benchmark in cast(list[object], payload["benchmarks"])
    )
    quantum_wins = cast(int, payload["quantum_wins"])
    classical_wins = cast(int, payload["classical_wins"])
    noise = payload.get("noise")
    test_size = payload.get("test_size")
    return StudyResult(
        samples=cast(int, payload["samples"]),
        features=cast(int, payload["features"]),
        seeds=tuple(cast(list[int], payload["seeds"])),
        feature_map_reps=cast(int, payload["feature_map_reps"]),
        classical=_load_metric_summary(cast(dict[str, Any], payload["classical"])),
        quantum=_load_metric_summary(cast(dict[str, Any], payload["quantum"])),
        quantum_advantage_mean=float(payload["quantum_advantage_mean"]),
        quantum_advantage_std=float(payload["quantum_advantage_std"]),
        quantum_wins=quantum_wins,
        ties=cast(int, payload["ties"]),
        classical_wins=classical_wins,
        benchmarks=benchmarks,
        noise=None if noise is None else float(noise),
        test_size=None if test_size is None else float(test_size),
        sign_test_pvalue=float(
            payload.get(
                "sign_test_pvalue",
                _two_sided_sign_test_pvalue(quantum_wins, classical_wins),
            )
        ),
    )
