"""Tests for the classical and quantum benchmark models."""

from dataclasses import replace

import numpy as np
import pytest

from qml_qiskit import make_moons_split
from qml_qiskit.models import build_feature_map, build_quantum_classifier, run_benchmark


def test_build_feature_map_has_expected_shape() -> None:
    circuit = build_feature_map(2, reps=2)

    assert circuit.num_qubits == 2
    assert circuit.num_parameters == 2
    assert circuit.name == "ZZFeatureMap"


@pytest.mark.parametrize(
    ("features", "reps", "message"),
    [
        (1, 1, "num_features must be at least 2"),
        (2, 0, "reps must be at least 1"),
    ],
)
def test_build_feature_map_rejects_invalid_values(features: int, reps: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_feature_map(features, reps=reps)


def test_build_quantum_classifier_uses_requested_feature_count() -> None:
    classifier = build_quantum_classifier(3, reps=1)

    assert classifier.quantum_kernel.feature_map.num_qubits == 3


def test_run_benchmark_returns_serializable_metrics() -> None:
    data = make_moons_split(samples=32, seed=11)

    result = run_benchmark(data, seed=11, feature_map_reps=1)
    payload = result.as_dict()

    assert result.samples == 32
    assert result.features == 2
    assert result.seed == 11
    assert result.feature_map_reps == 1
    assert result.noise == 0.12
    assert result.test_size == 0.25
    assert 0 <= result.classical.train_accuracy <= 1
    assert 0 <= result.classical.test_accuracy <= 1
    assert 0 <= result.quantum.train_accuracy <= 1
    assert 0 <= result.quantum.test_accuracy <= 1
    assert result.classical.fit_seconds >= 0
    assert result.quantum.fit_seconds >= 0
    assert result.classical.support_vectors > 0
    assert result.quantum.support_vectors > 0
    assert payload["schema_version"] == 1
    assert payload["noise"] == 0.12
    assert payload["test_size"] == 0.25
    assert payload["artifact_id"] == result.artifact_id
    assert len(result.artifact_id) == 64
    assert payload["runtime"]["packages"]["qml-qiskit"] == "1.0.0"
    assert payload["quantum_advantage"] == pytest.approx(
        result.quantum.test_accuracy - result.classical.test_accuracy
    )


def test_benchmark_artifact_id_changes_with_measured_content() -> None:
    result = run_benchmark(make_moons_split(samples=20), feature_map_reps=1)
    changed = replace(result, seed=43)

    assert result.artifact_id != changed.artifact_id


def test_run_benchmark_rejects_mismatched_dataset_seed() -> None:
    data = make_moons_split(samples=20, seed=7)

    with pytest.raises(ValueError, match="seed must match the dataset seed"):
        run_benchmark(data, seed=8, feature_map_reps=1)


def test_run_benchmark_inherits_generated_dataset_seed() -> None:
    data = make_moons_split(samples=20, seed=7)

    result = run_benchmark(data, feature_map_reps=1)

    assert result.seed == 7


def test_run_benchmark_validates_custom_split_before_model_construction() -> None:
    data = replace(make_moons_split(samples=20, seed=7), train_features=np.zeros(15))

    with pytest.raises(ValueError, match="feature arrays must be two-dimensional"):
        run_benchmark(data, feature_map_reps=1)


def test_run_benchmark_rejects_invalid_explicit_seed() -> None:
    data = replace(make_moons_split(samples=20), seed=None)

    with pytest.raises(ValueError, match="seed must be between 0 and 4294967295"):
        run_benchmark(data, seed=-1, feature_map_reps=1)
