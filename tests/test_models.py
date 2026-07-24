"""Tests for the classical and quantum benchmark models."""

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
def test_build_feature_map_rejects_invalid_values(
    features: int, reps: int, message: str
) -> None:
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
    assert 0 <= result.classical.train_accuracy <= 1
    assert 0 <= result.classical.test_accuracy <= 1
    assert 0 <= result.quantum.train_accuracy <= 1
    assert 0 <= result.quantum.test_accuracy <= 1
    assert result.classical.fit_seconds >= 0
    assert result.quantum.fit_seconds >= 0
    assert result.classical.support_vectors > 0
    assert result.quantum.support_vectors > 0
    assert payload["schema_version"] == 1
    assert payload["runtime"]["packages"]["qml-qiskit"] == "1.0.0"
    assert payload["quantum_advantage"] == pytest.approx(
        result.quantum.test_accuracy - result.classical.test_accuracy
    )
