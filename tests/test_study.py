"""Tests for repeated-seed benchmark studies."""

import json
from statistics import fmean, pstdev

import pytest

from qml_qiskit.study import _two_sided_sign_test_pvalue, run_study


def test_run_study_aggregates_paired_benchmarks() -> None:
    result = run_study(
        samples=24,
        noise=0.1,
        test_size=0.25,
        base_seed=7,
        runs=3,
        feature_map_reps=1,
    )
    advantages = [benchmark.quantum_advantage for benchmark in result.benchmarks]

    assert result.samples == 24
    assert result.features == 2
    assert result.noise == 0.1
    assert result.test_size == 0.25
    assert result.seeds == (7, 8, 9)
    assert result.feature_map_reps == 1
    assert len(result.benchmarks) == 3
    assert result.classical.runs == 3
    assert result.quantum.runs == 3
    assert result.quantum_wins + result.ties + result.classical_wins == 3
    assert result.sign_test_pvalue == _two_sided_sign_test_pvalue(
        result.quantum_wins,
        result.classical_wins,
    )
    assert 0 <= result.sign_test_pvalue <= 1
    assert result.quantum_advantage_mean == pytest.approx(fmean(advantages))
    assert result.quantum_advantage_std == pytest.approx(pstdev(advantages))
    assert result.classical.test_accuracy_mean == pytest.approx(
        fmean(benchmark.classical.test_accuracy for benchmark in result.benchmarks)
    )
    assert result.quantum.test_accuracy_mean == pytest.approx(
        fmean(benchmark.quantum.test_accuracy for benchmark in result.benchmarks)
    )
    assert result.classical.fit_seconds_mean >= 0
    assert result.quantum.fit_seconds_mean >= 0
    payload = json.loads(json.dumps(result.as_dict()))
    assert payload["seeds"] == [7, 8, 9]
    assert payload["noise"] == 0.1
    assert payload["test_size"] == 0.25
    assert payload["sign_test_pvalue"] == result.sign_test_pvalue
    assert payload["artifact_id"] == result.artifact_id
    assert len(result.artifact_id) == 64
    assert all("quantum_advantage" in benchmark for benchmark in payload["benchmarks"])
    assert all(benchmark["noise"] == 0.1 for benchmark in payload["benchmarks"])
    assert all(
        payload_benchmark["artifact_id"] == benchmark.artifact_id
        for payload_benchmark, benchmark in zip(
            payload["benchmarks"],
            result.benchmarks,
            strict=True,
        )
    )
    assert payload["schema_version"] == 1
    assert payload["runtime"]["packages"]["qml-qiskit"] == "1.0.0"


@pytest.mark.parametrize(
    ("quantum_wins", "classical_wins", "expected"),
    [
        (0, 0, 1.0),
        (1, 0, 1.0),
        (2, 0, 0.5),
        (2, 2, 1.0),
        (10, 0, 0.001953125),
        (9, 1, 0.021484375),
    ],
)
def test_two_sided_sign_test_is_exact(
    quantum_wins: int,
    classical_wins: int,
    expected: float,
) -> None:
    assert _two_sided_sign_test_pvalue(quantum_wins, classical_wins) == expected


def test_run_study_rejects_single_run() -> None:
    with pytest.raises(ValueError, match="runs must be at least 2"):
        run_study(runs=1)


@pytest.mark.parametrize("base_seed", [-1, 2**32 - 1])
def test_run_study_rejects_seed_ranges_that_cannot_fit_runs(base_seed: int) -> None:
    with pytest.raises(
        ValueError,
        match="base_seed and runs must produce seeds between 0 and 4294967295",
    ):
        run_study(base_seed=base_seed, runs=2)


def test_run_study_propagates_dataset_validation() -> None:
    with pytest.raises(ValueError, match="samples must be at least 8"):
        run_study(samples=7, runs=2)
