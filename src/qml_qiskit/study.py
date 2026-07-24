"""Repeated-seed studies for more rigorous model comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from statistics import fmean, pstdev
from typing import Any

from qml_qiskit.data import MAX_RANDOM_SEED, make_moons_split
from qml_qiskit.metadata import ARTIFACT_SCHEMA_VERSION, runtime_metadata
from qml_qiskit.models import BenchmarkResult, ModelMetrics, run_benchmark


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate measurements for one model across repeated data splits."""

    name: str
    runs: int
    train_accuracy_mean: float
    train_accuracy_std: float
    test_accuracy_mean: float
    test_accuracy_std: float
    fit_seconds_mean: float
    fit_seconds_std: float
    support_vectors_mean: float
    support_vectors_std: float


@dataclass(frozen=True, slots=True)
class StudyResult:
    """Paired benchmark results and their aggregate statistics."""

    samples: int
    features: int
    seeds: tuple[int, ...]
    feature_map_reps: int
    classical: MetricSummary
    quantum: MetricSummary
    quantum_advantage_mean: float
    quantum_advantage_std: float
    quantum_wins: int
    ties: int
    classical_wins: int
    benchmarks: tuple[BenchmarkResult, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return the complete study as a JSON-serializable mapping."""

        payload = asdict(self)
        payload["benchmarks"] = [
            {
                **asdict(benchmark),
                "quantum_advantage": benchmark.quantum_advantage,
            }
            for benchmark in self.benchmarks
        ]
        payload["schema_version"] = ARTIFACT_SCHEMA_VERSION
        payload["runtime"] = runtime_metadata()
        return payload


def run_study(
    *,
    samples: int = 60,
    noise: float = 0.12,
    test_size: float = 0.25,
    base_seed: int = 42,
    runs: int = 5,
    feature_map_reps: int = 2,
) -> StudyResult:
    """Run paired classical and quantum benchmarks over consecutive seeds."""

    if runs < 2:
        raise ValueError("runs must be at least 2")
    if base_seed < 0 or base_seed + runs - 1 > MAX_RANDOM_SEED:
        raise ValueError(
            f"base_seed and runs must produce seeds between 0 and {MAX_RANDOM_SEED}"
        )

    seeds = tuple(range(base_seed, base_seed + runs))
    benchmarks = tuple(
        run_benchmark(
            make_moons_split(
                samples=samples,
                noise=noise,
                test_size=test_size,
                seed=seed,
            ),
            seed=seed,
            feature_map_reps=feature_map_reps,
        )
        for seed in seeds
    )
    advantages = tuple(result.quantum_advantage for result in benchmarks)
    quantum_wins = sum(
        advantage > 0 and not isclose(advantage, 0, abs_tol=1e-12)
        for advantage in advantages
    )
    classical_wins = sum(
        advantage < 0 and not isclose(advantage, 0, abs_tol=1e-12)
        for advantage in advantages
    )
    ties = runs - quantum_wins - classical_wins
    advantage_mean = fmean(advantages)
    if isclose(advantage_mean, 0, abs_tol=1e-12):
        advantage_mean = 0.0

    return StudyResult(
        samples=samples,
        features=benchmarks[0].features,
        seeds=seeds,
        feature_map_reps=feature_map_reps,
        classical=_summarize(tuple(result.classical for result in benchmarks)),
        quantum=_summarize(tuple(result.quantum for result in benchmarks)),
        quantum_advantage_mean=advantage_mean,
        quantum_advantage_std=pstdev(advantages),
        quantum_wins=quantum_wins,
        ties=ties,
        classical_wins=classical_wins,
        benchmarks=benchmarks,
    )


def _summarize(metrics: tuple[ModelMetrics, ...]) -> MetricSummary:
    return MetricSummary(
        name=metrics[0].name,
        runs=len(metrics),
        train_accuracy_mean=fmean(metric.train_accuracy for metric in metrics),
        train_accuracy_std=pstdev(metric.train_accuracy for metric in metrics),
        test_accuracy_mean=fmean(metric.test_accuracy for metric in metrics),
        test_accuracy_std=pstdev(metric.test_accuracy for metric in metrics),
        fit_seconds_mean=fmean(metric.fit_seconds for metric in metrics),
        fit_seconds_std=pstdev(metric.fit_seconds for metric in metrics),
        support_vectors_mean=fmean(metric.support_vectors for metric in metrics),
        support_vectors_std=pstdev(metric.support_vectors for metric in metrics),
    )
