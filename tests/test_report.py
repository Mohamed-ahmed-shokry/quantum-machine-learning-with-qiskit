"""Tests for self-contained HTML experiment reports."""

from qml_qiskit import make_moons_split, render_html_report, run_benchmark, run_study
from qml_qiskit.models import BenchmarkResult, ModelMetrics


def test_benchmark_report_contains_metrics_and_provenance() -> None:
    result = run_benchmark(make_moons_split(samples=20, seed=2), seed=2, feature_map_reps=1)
    report = render_html_report(result)

    assert report.startswith("<!doctype html>")
    assert "<h2>Model metrics</h2>" in report
    assert "Classical RBF SVC" in report
    assert "Quantum fidelity QSVC" in report
    assert "Noise" in report
    assert "0.12" in report
    assert "Test split" in report
    assert "0.25" in report
    assert result.artifact_id in report
    assert 'role="img"' in report
    assert "qml-qiskit 1.0.0" in report
    assert "Responsible interpretation" in report


def test_study_report_contains_paired_run_audit() -> None:
    result = run_study(samples=20, base_seed=2, runs=2, feature_map_reps=1)
    report = render_html_report(result)

    assert "Paired runs" in report
    assert "Seed range" in report
    assert "2-3" in report
    assert "Noise" in report
    assert "Test split" in report
    assert result.artifact_id in report
    assert "<h2>Paired run audit</h2>" in report
    assert "Quantum wins" in report
    assert "Sign-test p" in report
    assert "Paired sign test" in report
    assert "Population standard deviation:" in report


def test_report_escapes_model_names() -> None:
    metrics = ModelMetrics(
        name="<script>alert('unsafe')</script>",
        train_accuracy=0.5,
        test_accuracy=0.5,
        fit_seconds=0.1,
        support_vectors=2,
    )
    result = BenchmarkResult(
        samples=8,
        features=2,
        seed=1,
        feature_map_reps=1,
        classical=metrics,
        quantum=metrics,
    )

    report = render_html_report(result)

    assert "<script>alert" not in report
    assert "&lt;script&gt;alert" in report
