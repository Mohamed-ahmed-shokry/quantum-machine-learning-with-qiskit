"""Tests for the QML benchmark command-line interface."""

import json

import pytest

from qml_qiskit.cli import main


def test_cli_prints_package_version(capsys) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--version"])

    assert capsys.readouterr().out == "qml-qiskit 1.0.0\n"


def test_cli_prints_json_and_writes_result(tmp_path, capsys) -> None:
    output = tmp_path / "nested" / "benchmark.json"

    exit_code = main(
        [
            "--samples",
            "20",
            "--feature-map-reps",
            "1",
            "--seed",
            "9",
            "--json",
            "--output",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert printed == saved
    assert printed["samples"] == 20
    assert printed["seed"] == 9
    assert printed["noise"] == 0.12
    assert printed["test_size"] == 0.25
    assert printed["classical"]["name"] == "Classical RBF SVC"
    assert printed["quantum"]["name"] == "Quantum fidelity QSVC"


def test_cli_prints_readable_report(capsys) -> None:
    exit_code = main(["--samples", "20", "--feature-map-reps", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "QML benchmark | 20 samples" in output
    assert "Dataset configuration: noise 0.12 | test split 0.25" in output
    assert "Classical RBF SVC" in output
    assert "Quantum fidelity QSVC" in output
    assert "Quantum test-score delta:" in output


def test_cli_prints_repeated_study_json(capsys) -> None:
    exit_code = main(
        [
            "--samples",
            "20",
            "--feature-map-reps",
            "1",
            "--seed",
            "5",
            "--repeats",
            "2",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert result["seeds"] == [5, 6]
    assert result["classical"]["runs"] == 2
    assert result["quantum"]["runs"] == 2
    assert len(result["benchmarks"]) == 2


def test_cli_prints_readable_study_report(capsys) -> None:
    exit_code = main(["--samples", "20", "--feature-map-reps", "1", "--repeats", "2"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "QML study | 20 samples | 2 paired runs" in output
    assert "Dataset configuration: noise 0.12 | test split 0.25" in output
    assert "Test mean ± sd" in output
    assert "Paired outcomes (quantum / tie / classical):" in output
    assert "Mean quantum test-score delta:" in output


def test_cli_reports_output_write_failures(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--samples",
                "20",
                "--feature-map-reps",
                "1",
                "--output",
                str(tmp_path),
            ]
        )

    assert "could not write --output" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_rejects_conflicting_artifact_paths(tmp_path, capsys) -> None:
    artifact = tmp_path / "result"

    with pytest.raises(SystemExit, match="2"):
        main(["--output", str(artifact), "--report", str(artifact)])

    assert "--output and --report must use different paths" in capsys.readouterr().err


def test_cli_writes_html_study_report(tmp_path, capsys) -> None:
    report_path = tmp_path / "reports" / "study.html"

    exit_code = main(
        [
            "--samples",
            "20",
            "--feature-map-reps",
            "1",
            "--repeats",
            "2",
            "--report",
            str(report_path),
        ]
    )
    report = report_path.read_text(encoding="utf-8")

    assert exit_code == 0
    assert "QML study" in capsys.readouterr().out
    assert report.startswith("<!doctype html>")
    assert "Paired run audit" in report
    assert "Responsible interpretation" in report


@pytest.mark.parametrize(
    "arguments",
    [
        ["--samples", "7"],
        ["--noise", "-1"],
        ["--noise", "nan"],
        ["--noise", "inf"],
        ["--test-size", "1"],
        ["--test-size", "nan"],
        ["--seed", "-1"],
        ["--samples", "8", "--test-size", "0.1"],
        ["--repeats", "0"],
        ["--feature-map-reps", "0"],
    ],
)
def test_cli_rejects_invalid_arguments(arguments, capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(arguments)

    assert "error:" in capsys.readouterr().err
