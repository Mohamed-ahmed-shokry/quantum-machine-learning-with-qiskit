"""Tests for the QML benchmark command-line interface."""

import json

import pytest

from qml_qiskit.cli import main


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
    assert printed["classical"]["name"] == "Classical RBF SVC"
    assert printed["quantum"]["name"] == "Quantum fidelity QSVC"


def test_cli_prints_readable_report(capsys) -> None:
    exit_code = main(["--samples", "20", "--feature-map-reps", "1"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "QML benchmark | 20 samples" in output
    assert "Classical RBF SVC" in output
    assert "Quantum fidelity QSVC" in output
    assert "Quantum test-score delta:" in output


@pytest.mark.parametrize(
    "arguments",
    [
        ["--samples", "7"],
        ["--noise", "-1"],
        ["--test-size", "1"],
        ["--samples", "8", "--test-size", "0.1"],
        ["--feature-map-reps", "0"],
    ],
)
def test_cli_rejects_invalid_arguments(arguments, capsys) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(arguments)

    assert "error:" in capsys.readouterr().err
