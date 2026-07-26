"""Command-line entry point for the reproducible QML benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from contextlib import suppress
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from qml_qiskit import __version__
from qml_qiskit.data import make_moons_split
from qml_qiskit.metadata import verify_artifact_identifier
from qml_qiskit.models import BenchmarkResult, run_benchmark
from qml_qiskit.report import render_html_report
from qml_qiskit.study import StudyResult, run_study


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="qml-qiskit",
        description="Compare a classical SVC with a Qiskit fidelity-kernel QSVC.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--samples", type=int, default=60, help="total dataset size (default: 60)")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.12,
        help="moon dataset noise (default: 0.12)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="fraction reserved for testing (default: 0.25)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="number of consecutive-seed paired runs (default: 1)",
    )
    parser.add_argument(
        "--feature-map-reps",
        type=int,
        default=2,
        help="ZZ feature-map repetitions (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print machine-readable JSON instead of a table",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also save the complete benchmark result as JSON",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="save a self-contained HTML experiment report",
    )
    parser.add_argument(
        "--verify-artifact",
        type=Path,
        metavar="PATH",
        help="verify a saved artifact ID without running a benchmark",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.verify_artifact is not None:
        return _verify_artifact(parser, args.verify_artifact)
    result: BenchmarkResult | StudyResult

    try:
        if args.repeats == 1:
            data = make_moons_split(
                samples=args.samples,
                noise=args.noise,
                test_size=args.test_size,
                seed=args.seed,
            )
            result = run_benchmark(
                data,
                seed=args.seed,
                feature_map_reps=args.feature_map_reps,
            )
        else:
            result = run_study(
                samples=args.samples,
                noise=args.noise,
                test_size=args.test_size,
                base_seed=args.seed,
                runs=args.repeats,
                feature_map_reps=args.feature_map_reps,
            )
    except ValueError as error:
        parser.error(str(error))
    serialized = json.dumps(result.as_dict(), indent=2)

    if args.output is not None:
        _write_artifact(parser, args.output, f"{serialized}\n", "--output")
    if args.report is not None:
        _write_artifact(parser, args.report, render_html_report(result), "--report")

    print(serialized if args.as_json else _format_report(result))
    return 0


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.verify_artifact is not None:
        if any(
            (
                args.samples != 60,
                args.noise != 0.12,
                args.test_size != 0.25,
                args.seed != 42,
                args.repeats != 1,
                args.feature_map_reps != 2,
                args.as_json,
                args.output is not None,
                args.report is not None,
            )
        ):
            parser.error("--verify-artifact cannot be combined with benchmark or output options")
        return

    if args.samples < 8:
        parser.error("--samples must be at least 8")
    if not isfinite(args.noise) or args.noise < 0:
        parser.error("--noise must be a finite non-negative number")
    if not isfinite(args.test_size) or not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.feature_map_reps < 1:
        parser.error("--feature-map-reps must be at least 1")
    if (
        args.output is not None
        and args.report is not None
        and args.output.resolve() == args.report.resolve()
    ):
        parser.error("--output and --report must use different paths")


def _verify_artifact(parser: argparse.ArgumentParser, path: Path) -> int:
    try:
        serialized = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(f"could not read --verify-artifact {path}: {error}")
    try:
        loaded = json.loads(serialized)
    except json.JSONDecodeError as error:
        parser.error(f"invalid JSON in --verify-artifact {path}: {error}")
    if not isinstance(loaded, dict):
        parser.error(f"--verify-artifact {path} must contain a JSON object")

    payload = cast(dict[str, object], loaded)
    claimed_identifier = payload.get("artifact_id")
    if not isinstance(claimed_identifier, str):
        print(f"{path}: missing artifact_id", file=sys.stderr)
        return 1
    if not verify_artifact_identifier(payload):
        print(f"{path}: artifact_id does not match measured content", file=sys.stderr)
        return 1

    print(f"{path}: verified artifact {claimed_identifier}")
    return 0


def _format_report(result: BenchmarkResult | StudyResult) -> str:
    if isinstance(result, StudyResult):
        return _format_study_report(result)
    return _format_benchmark_report(result)


def _format_benchmark_report(result: BenchmarkResult) -> str:
    header = (
        f"QML benchmark | {result.samples} samples | {result.features} features | "
        f"seed {result.seed}"
    )
    separator = "-" * len(header)
    rows = [
        header,
        f"Artifact ID: {result.artifact_id}",
        _format_dataset_config(result.noise, result.test_size),
        separator,
        f"{'Model':<25} {'Train':>8} {'Test':>8} {'Fit (s)':>10} {'SVs':>6}",
        f"{result.classical.name:<25} "
        f"{result.classical.train_accuracy:>8.3f} "
        f"{result.classical.test_accuracy:>8.3f} "
        f"{result.classical.fit_seconds:>10.4f} "
        f"{result.classical.support_vectors:>6}",
        f"{result.quantum.name:<25} "
        f"{result.quantum.train_accuracy:>8.3f} "
        f"{result.quantum.test_accuracy:>8.3f} "
        f"{result.quantum.fit_seconds:>10.4f} "
        f"{result.quantum.support_vectors:>6}",
        separator,
        f"Quantum test-score delta: {result.quantum_advantage:+.3f}",
    ]
    return "\n".join(rows)


def _format_study_report(result: StudyResult) -> str:
    header = (
        f"QML study | {result.samples} samples | {len(result.seeds)} paired runs | "
        f"seeds {result.seeds[0]}-{result.seeds[-1]}"
    )
    separator = "-" * len(header)
    rows = [
        header,
        f"Artifact ID: {result.artifact_id}",
        _format_dataset_config(result.noise, result.test_size),
        separator,
        f"{'Model':<25} {'Train mean':>10} {'Test mean ± sd':>16} {'Fit mean':>10} {'SV mean':>9}",
        f"{result.classical.name:<25} "
        f"{result.classical.train_accuracy_mean:>10.3f} "
        f"{result.classical.test_accuracy_mean:>8.3f} ± "
        f"{result.classical.test_accuracy_std:<5.3f} "
        f"{result.classical.fit_seconds_mean:>10.4f} "
        f"{result.classical.support_vectors_mean:>9.1f}",
        f"{result.quantum.name:<25} "
        f"{result.quantum.train_accuracy_mean:>10.3f} "
        f"{result.quantum.test_accuracy_mean:>8.3f} ± "
        f"{result.quantum.test_accuracy_std:<5.3f} "
        f"{result.quantum.fit_seconds_mean:>10.4f} "
        f"{result.quantum.support_vectors_mean:>9.1f}",
        separator,
        (
            "Paired outcomes (quantum / tie / classical): "
            f"{result.quantum_wins} / {result.ties} / {result.classical_wins}"
        ),
        (
            "Mean quantum test-score delta: "
            f"{result.quantum_advantage_mean:+.3f} ± {result.quantum_advantage_std:.3f}"
        ),
        f"Exact paired sign-test p-value: {result.sign_test_pvalue:.4g}",
    ]
    return "\n".join(rows)


def _format_dataset_config(noise: float | None, test_size: float | None) -> str:
    if noise is None or test_size is None:
        return "Dataset configuration: custom split"
    return f"Dataset configuration: noise {noise:g} | test split {test_size:g}"


def _write_artifact(
    parser: argparse.ArgumentParser,
    path: Path,
    content: str,
    option: str,
) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
    except OSError as error:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
        parser.error(f"could not write {option} {path}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
