"""Command-line entry point for the reproducible QML benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qml_qiskit.data import make_moons_split
from qml_qiskit.models import BenchmarkResult, run_benchmark
from qml_qiskit.study import StudyResult, run_study


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(
        prog="qml-qiskit",
        description="Compare a classical SVC with a Qiskit fidelity-kernel QSVC.",
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

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
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(f"{serialized}\n", encoding="utf-8")
        except OSError as error:
            parser.error(f"could not write --output {args.output}: {error}")

    print(serialized if args.as_json else _format_report(result))
    return 0


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples < 8:
        parser.error("--samples must be at least 8")
    if args.noise < 0:
        parser.error("--noise must be non-negative")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.feature_map_reps < 1:
        parser.error("--feature-map-reps must be at least 1")


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
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
