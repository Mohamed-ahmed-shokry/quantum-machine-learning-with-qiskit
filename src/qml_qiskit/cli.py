"""Command-line entry point for the reproducible QML benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from qml_qiskit.data import make_moons_split
from qml_qiskit.models import BenchmarkResult, run_benchmark


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
        data = make_moons_split(
            samples=args.samples,
            noise=args.noise,
            test_size=args.test_size,
            seed=args.seed,
        )
    except ValueError as error:
        parser.error(str(error))
    result = run_benchmark(data, seed=args.seed, feature_map_reps=args.feature_map_reps)
    serialized = json.dumps(result.as_dict(), indent=2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{serialized}\n", encoding="utf-8")

    print(serialized if args.as_json else _format_report(result))
    return 0


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.samples < 8:
        parser.error("--samples must be at least 8")
    if args.noise < 0:
        parser.error("--noise must be non-negative")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")
    if args.feature_map_reps < 1:
        parser.error("--feature-map-reps must be at least 1")


def _format_report(result: BenchmarkResult) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
