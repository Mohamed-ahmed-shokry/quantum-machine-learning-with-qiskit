"""Reproducible quantum machine-learning experiments built with Qiskit."""

from importlib.metadata import PackageNotFoundError, version

from qml_qiskit.data import DatasetSplit, make_moons_split
from qml_qiskit.models import (
    BenchmarkResult,
    ModelMetrics,
    build_feature_map,
    build_quantum_classifier,
    run_benchmark,
)

try:
    __version__ = version("qml-qiskit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BenchmarkResult",
    "DatasetSplit",
    "ModelMetrics",
    "__version__",
    "build_feature_map",
    "build_quantum_classifier",
    "make_moons_split",
    "run_benchmark",
]
