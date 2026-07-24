"""Reproducible quantum machine-learning experiments built with Qiskit."""

from importlib.metadata import PackageNotFoundError, version

from qml_qiskit.data import DatasetSplit, make_moons_split

try:
    __version__ = version("qml-qiskit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["DatasetSplit", "__version__", "make_moons_split"]
