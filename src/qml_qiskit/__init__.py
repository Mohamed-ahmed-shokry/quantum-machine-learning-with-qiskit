"""Reproducible quantum machine-learning experiments built with Qiskit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qml-qiskit")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
