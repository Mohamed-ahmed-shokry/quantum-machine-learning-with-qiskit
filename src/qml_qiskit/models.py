"""Classical and quantum-kernel classifiers used by the benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Protocol

from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.svm import SVC

from qml_qiskit.data import MAX_RANDOM_SEED, DatasetSplit
from qml_qiskit.metadata import ARTIFACT_SCHEMA_VERSION, runtime_metadata


class _Classifier(Protocol):
    def fit(self, features: Any, labels: Any) -> Any: ...

    def score(self, features: Any, labels: Any) -> float: ...


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """The measured quality and training cost for one classifier."""

    name: str
    train_accuracy: float
    test_accuracy: float
    fit_seconds: float
    support_vectors: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Comparable metrics for the classical baseline and quantum model."""

    samples: int
    features: int
    seed: int
    feature_map_reps: int
    classical: ModelMetrics
    quantum: ModelMetrics
    noise: float | None = None
    test_size: float | None = None

    @property
    def quantum_advantage(self) -> float:
        """Return the quantum test score minus the classical test score."""

        return self.quantum.test_accuracy - self.classical.test_accuracy

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        payload = asdict(self)
        payload["quantum_advantage"] = self.quantum_advantage
        payload["schema_version"] = ARTIFACT_SCHEMA_VERSION
        payload["runtime"] = runtime_metadata()
        return payload


def build_feature_map(num_features: int, *, reps: int = 2) -> QuantumCircuit:
    """Build the entangling feature map used by the quantum kernel."""

    if num_features < 2:
        raise ValueError("num_features must be at least 2")
    if reps < 1:
        raise ValueError("reps must be at least 1")
    return zz_feature_map(feature_dimension=num_features, reps=reps, entanglement="linear")


def build_quantum_classifier(num_features: int, *, reps: int = 2) -> QSVC:
    """Create an exact statevector quantum-kernel support vector classifier."""

    feature_map = build_feature_map(num_features, reps=reps)
    kernel = FidelityStatevectorKernel(feature_map=feature_map, shots=None, enforce_psd=True)
    return QSVC(quantum_kernel=kernel)


def run_benchmark(
    data: DatasetSplit,
    *,
    seed: int | None = None,
    feature_map_reps: int = 2,
) -> BenchmarkResult:
    """Fit a classical RBF SVC and an exact quantum-kernel SVC on one split."""

    data.validate()
    if seed is None:
        seed = data.seed if data.seed is not None else 42
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_RANDOM_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_RANDOM_SEED}")
    if data.seed is not None and seed != data.seed:
        raise ValueError(f"seed must match the dataset seed ({data.seed})")

    classical_model = SVC(kernel="rbf", random_state=seed)
    quantum_model = build_quantum_classifier(data.num_features, reps=feature_map_reps)

    classical_metrics = _fit_and_measure("Classical RBF SVC", classical_model, data)
    quantum_metrics = _fit_and_measure("Quantum fidelity QSVC", quantum_model, data)

    return BenchmarkResult(
        samples=len(data.train_labels) + len(data.test_labels),
        features=data.num_features,
        seed=seed,
        feature_map_reps=feature_map_reps,
        classical=classical_metrics,
        quantum=quantum_metrics,
        noise=data.noise,
        test_size=data.test_size,
    )


def _fit_and_measure(name: str, model: _Classifier, data: DatasetSplit) -> ModelMetrics:
    started_at = perf_counter()
    model.fit(data.train_features, data.train_labels)
    fit_seconds = perf_counter() - started_at
    support_vectors = int(sum(getattr(model, "n_support_", ())))

    return ModelMetrics(
        name=name,
        train_accuracy=float(model.score(data.train_features, data.train_labels)),
        test_accuracy=float(model.score(data.test_features, data.test_labels)),
        fit_seconds=fit_seconds,
        support_vectors=support_vectors,
    )
