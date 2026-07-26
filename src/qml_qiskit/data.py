"""Dataset preparation utilities for the quantum-classification examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FeatureArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]
MAX_RANDOM_SEED = 2**32 - 1


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A deterministic train/test split with features scaled for rotation angles."""

    train_features: FeatureArray
    test_features: FeatureArray
    train_labels: LabelArray
    test_labels: LabelArray
    noise: float | None = None
    test_size: float | None = None
    seed: int | None = None

    @property
    def num_features(self) -> int:
        """Return the number of input features."""

        if self.train_features.ndim != 2:
            raise ValueError("feature arrays must be two-dimensional")
        return int(self.train_features.shape[1])

    def validate(self) -> None:
        """Validate the split before it is passed to either classifier."""

        feature_arrays = (self.train_features, self.test_features)
        label_arrays = (self.train_labels, self.test_labels)
        if not all(isinstance(array, np.ndarray) for array in (*feature_arrays, *label_arrays)):
            raise ValueError("features and labels must be NumPy arrays")
        if any(array.ndim != 2 for array in feature_arrays):
            raise ValueError("feature arrays must be two-dimensional")
        if any(array.ndim != 1 for array in label_arrays):
            raise ValueError("label arrays must be one-dimensional")
        if any(array.shape[0] < 2 for array in feature_arrays):
            raise ValueError("each split must contain at least two samples")
        samples = sum(array.shape[0] for array in feature_arrays)
        if samples < 8:
            raise ValueError("dataset must contain at least 8 samples")
        if self.train_features.shape[1] != self.test_features.shape[1]:
            raise ValueError("feature arrays must have the same number of columns")
        if self.num_features < 2:
            raise ValueError("dataset must contain at least two features")
        if any(
            not (np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer))
            for array in feature_arrays
        ):
            raise ValueError("feature arrays must use real numeric values")
        if not all(np.isfinite(array).all() for array in feature_arrays):
            raise ValueError("feature arrays must contain only finite values")
        if any(not np.issubdtype(array.dtype, np.integer) for array in label_arrays):
            raise ValueError("label arrays must use integer values")
        if any(
            features.shape[0] != labels.shape[0]
            for features, labels in zip(feature_arrays, label_arrays, strict=True)
        ):
            raise ValueError("feature and label counts must match in each split")

        training_classes = np.unique(self.train_labels)
        if len(training_classes) != 2:
            raise ValueError("training labels must contain exactly two classes")
        if not np.isin(np.unique(self.test_labels), training_classes).all():
            raise ValueError("test labels must not contain classes absent from training labels")
        if self.noise is not None and (not _is_finite_real(self.noise) or self.noise < 0):
            raise ValueError("noise must be a finite non-negative number")
        if self.test_size is not None and (
            not _is_finite_real(self.test_size) or not 0 < self.test_size < 1
        ):
            raise ValueError("test_size must be between 0 and 1")
        if self.test_size is not None and self.test_features.shape[0] != ceil(
            samples * self.test_size
        ):
            raise ValueError("test_size does not match the number of test samples")
        if self.seed is not None and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= MAX_RANDOM_SEED
        ):
            raise ValueError(f"seed must be between 0 and {MAX_RANDOM_SEED}")


def _is_finite_real(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def make_moons_split(
    *,
    samples: int = 60,
    noise: float = 0.12,
    test_size: float = 0.25,
    seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible nonlinear binary-classification problem.

    A scaler is fitted on the training partition only to avoid leaking information
    from the test partition. Features are mapped to ``[0, π]``, a natural range for
    the rotation angles used by Qiskit feature maps.
    """

    if samples < 8:
        raise ValueError("samples must be at least 8")
    if not isfinite(noise) or noise < 0:
        raise ValueError("noise must be a finite non-negative number")
    if not isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 <= seed <= MAX_RANDOM_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_RANDOM_SEED}")
    test_samples = ceil(samples * test_size)
    if test_samples < 2 or samples - test_samples < 2:
        raise ValueError("test_size must leave at least two samples in each split")

    features, labels = make_moons(n_samples=samples, noise=noise, random_state=seed)
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaled_train = np.clip(scaler.fit_transform(train_features), 0, np.pi)
    scaled_test = np.clip(scaler.transform(test_features), 0, np.pi)

    return DatasetSplit(
        train_features=np.asarray(scaled_train, dtype=np.float64),
        test_features=np.asarray(scaled_test, dtype=np.float64),
        train_labels=np.asarray(train_labels, dtype=np.int_),
        test_labels=np.asarray(test_labels, dtype=np.int_),
        noise=noise,
        test_size=test_size,
        seed=seed,
    )
