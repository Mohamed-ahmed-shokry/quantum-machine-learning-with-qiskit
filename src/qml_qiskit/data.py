"""Dataset preparation utilities for the quantum-classification examples."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

FeatureArray = NDArray[np.float64]
LabelArray = NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """A deterministic train/test split with features scaled for rotation angles."""

    train_features: FeatureArray
    test_features: FeatureArray
    train_labels: LabelArray
    test_labels: LabelArray

    @property
    def num_features(self) -> int:
        """Return the number of input features."""

        return int(self.train_features.shape[1])


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
    if noise < 0:
        raise ValueError("noise must be non-negative")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
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
    )
