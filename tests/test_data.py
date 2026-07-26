"""Tests for deterministic dataset preparation."""

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from qml_qiskit.data import DatasetSplit, make_moons_split


def test_make_moons_split_is_reproducible_and_scaled() -> None:
    first = make_moons_split(samples=40, test_size=0.25, seed=7)
    second = make_moons_split(samples=40, test_size=0.25, seed=7)

    np.testing.assert_array_equal(first.train_features, second.train_features)
    np.testing.assert_array_equal(first.test_features, second.test_features)
    np.testing.assert_array_equal(first.train_labels, second.train_labels)
    np.testing.assert_array_equal(first.test_labels, second.test_labels)

    assert first.train_features.shape == (30, 2)
    assert first.test_features.shape == (10, 2)
    assert first.num_features == 2
    assert first.noise == 0.12
    assert first.test_size == 0.25
    assert first.seed == 7
    assert np.all(first.train_features >= 0)
    assert np.all(first.train_features <= np.pi)
    assert np.all(first.test_features >= 0)
    assert np.all(first.test_features <= np.pi)
    assert set(first.train_labels) == {0, 1}
    assert set(first.test_labels) == {0, 1}
    first.validate()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 7}, "samples must be at least 8"),
        ({"noise": -0.01}, "noise must be a finite non-negative number"),
        ({"noise": float("nan")}, "noise must be a finite non-negative number"),
        ({"noise": float("inf")}, "noise must be a finite non-negative number"),
        ({"test_size": 0}, "test_size must be between 0 and 1"),
        ({"test_size": 1}, "test_size must be between 0 and 1"),
        ({"test_size": float("nan")}, "test_size must be between 0 and 1"),
        ({"seed": -1}, "seed must be between 0 and 4294967295"),
        ({"seed": 2**32}, "seed must be between 0 and 4294967295"),
        (
            {"samples": 8, "test_size": 0.1},
            "test_size must leave at least two samples in each split",
        ),
    ],
)
def test_make_moons_split_rejects_invalid_values(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make_moons_split(**kwargs)


@pytest.mark.parametrize(
    ("make_invalid", "message"),
    [
        (
            lambda data: replace(data, train_features=np.zeros(30)),
            "feature arrays must be two-dimensional",
        ),
        (
            lambda data: replace(data, train_features=[[0.0, 1.0]] * 15),
            "features and labels must be NumPy arrays",
        ),
        (
            lambda data: replace(
                data,
                test_features=np.zeros((1, 2)),
                test_labels=np.zeros(1, dtype=int),
            ),
            "each split must contain at least two samples",
        ),
        (
            lambda data: replace(data, test_features=np.zeros((5, 3))),
            "feature arrays must have the same number of columns",
        ),
        (
            lambda data: replace(
                data,
                train_features=np.zeros((15, 1)),
                test_features=np.zeros((5, 1)),
            ),
            "dataset must contain at least two features",
        ),
        (
            lambda data: replace(data, train_features=np.full((15, 2), "invalid")),
            "feature arrays must use real numeric values",
        ),
        (
            lambda data: replace(data, test_features=np.full((5, 2), np.nan)),
            "feature arrays must contain only finite values",
        ),
        (
            lambda data: replace(data, train_labels=np.zeros((15, 1), dtype=int)),
            "label arrays must be one-dimensional",
        ),
        (
            lambda data: replace(data, train_labels=np.zeros(15)),
            "label arrays must use integer values",
        ),
        (
            lambda data: replace(data, train_labels=np.zeros(14, dtype=int)),
            "feature and label counts must match in each split",
        ),
        (
            lambda data: replace(data, train_labels=np.zeros(15, dtype=int)),
            "training labels must contain exactly two classes",
        ),
        (
            lambda data: replace(data, test_labels=np.full(5, 2, dtype=int)),
            "test labels must not contain classes absent from training labels",
        ),
        (
            lambda data: replace(data, noise=float("nan")),
            "noise must be a finite non-negative number",
        ),
        (
            lambda data: replace(data, test_size=1),
            "test_size must be between 0 and 1",
        ),
        (
            lambda data: replace(data, test_size=0.3),
            "test_size does not match the number of test samples",
        ),
        (
            lambda data: replace(data, seed=-1),
            "seed must be between 0 and 4294967295",
        ),
    ],
)
def test_dataset_split_rejects_invalid_contracts(
    make_invalid: Callable[[DatasetSplit], DatasetSplit],
    message: str,
) -> None:
    data = make_invalid(make_moons_split(samples=20, seed=7))

    with pytest.raises(ValueError, match=message):
        data.validate()


def test_dataset_split_requires_minimum_total_samples() -> None:
    data = DatasetSplit(
        train_features=np.zeros((4, 2)),
        test_features=np.zeros((2, 2)),
        train_labels=np.array([0, 1, 0, 1]),
        test_labels=np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="dataset must contain at least 8 samples"):
        data.validate()


def test_num_features_rejects_one_dimensional_training_features() -> None:
    data = replace(make_moons_split(samples=20), train_features=np.zeros(15))

    with pytest.raises(ValueError, match="feature arrays must be two-dimensional"):
        _ = data.num_features
