"""Tests for deterministic dataset preparation."""

import numpy as np
import pytest

from qml_qiskit.data import make_moons_split


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
    assert np.all(first.train_features >= 0)
    assert np.all(first.train_features <= np.pi)
    assert np.all(first.test_features >= 0)
    assert np.all(first.test_features <= np.pi)
    assert set(first.train_labels) == {0, 1}
    assert set(first.test_labels) == {0, 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": 7}, "samples must be at least 8"),
        ({"noise": -0.01}, "noise must be non-negative"),
        ({"test_size": 0}, "test_size must be between 0 and 1"),
        ({"test_size": 1}, "test_size must be between 0 and 1"),
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
