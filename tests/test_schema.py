"""Contract tests for versioned experiment artifacts."""

from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator, ValidationError

from qml_qiskit import load_artifact_schema, make_moons_split, run_benchmark, run_study


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = load_artifact_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_benchmark_artifact_matches_schema(validator: Draft202012Validator) -> None:
    data = make_moons_split(samples=20, seed=3)
    artifact = run_benchmark(data, seed=3, feature_map_reps=1).as_dict()

    validator.validate(artifact)


def test_study_artifact_matches_schema(validator: Draft202012Validator) -> None:
    artifact = run_study(
        samples=20,
        base_seed=3,
        runs=2,
        feature_map_reps=1,
    ).as_dict()

    validator.validate(artifact)


def test_schema_rejects_out_of_range_accuracy(validator: Draft202012Validator) -> None:
    data = make_moons_split(samples=20, seed=3)
    artifact = deepcopy(run_benchmark(data, seed=3, feature_map_reps=1).as_dict())
    artifact["quantum"]["test_accuracy"] = 1.5

    with pytest.raises(ValidationError):
        validator.validate(artifact)


def test_schema_remains_compatible_with_existing_v1_artifacts(
    validator: Draft202012Validator,
) -> None:
    artifact = run_study(
        samples=20,
        base_seed=3,
        runs=2,
        feature_map_reps=1,
    ).as_dict()
    artifact.pop("noise")
    artifact.pop("test_size")
    for benchmark in artifact["benchmarks"]:
        benchmark.pop("noise")
        benchmark.pop("test_size")

    validator.validate(artifact)
