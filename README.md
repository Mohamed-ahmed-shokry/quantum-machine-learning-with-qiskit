# Quantum Machine Learning with Qiskit

[![CI](https://github.com/Mohamed-ahmed-shokry/quantum-machine-learning-with-qiskit/actions/workflows/ci.yml/badge.svg)](https://github.com/Mohamed-ahmed-shokry/quantum-machine-learning-with-qiskit/actions/workflows/ci.yml)
[![Python 3.10–3.14](https://img.shields.io/badge/python-3.10%E2%80%933.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Qiskit 2.x](https://img.shields.io/badge/Qiskit-2.x-6929C4)](https://www.ibm.com/quantum/qiskit)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A reproducible, test-driven introduction to quantum machine learning with
modern Qiskit primitives. The project builds a nonlinear classification
benchmark, inspects its quantum fidelity kernel, and compares a Qiskit QSVC
against a classical support vector machine on exactly the same data.

No IBM Quantum account, API token, or GPU is required for the core workflow.

## What is included

| Component | Purpose |
| --- | --- |
| `qml_qiskit.artifacts` | Validated artifact loading and report regeneration |
| `qml_qiskit.data` | Leakage-safe, seeded dataset generation and angle scaling |
| `qml_qiskit.models` | Classical RBF SVC, fidelity-statevector kernel, and QSVC |
| `qml_qiskit.study` | Repeated paired runs with aggregate statistics |
| `qml-qiskit` | Console, JSON, and self-contained HTML reporting |
| `notebooks/01_quantum_circuits.ipynb` | Bell states and V2 `StatevectorSampler` |
| `notebooks/02_quantum_kernel_benchmark.ipynb` | End-to-end quantum-kernel lab |
| `notebooks/03_repeated_seed_study.ipynb` | Paired multi-seed experiment design |
| `tests/` | Fast behavioral tests with a 95% coverage gate |
| `.github/workflows/ci.yml` | CI on Python 3.10, 3.12, and 3.14 |

The original pre-1.0 experiments remain under `notebooks/legacy/` for
historical context, but they are deliberately excluded from the maintained
workflow.

## Quick start

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Mohamed-ahmed-shokry/quantum-machine-learning-with-qiskit.git
cd quantum-machine-learning-with-qiskit
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and run the benchmark:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
qml-qiskit --version
qml-qiskit
```

A default run trains both models on a deterministic 60-sample moon dataset and
prints comparable train accuracy, test accuracy, fit time, and support-vector
count:

```text
QML benchmark | 60 samples | 2 features | seed 42
-------------------------------------------------
Model                        Train     Test    Fit (s)    SVs
Classical RBF SVC            0.978    0.867      ...       21
Quantum fidelity QSVC        0.800    0.867      ...       35
-------------------------------------------------
Quantum test-score delta: +0.000
```

Timings vary by machine. The seeded data and exact statevector kernel make the
scores reproducible for a fixed dependency set.

## Explore experiments

Change the sample count, noise, split, seed, or circuit depth:

```bash
qml-qiskit --samples 80 --noise 0.08 --test-size 0.3 \
  --seed 7 --feature-map-reps 3
```

One split can be misleading. Run a paired study over consecutive seeds to
report mean accuracy, population standard deviation, win/tie counts, and an
exact two-sided paired sign-test p-value:

```bash
qml-qiskit --repeats 10 --samples 60 --seed 42
```

The sign test excludes tied pairs and checks whether either model wins more
often across the selected seeds. It does not measure the size of the accuracy
difference, prove practical relevance, or establish quantum advantage.

Every repeated run is retained in JSON output, so aggregate claims remain
auditable. Serialized results also include the dataset noise and test split,
an artifact schema version, Python version, platform, and core dependency
versions. The packaged
[JSON Schema](src/qml_qiskit/schemas/result-v1.schema.json) validates both
single benchmarks and repeated studies:

```bash
qml-qiskit --repeats 10 --json \
  --output artifacts/ten-seed-study.json \
  --report artifacts/ten-seed-study.html
```

Generating both files in one invocation guarantees that their measured scores
and timings come from the same run. Each output carries the same content-derived
SHA-256 artifact ID, making the pair easy to verify.

Verify a saved JSON artifact without rerunning either model:

```bash
qml-qiskit --verify-artifact artifacts/ten-seed-study.json
```

Verification exits with status 1 when the ID is missing or no longer matches
the measured content, and status 2 when the file cannot be read or parsed.
The ID detects accidental content changes; it is not a signature or proof of
who produced the artifact.

Regenerate an HTML report later without rerunning the experiment:

```bash
qml-qiskit --from-artifact artifacts/ten-seed-study.json \
  --report artifacts/ten-seed-study.html
```

Loading validates the packaged JSON Schema, artifact IDs, and cross-field
semantics before reconstructing the result. Aggregates, paired outcomes,
configuration, and nested runs must agree. The regenerated report retains the
saved runtime provenance rather than describing the machine that renders it.

Produce machine-readable output or save an experiment artifact:

```bash
qml-qiskit --json
qml-qiskit --output artifacts/benchmark.json
```

The same workflow is available as a Python API:

```python
from qml_qiskit import (
    load_artifact_schema,
    load_artifact,
    make_moons_split,
    render_html_report,
    run_benchmark,
    run_study,
    verify_artifact_identifier,
)

data = make_moons_split(samples=60, seed=42)
result = run_benchmark(data, seed=42, feature_map_reps=2)

print(result.quantum.test_accuracy)
print(result.as_dict())
assert verify_artifact_identifier(result.as_dict())

study = run_study(samples=60, base_seed=42, runs=10)
print(study.quantum.test_accuracy_mean)
print(study.quantum_wins, study.ties, study.classical_wins)

schema = load_artifact_schema()
html = render_html_report(study)

loaded = load_artifact("artifacts/ten-seed-study.json")
saved_html = loaded.render_html()
```

`run_benchmark` validates custom `DatasetSplit` instances before model
construction, so malformed shapes, non-finite features, invalid binary labels,
and inconsistent provenance fail with actionable errors.

## Run the notebooks

Install the notebook extras and launch JupyterLab:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Follow the notebooks in order:

1. `01_quantum_circuits.ipynb` introduces circuits, entanglement, measurement,
   and Qiskit's V2 sampler interface.
2. `02_quantum_kernel_benchmark.ipynb` prepares data, builds a `ZZFeatureMap`,
   verifies the kernel matrix, and evaluates both classifiers.
3. `03_repeated_seed_study.ipynb` measures split-to-split variation, audits
   every paired run, and explains responsible interpretation.

The notebooks contain assertions as well as explanations, so incorrect
intermediate results fail visibly.

## How the benchmark works

1. `make_moons` creates a balanced nonlinear binary problem.
2. A stratified train/test split is produced from a fixed random seed.
3. A `MinMaxScaler` is fitted only on the training partition. Values are mapped
   to `[0, π]`, which is suitable for parameterized rotation angles.
4. The classical baseline uses scikit-learn's radial-basis SVC.
5. The quantum model encodes samples with an entangling `ZZFeatureMap`.
6. `FidelityStatevectorKernel` computes
   `K(x, y) = |<φ(x)|φ(y)>|²` exactly.
7. QSVC learns its decision boundary from that precomputed quantum kernel.
8. Study mode repeats the paired comparison over consecutive seeds and reports
   the mean, population standard deviation, and every underlying run.

The exact simulator keeps the example fast and reproducible. It also makes the
kernel easy to inspect: the maintained notebook checks symmetry, unit diagonal,
and positive semidefiniteness.

## Project structure

```text
.
├── .github/workflows/ci.yml
├── notebooks/
│   ├── 01_quantum_circuits.ipynb
│   ├── 02_quantum_kernel_benchmark.ipynb
│   ├── 03_repeated_seed_study.ipynb
│   └── legacy/
├── src/qml_qiskit/
│   ├── artifacts.py
│   ├── cli.py
│   ├── data.py
│   ├── metadata.py
│   ├── models.py
│   ├── report.py
│   ├── schemas/
│   └── study.py
├── tests/
├── pyproject.toml
└── README.md
```

## Development

Install the development dependencies, then run the same checks used by CI:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests notebooks
python -m ruff format --check src tests notebooks
python -m mypy src
python -m pip_audit --skip-editable
python -m pytest --cov --cov-report=term-missing
```

The test suite covers data validation and reproducibility, circuit
construction, quantum and classical model execution, result serialization,
repeated-study aggregation, CLI behavior, and output persistence.
HTML reports are dependency-free, responsive, and tested for safe escaping.

## Real quantum hardware

The core benchmark intentionally uses local exact simulation. To experiment
with IBM Quantum Runtime separately, install the optional dependency:

```bash
python -m pip install -e ".[runtime]"
```

Keep credentials outside source code and notebooks:

```bash
export QISKIT_IBM_TOKEN="your-token"
```

On PowerShell, use
`$env:QISKIT_IBM_TOKEN = "your-token"` for the current process. Never commit a
token. If a credential is committed even once, revoke it; deleting it from the
latest revision does not erase Git history.

## Scope and interpretation

This repository is an educational benchmark, not a claim of quantum advantage.
A classical simulator performs the quantum-state calculations, the dataset is
small, and accuracy varies with the split and feature map. The useful result is
the transparent experiment: both models share the same preprocessing and data,
all configuration is explicit, and the comparison can be repeated or extended.

For larger experiments, track wall-clock cost, circuit depth, sampling noise,
and classical baselines in addition to accuracy.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, quality gates,
commit expectations, and pull-request guidance. Report vulnerabilities
privately using [SECURITY.md](SECURITY.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
