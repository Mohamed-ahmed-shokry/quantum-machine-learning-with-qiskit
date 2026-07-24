# Contributing

Thank you for improving Quantum Machine Learning with Qiskit. Contributions
should keep the project reproducible, approachable, and technically honest.

## Set up a development environment

```bash
git clone https://github.com/Mohamed-ahmed-shokry/quantum-machine-learning-with-qiskit.git
cd quantum-machine-learning-with-qiskit
python -m venv .venv
python -m pip install -e ".[dev,notebooks]"
```

Activate `.venv` using the command appropriate for your shell before running
the checks below.

## Make a focused change

1. Create a branch from the latest `main`.
2. Keep each commit limited to one logical improvement.
3. Add or update tests for behavior changes.
4. Update the README, notebooks, and changelog when user-facing behavior
   changes.
5. Do not commit generated artifacts, notebook checkpoints, credentials, or
   local environment files.

Prefer the modern Qiskit V2 primitive interfaces and function-based circuit
constructors. Avoid reintroducing retired APIs preserved in
`notebooks/legacy/`.

## Run the quality gates

```bash
python -m ruff check src tests notebooks
python -m ruff format --check src tests notebooks
python -m mypy src
python -m pytest --cov --cov-report=term-missing
python -m build
python -m pip check
```

Maintained notebooks are executed by the test suite. Keep code cells
deterministic, reasonably fast, and free of cloud-account requirements.

## Submit a pull request

Describe:

- the problem and why it matters;
- the implementation and important tradeoffs;
- the verification commands you ran; and
- any effect on public APIs or serialized artifact schemas.

CI must pass on every supported Python version. Reviewers may request smaller
commits when unrelated changes are combined.

## Report security issues privately

Follow [SECURITY.md](SECURITY.md). Never paste a credential or exploitable
detail into a public issue.
