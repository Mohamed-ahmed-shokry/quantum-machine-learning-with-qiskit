# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 1.0.0 - 2026-07-24

### Added

- Installable `qml-qiskit` Python package with typed public APIs.
- Leakage-safe, deterministic moon-dataset preparation.
- Classical RBF SVC and exact fidelity-statevector QSVC benchmark.
- Repeated paired studies with mean, standard deviation, and win/tie counts.
- Human-readable and JSON CLI with artifact output.
- Schema-versioned artifacts with Python, platform, and dependency metadata.
- Three maintained, executable educational notebooks.
- Behavioral test suite with notebook execution and a 95% coverage gate.
- Python 3.10, 3.12, and 3.14 GitHub Actions matrix.
- Wheel and source-distribution build validation.
- Dependabot, security policy, and credential-handling guidance.

### Changed

- Replaced retired Qiskit APIs with Qiskit 2.x and V2 primitive workflows.
- Moved pre-1.0 notebook experiments into a documented legacy archive.
- Expanded the README into a complete installation, usage, methodology, and
  interpretation guide.

### Security

- Removed a plaintext IBM Quantum token from the current repository tree.
- Added explicit guidance to revoke exposed credentials because Git history,
  forks, and caches may retain them.
