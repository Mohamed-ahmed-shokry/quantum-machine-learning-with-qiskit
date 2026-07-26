# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Dataset noise and test-split provenance in benchmark and study artifacts,
  terminal summaries, and HTML reports.
- Content-derived SHA-256 identifiers shared by JSON, terminal, and HTML
  representations of the same measured run.

### Changed

- Numeric dataset parameters now reject `NaN` and infinity with clear validation
  errors.
- Benchmark runs now reject a model seed that conflicts with a generated
  dataset's recorded seed.
- JSON and HTML outputs are written atomically, and the CLI rejects conflicting
  output paths instead of silently overwriting one artifact with the other.
- Custom `DatasetSplit` instances are validated before model construction,
  producing direct errors for malformed arrays, invalid binary labels, and
  inconsistent provenance.

## 1.0.0 - 2026-07-24

### Added

- Installable `qml-qiskit` Python package with typed public APIs.
- Leakage-safe, deterministic moon-dataset preparation.
- Classical RBF SVC and exact fidelity-statevector QSVC benchmark.
- Repeated paired studies with mean, standard deviation, and win/tie counts.
- Human-readable and JSON CLI with artifact output.
- Schema-versioned artifacts with Python, platform, and dependency metadata.
- Packaged JSON Schema and a public loader for artifact contract validation.
- Responsive, self-contained HTML reports for benchmarks and paired studies.
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
