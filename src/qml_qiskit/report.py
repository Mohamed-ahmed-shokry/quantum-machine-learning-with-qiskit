"""Dependency-free HTML reports for benchmark and study results."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import cast

from qml_qiskit.metadata import runtime_metadata
from qml_qiskit.models import BenchmarkResult, ModelMetrics
from qml_qiskit.study import StudyResult


def render_html_report(
    result: BenchmarkResult | StudyResult,
    *,
    runtime: Mapping[str, object] | None = None,
    artifact_id: str | None = None,
) -> str:
    """Render a self-contained report with optional saved provenance."""

    runtime_info = runtime_metadata() if runtime is None else runtime
    report_artifact_id = result.artifact_id if artifact_id is None else artifact_id
    body = _render_study(result) if isinstance(result, StudyResult) else _render_benchmark(result)
    packages = cast(Mapping[str, str], runtime_info["packages"])
    package_list = " · ".join(
        f"{escape(name)} {escape(version)}" for name, version in packages.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QML Qiskit experiment report</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #080b16;
      --panel: #11172a;
      --panel-2: #171f37;
      --text: #eef2ff;
      --muted: #aab5d1;
      --classical: #39c6b5;
      --quantum: #9b7cff;
      --line: #2a3557;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at 15% 0%, #202557 0, transparent 35rem),
        var(--bg);
      color: var(--text);
      font: 16px/1.55 Inter, ui-sans-serif, system-ui, sans-serif;
    }}
    main {{ width: min(1080px, calc(100% - 2rem)); margin: 0 auto; padding: 3rem 0; }}
    header {{ margin-bottom: 2rem; }}
    .eyebrow {{
      color: #b9a9ff;
      font-size: .78rem;
      font-weight: 800;
      letter-spacing: .16em;
      text-transform: uppercase;
    }}
    h1 {{ margin: .35rem 0 .65rem; font-size: clamp(2rem, 6vw, 4rem); line-height: 1; }}
    h2 {{ margin-top: 0; font-size: 1.15rem; }}
    p {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
    }}
    .card {{
      background: linear-gradient(145deg, var(--panel-2), var(--panel));
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1.25rem;
      box-shadow: 0 18px 50px rgb(0 0 0 / 20%);
    }}
    .metric {{ font-size: 2rem; font-weight: 800; }}
    .label {{ color: var(--muted); font-size: .82rem; text-transform: uppercase; }}
    section {{ margin: 1rem 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: .75rem; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ color: var(--muted); font-size: .78rem; text-transform: uppercase; }}
    .bar-row {{ margin: 1rem 0; }}
    .bar-label {{ display: flex; justify-content: space-between; margin-bottom: .35rem; }}
    .track {{ height: .7rem; overflow: hidden; border-radius: 999px; background: #080c18; }}
    .bar {{ height: 100%; border-radius: inherit; }}
    .classical {{ background: var(--classical); }}
    .quantum {{ background: var(--quantum); }}
    .note {{ border-left: 3px solid var(--quantum); }}
    footer {{ color: var(--muted); font-size: .8rem; margin-top: 2rem; }}
    code {{ color: #d8d0ff; }}
    .artifact {{ overflow-wrap: anywhere; }}
    @media (max-width: 720px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .table-wrap {{ overflow-x: auto; }}
      main {{ padding-top: 1.5rem; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="eyebrow">Quantum machine learning · Qiskit</div>
    <h1>Experiment report</h1>
    <p>Paired classical and quantum-kernel classification on a reproducible nonlinear dataset.</p>
    <p class="artifact">Artifact ID <code>{escape(report_artifact_id)}</code></p>
  </header>
  {body}
  <section class="card note">
    <h2>Responsible interpretation</h2>
    <p>
      This educational statevector simulation is not evidence of practical quantum advantage.
      Compare accuracy, variation, runtime, and classical baselines before drawing conclusions.
    </p>
  </section>
  <footer>
    qml-qiskit {escape(packages["qml-qiskit"])} · Python {escape(str(runtime_info["python"]))} ·
    {escape(str(runtime_info["platform"]))}<br>
    {package_list}
  </footer>
</main>
</body>
</html>
"""


def _render_benchmark(result: BenchmarkResult) -> str:
    delta = result.quantum_advantage
    return f"""
  <section class="grid">
    {_metric_card("Samples", str(result.samples))}
    {_metric_card("Seed", str(result.seed))}
    {_metric_card("Noise", _format_optional(result.noise))}
    {_metric_card("Test split", _format_optional(result.test_size))}
    {_metric_card("Quantum delta", f"{delta:+.3f}")}
  </section>
  <section class="card">
    <h2>Test accuracy</h2>
    {_score_bar(result.classical.name, result.classical.test_accuracy, "classical")}
    {_score_bar(result.quantum.name, result.quantum.test_accuracy, "quantum")}
  </section>
  <section class="card table-wrap">
    <h2>Model metrics</h2>
    <table>
      <thead>
        <tr><th>Model</th><th>Train</th><th>Test</th><th>Fit seconds</th><th>SVs</th></tr>
      </thead>
      <tbody>
        {_model_row(result.classical)}
        {_model_row(result.quantum)}
      </tbody>
    </table>
  </section>
"""


def _render_study(result: StudyResult) -> str:
    rows = "\n".join(
        f"""<tr>
          <td>{benchmark.seed}</td>
          <td>{benchmark.classical.test_accuracy:.3f}</td>
          <td>{benchmark.quantum.test_accuracy:.3f}</td>
          <td>{benchmark.quantum_advantage:+.3f}</td>
        </tr>"""
        for benchmark in result.benchmarks
    )
    return f"""
  <section class="grid">
    {_metric_card("Paired runs", str(len(result.seeds)))}
    {_metric_card("Seed range", f"{result.seeds[0]}-{result.seeds[-1]}")}
    {_metric_card("Noise", _format_optional(result.noise))}
    {_metric_card("Test split", _format_optional(result.test_size))}
    {_metric_card("Mean quantum delta", f"{result.quantum_advantage_mean:+.3f}")}
  </section>
  <section class="card">
    <h2>Mean test accuracy</h2>
    {_score_bar(result.classical.name, result.classical.test_accuracy_mean, "classical")}
    {_score_bar(result.quantum.name, result.quantum.test_accuracy_mean, "quantum")}
    <p>
      Population standard deviation:
      classical {result.classical.test_accuracy_std:.3f} ·
      quantum {result.quantum.test_accuracy_std:.3f}
    </p>
  </section>
  <section class="grid">
    {_metric_card("Quantum wins", str(result.quantum_wins))}
    {_metric_card("Ties", str(result.ties))}
    {_metric_card("Classical wins", str(result.classical_wins))}
    {_metric_card("Sign-test p", f"{result.sign_test_pvalue:.4g}")}
  </section>
  <section class="card note">
    <h2>Paired sign test</h2>
    <p>
      The exact two-sided sign test excludes ties and tests whether either model wins
      more often across the selected seeds. It does not measure effect size or establish
      practical quantum advantage.
    </p>
  </section>
  <section class="card table-wrap">
    <h2>Paired run audit</h2>
    <table>
      <thead>
        <tr><th>Seed</th><th>Classical test</th><th>Quantum test</th><th>Delta</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
"""


def _metric_card(label: str, value: str) -> str:
    return (
        f'<article class="card"><div class="label">{escape(label)}</div>'
        f'<div class="metric">{escape(value)}</div></article>'
    )


def _format_optional(value: float | None) -> str:
    return "custom" if value is None else f"{value:g}"


def _score_bar(label: str, score: float, css_class: str) -> str:
    return f"""
    <div class="bar-row">
      <div class="bar-label"><span>{escape(label)}</span><strong>{score:.3f}</strong></div>
      <div class="track" role="img" aria-label="{escape(label)} score {score:.3f}">
        <div class="bar {css_class}" style="width: {score * 100:.1f}%"></div>
      </div>
    </div>"""


def _model_row(metrics: ModelMetrics) -> str:
    name = escape(metrics.name)
    return (
        f"<tr><td>{name}</td>"
        f"<td>{metrics.train_accuracy:.3f}</td>"
        f"<td>{metrics.test_accuracy:.3f}</td>"
        f"<td>{metrics.fit_seconds:.4f}</td>"
        f"<td>{metrics.support_vectors}</td></tr>"
    )
