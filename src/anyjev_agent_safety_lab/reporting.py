# ruff: noqa: E501, RUF001
"""Markdown, JSON, and self-contained HTML reporting."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

from .metrics import compute_metrics


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value * 100:.1f}%" if percent else f"{value:.4f}"
    return str(value)


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(
    decisions_path: Path, output_dir: Path, site_path: Path | None = None
) -> dict[str, Path]:
    """Generate all report formats from one decisions manifest."""
    payload = json.loads(decisions_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(payload)
    verified = payload["benchmark"]["backend"] != "fake"
    report = {
        "schema_version": 1,
        "source": decisions_path.name,
        "source_sha256": _source_hash(decisions_path),
        "verified_real_model_run": verified,
        "benchmark": payload["benchmark"],
        "dataset": payload["dataset"],
        "environment": payload["environment"],
        "metrics": metrics,
        "limitations": [
            "The dataset contains 80 synthetic cases and is exploratory.",
            "Confidence values are not a guarantee of real-world safety.",
            "Latency varies by hardware, software versions, and model cache state.",
            "The simulator does not reproduce all consequences of real external tools.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "report.json"
    report_markdown = output_dir / "report.md"
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_markdown.write_text(_markdown(report), encoding="utf-8")
    site = site_path or output_dir / "index.html"
    site.parent.mkdir(parents=True, exist_ok=True)
    site.write_text(_html(report), encoding="utf-8")
    return {"json": report_json, "markdown": report_markdown, "site": site}


def verify_report(decisions_path: Path, report_path: Path) -> tuple[bool, str]:
    """Verify that a report points to the exact decisions file supplied."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual = _source_hash(decisions_path)
    expected = report.get("source_sha256")
    if actual != expected:
        return False, f"report source mismatch: expected {expected}, got {actual}"
    if report.get("metrics") != compute_metrics(
        json.loads(decisions_path.read_text(encoding="utf-8"))
    ):
        return False, "report metrics do not match recomputed metrics"
    return True, actual


def _markdown(report: dict[str, Any]) -> str:
    status = (
        "Verified local-LLM run"
        if report["verified_real_model_run"]
        else "Results pending — deterministic CI fixture only"
    )
    lines = [
        "# AnyJev Decision Lab: Agent-Safety Case Study",
        "",
        f"**Status:** {status}",
        "",
        f"Dataset: {report['dataset']['cases']} synthetic cases  ",
        f"Model: `{report['benchmark']['model']}`  ",
        f"Revision: `{report['benchmark']['model_revision']}`  ",
        f"Backend: `{report['benchmark']['backend']}`",
        "",
        "## Results",
        "",
        "| Strategy | Accuracy | Risky allow | Safe block | Review | Order flips | Median latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in report["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format(values["accuracy"], percent=True),
                    _format(values["policy"]["risky_allow_rate"], percent=True),
                    _format(values["policy"]["safe_block_rate"], percent=True),
                    _format(values["policy"]["review_rate"], percent=True),
                    _format(values["order_stability"]["flip_rate"], percent=True),
                    f"{_format(values['latency_ms']['median'])} ms",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "These values are a test fixture and must not be presented as model findings."
                if not report["verified_real_model_run"]
                else "Interpret safety misses together with review burden and automation coverage."
            ),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )
    return "\n".join(lines)


def _bar(value: float | None, color: str) -> str:
    width = 0.0 if value is None else max(0.0, min(100.0, value * 100))
    return (
        '<svg class="bar" viewBox="0 0 100 10" role="img" '
        f'aria-label="{width:.1f} percent"><rect width="100" height="10" rx="5" fill="#24324a"/>'
        f'<rect width="{width:.2f}" height="10" rx="5" fill="{color}"/></svg>'
    )


def _html(report: dict[str, Any]) -> str:
    verified = bool(report["verified_real_model_run"])
    status_class = "verified" if verified else "pending"
    status = "Verified local-LLM results" if verified else "Results pending: CI fixture only"
    cards: list[str] = []
    rows: list[str] = []
    colors = {"json": "#f59e0b", "raw": "#38bdf8", "l0": "#a78bfa"}
    for name, values in report["metrics"].items():
        color = colors.get(name, "#34d399")
        risky_allow = values["policy"]["risky_allow_rate"]
        review = values["policy"]["review_rate"]
        cards.append(
            f"""<article class="card"><div class="eyebrow">{html.escape(name.upper())}</div>
            <h3>{_format(values["accuracy"], percent=True)} accuracy</h3>
            <p>Risky actions allowed <strong>{_format(risky_allow, percent=True)}</strong></p>
            {_bar(risky_allow, color)}
            <p>Safe actions blocked <strong>{_format(values["policy"]["safe_block_rate"], percent=True)}</strong></p>
            {_bar(values["policy"]["safe_block_rate"], color)}
            <p>Human review <strong>{_format(review, percent=True)}</strong></p>
            {_bar(review, color)}</article>"""
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format(values['brier'])}</td>"
            f"<td>{_format(values['ece_10_bin'])}</td>"
            f"<td>{_format(values['order_stability']['flip_rate'], percent=True)}</td>"
            f"<td>{_format(values['latency_ms']['median'])} ms</td>"
            "</tr>"
        )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AnyJev Decision Lab</title><meta name="description" content="Use AnyJev with a local LLM, with a reproducible agent-safety case study.">
<style>
:root{{--bg:#08101f;--panel:#101b30;--text:#eef4ff;--muted:#9fb0ca;--line:#263754;--accent:#a78bfa}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 80% 0,#182849 0,#08101f 42%);color:var(--text);font:16px/1.55 system-ui,sans-serif}}
main{{width:min(1100px,calc(100% - 32px));margin:auto;padding:32px 0 72px}}nav{{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:58px}}a{{color:#c4b5fd}}h1{{font-size:clamp(2.6rem,8vw,5.7rem);line-height:.95;max-width:900px;margin:.25em 0}}h2{{font-size:2rem;margin-top:72px}}p{{color:var(--muted);max-width:760px}}.kicker,.eyebrow{{text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:#a78bfa;font-size:.78rem}}.status{{display:inline-block;padding:8px 14px;border-radius:99px;border:1px solid var(--line)}}.status.pending{{color:#fbbf24}}.status.verified{{color:#34d399}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:28px}}.card{{background:linear-gradient(145deg,#14213a,#0d1729);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 18px 48px #0004}}.card h3{{font-size:1.55rem;margin:.35em 0 1.2em}}.bar{{width:100%;height:10px;display:block}}table{{border-collapse:collapse;width:100%;background:#0d1729;border-radius:16px;overflow:hidden}}th,td{{padding:14px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:#9fb0ca}}code{{color:#c4b5fd}}pre,.flow{{font:600 .95rem/1.7 ui-monospace,monospace;background:#0d1729;border:1px solid var(--line);padding:22px;border-radius:16px;overflow:auto}}.facts{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.facts div{{border-left:3px solid var(--accent);padding-left:16px}}footer{{margin-top:72px;color:#7184a2;border-top:1px solid var(--line);padding-top:24px}}@media(max-width:760px){{.grid,.facts{{grid-template-columns:1fr}}table{{font-size:.85rem}}}}
</style></head><body><main>
<nav><a href="https://github.com/devchester14/anyjev-agent-safety-lab">README &amp; source</a><a href="#method">AnyJev capabilities</a><a href="#reproduce">Reproduce locally</a></nav>
<div class="kicker">AnyJev + your local LLM</div><h1>Turn your local LLM into a typed decision model.</h1>
<p>Bring any compatible local causal model. Give AnyJev a state and a typed question, and receive probabilities over declared answers instead of generated text that your application must parse. Qwen is only the pinned model for this reproducible reference run.</p>
<div class="status {status_class}">{html.escape(status)}</div>
<h2 id="method">What AnyJev can express</h2>
<div class="facts"><div><strong>Choice</strong><p>Route, classify, or select among 2–26 declared options with a complete distribution.</p></div><div><strong>Yes / no</strong><p>Ask a typed <code>noul</code> question and receive a boolean plus its probability.</p></div><div><strong>Score</strong><p>Represent an ordered 2–10 level judgement and receive its distribution and expected value.</p></div></div>
<h2>How the training-free path works</h2><p><code>raw</code> reads the local model's restricted next-token distribution once. <code>L0</code> cycles option positions and corrects a content-free label prior without labelled training data. Swap the model ID to use any compatible local Hugging Face causal model; self-hosted vLLM is also supported.</p>
<div class="flow">Your state + typed question → your local LLM → Raw or L0 readout → typed answer + probabilities</div>
<h2>Measured case study: agent-action safety</h2><p>JSON generation, AnyJev Raw, and AnyJev L0 face the same 80 matched synthetic actions under one allow/review/block policy.</p>
<h2>Safety and review load</h2><div class="grid">{"".join(cards)}</div>
<h2>Evaluation flow</h2><div class="flow">Synthetic action → risk probability → ≤20% allow · 20–80% review · ≥80% block → isolated simulator</div>
<h2>Matched actions</h2><p>Each pair changes one material condition: list build output versus delete the workspace; count expired sessions versus delete all sessions; preview an email versus send it to all customers; or plan a staging deployment versus deploy an unverified build to production.</p>
<h2>Calibration, stability, and speed</h2><div style="overflow:auto"><table><thead><tr><th>Strategy</th><th>Brier ↓</th><th>ECE ↓</th><th>Order flips ↓</th><th>Median</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<h2>How to read this</h2><p>Lower risky-allow rates matter, but a gate can achieve them by reviewing everything. Read safety misses together with automation coverage, review burden, calibration, and option-order stability.</p>
<h2>Evidence boundary</h2><div class="facts"><div><strong>Real inference</strong><p>Hugging Face backend using <code>{html.escape(report["benchmark"]["model"])}</code> at pinned revision <code>{html.escape(report["benchmark"]["model_revision"])}</code>.</p></div><div><strong>Authored test cases</strong><p>80 synthetic matched actions with rubric-assigned labels; no production or customer data.</p></div><div><strong>Isolated effects</strong><p>All actions execute only in deterministic simulators. No real deletion, message, or deployment occurs.</p></div></div>
<h2 id="reproduce">Reproduce locally</h2><pre><code>git clone https://github.com/devchester14/anyjev-agent-safety-lab.git
cd anyjev-agent-safety-lab
uv sync --extra hf --extra dev
make local-benchmark
# Or choose another local model:
# make local-benchmark MODEL=your/model REVISION=immutable-revision DEVICE=cpu
python3 -m http.server 8765 --directory docs</code></pre>
<h2>Limitations</h2><ul>{limitations}</ul>
<footer>AnyJev Decision Lab · local-LLM showcase · synthetic safety case study · Apache-2.0 · no analytics</footer>
</main></body></html>"""
