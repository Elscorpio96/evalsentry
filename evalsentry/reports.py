"""Markdown (for GitHub job summaries and PR comments) and a self-contained HTML report."""

from __future__ import annotations

from importlib import resources

from jinja2 import Environment, select_autoescape

from .compare import Comparison, GateResult
from .results import CaseResult, RunResult


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(x: float | None) -> str:
    return "—" if x is None else f"${x:.4f}"


def markdown(run: RunResult, gate: GateResult, comparison: Comparison | None = None, max_failures: int = 15) -> str:
    icon = "✅" if gate.passed else "❌"
    lines = [f"## {icon} EvalSentry — {run.suite}", ""]
    if run.description:
        lines += [f"_{run.description}_", ""]
    lines += [f"**Pass rate {_pct(run.pass_rate)}** · {len(run.results)} calls · cost {_money(run.total_cost)} · "
              f"p95 latency {run.p95_latency_ms:.0f} ms · {run.duration_s:.1f} s", ""]
    if gate.checks:
        lines += ["| Gate | Result |", "|---|---|"]
        lines += [f"| `{name}` | {'✅' if ok else '❌'} {detail} |" for name, ok, detail in gate.checks]
        lines.append("")
    lines += ["| Prompt | Provider | Pass rate | Mean score | p50 / p95 ms | Cost | Flaky |", "|---|---|---|---|---|---|---|"]
    for g in run.groups():
        lines.append(f"| `{g.prompt_id}` | `{g.provider_id}` | {g.passed}/{g.cases} ({_pct(g.pass_rate)}) | {g.mean_score:.2f} | "
                     f"{g.p50_latency_ms:.0f} / {g.p95_latency_ms:.0f} | {_money(g.cost_usd)} | {len(g.flaky) or ''} |")
    lines.append("")
    if comparison:
        lines += [f"### Compared with baseline: {_pct(comparison.pass_rate_before)} → {_pct(comparison.pass_rate_after)}", ""]
        if comparison.regressions:
            lines += ["**Regressions** (passed before, failing now):", ""]
            lines += [f"- `{c.test_id}` · `{c.prompt_id}` · `{c.provider_id}` (score {c.before:.2f} → {c.after:.2f})" for c in comparison.regressions]
            lines.append("")
        if comparison.fixes:
            lines += [f"**Fixed:** {', '.join(f'`{c.test_id}`' for c in comparison.fixes)}", ""]
        if not comparison.regressions and not comparison.fixes:
            lines += ["No tests changed outcome.", ""]
    failures = [r for r in run.results if not r.passed]
    if failures:
        lines += [f"<details><summary>{len(failures)} failing case(s)</summary>", ""]
        for r in failures[:max_failures]:
            why = r.error or "; ".join(a["reason"] for a in r.assertions if not a["passed"])
            lines.append(f"- **{r.test_id}** · `{r.prompt_id}` · `{r.provider_id}` — {why.replace('|', '/')[:300]}")
        if len(failures) > max_failures:
            lines.append(f"- … and {len(failures) - max_failures} more (see the HTML report)")
        lines += ["", "</details>", ""]
    return "\n".join(lines)


def html(run: RunResult, gate: GateResult, comparison: Comparison | None = None) -> str:
    env = Environment(autoescape=select_autoescape(default=True, default_for_string=True))
    env.filters.update(pct=_pct, money=_money)
    template = env.from_string(resources.files("evalsentry").joinpath("report.html.j2").read_text(encoding="utf-8"))
    tests = sorted({r.test_id for r in run.results}, key=lambda t: [r.test_id for r in run.results].index(t))
    columns = sorted({(r.prompt_id, r.provider_id) for r in run.results})
    cell: dict[tuple[str, str, str], list[CaseResult]] = {}
    for r in run.results:
        cell.setdefault((r.test_id, r.prompt_id, r.provider_id), []).append(r)
    regressed = {c.key for c in comparison.regressions} if comparison else set()
    return template.render(run=run, gate=gate, comparison=comparison, groups=run.groups(), tests=tests, columns=columns, cell=cell,
                           regressed=regressed, failures=[r for r in run.results if not r.passed])
