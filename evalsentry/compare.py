"""Compare a run with a baseline, and evaluate the CI quality gate."""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Gate
from .results import RunResult


@dataclass
class Change:
    key: str
    test_id: str
    prompt_id: str
    provider_id: str
    before: float
    after: float


@dataclass
class Comparison:
    regressions: list[Change] = field(default_factory=list)  # passed before, fails now
    fixes: list[Change] = field(default_factory=list)  # failed before, passes now
    new_tests: list[str] = field(default_factory=list)
    removed_tests: list[str] = field(default_factory=list)
    pass_rate_before: float = 0.0
    pass_rate_after: float = 0.0
    cost_before: float | None = None
    cost_after: float | None = None
    score_drops: list[Change] = field(default_factory=list)  # still passing, but score fell by >= 0.1


def _keyed(run: RunResult, ignore_prompt: bool) -> tuple[dict[str, bool], dict[str, float], dict[str, str]]:
    """Verdicts and mean scores per case key. With ignore_prompt the key is test|*|provider."""
    verdicts: dict[str, bool] = {}
    scores: dict[str, list[float]] = {}
    prompt_of: dict[str, str] = {}
    for r in run.results:
        key = f"{r.test_id}|*|{r.provider_id}" if ignore_prompt else r.key
        verdicts[key] = verdicts.get(key, True) and r.passed
        scores.setdefault(key, []).append(r.score)
        prompt_of[key] = r.prompt_id
    return verdicts, {k: sum(v) / len(v) for k, v in scores.items()}, prompt_of


def compare(baseline: RunResult, current: RunResult, ignore_prompt: bool | None = None) -> Comparison:
    """Match cases by test, prompt and provider. When each run used exactly one prompt and they differ
    (e.g. comparing prompt v3 with a v2 baseline), match by test and provider only."""
    if ignore_prompt is None:
        pb, pa = {r.prompt_id for r in baseline.results}, {r.prompt_id for r in current.results}
        ignore_prompt = len(pb) == 1 and len(pa) == 1 and pb != pa
    vb, sb, _ = _keyed(baseline, ignore_prompt)
    va, sa, prompt_now = _keyed(current, ignore_prompt)
    c = Comparison(pass_rate_before=baseline.pass_rate, pass_rate_after=current.pass_rate,
                   cost_before=baseline.total_cost, cost_after=current.total_cost)
    for key in sorted(set(vb) & set(va)):
        t, p, m = key.split("|")
        ch = Change(key.replace("|*|", f"|{prompt_now[key]}|"), t, prompt_now[key] if p == "*" else p, m, round(sb[key], 3), round(sa[key], 3))
        if vb[key] and not va[key]:
            c.regressions.append(ch)
        elif not vb[key] and va[key]:
            c.fixes.append(ch)
        elif va[key] and sb[key] - sa[key] >= 0.1:
            c.score_drops.append(ch)
    c.new_tests = sorted(set(va) - set(vb))
    c.removed_tests = sorted(set(vb) - set(va))
    return c


@dataclass
class GateResult:
    passed: bool
    checks: list[tuple[str, bool, str]]  # (rule, ok, detail)


def evaluate_gate(gate: Gate, run: RunResult, comparison: Comparison | None = None) -> GateResult:
    checks: list[tuple[str, bool, str]] = []
    if gate.min_pass_rate is not None:
        ok = run.pass_rate >= gate.min_pass_rate
        checks.append(("min_pass_rate", ok, f"pass rate {run.pass_rate:.1%} (minimum {gate.min_pass_rate:.0%})"))
    if gate.max_cost_usd is not None:
        cost = run.total_cost or 0.0
        checks.append(("max_cost_usd", cost <= gate.max_cost_usd, f"cost ${cost:.4f} (maximum ${gate.max_cost_usd:.4f})"))
    if gate.max_p95_latency_ms is not None:
        p95 = run.p95_latency_ms
        checks.append(("max_p95_latency_ms", p95 <= gate.max_p95_latency_ms, f"p95 latency {p95:.0f} ms (maximum {gate.max_p95_latency_ms:.0f} ms)"))
    if gate.max_regressions is not None and comparison is not None:
        n = len(comparison.regressions)
        checks.append(("max_regressions", n <= gate.max_regressions, f"{n} regression(s) vs baseline (maximum {gate.max_regressions})"))
    errors = sum(1 for r in run.results if r.error)
    if errors:
        checks.append(("no_errors", False, f"{errors} case(s) errored (provider, template or code errors)"))
    return GateResult(passed=all(ok for _, ok, _ in checks), checks=checks)
