from __future__ import annotations

import json

import pytest
from conftest import case, run_of

from evalsentry.compare import compare, evaluate_gate
from evalsentry.config import Gate
from evalsentry.reports import html, markdown
from evalsentry.results import RunResult, case_verdicts, percentile


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([100.0], 95) == 100.0
    assert percentile(list(map(float, range(1, 101))), 50) == pytest.approx(50.5)


def test_a_case_passes_only_if_every_repeat_passes_and_flaky_is_flagged():
    run = run_of(case("a", True), case("a", False, repeat=1), case("b", True), case("b", True, repeat=1))
    assert case_verdicts(run.results) == {"a|v1|m": False, "b|v1|m": True}
    assert run.pass_rate == 0.5
    assert run.groups()[0].flaky == ["a"]


def test_compare_same_prompt():
    base = run_of(case("a", True), case("b", False), case("c", True, score=0.9), case("gone", True))
    cur = run_of(case("a", False), case("b", True), case("c", True, score=0.7), case("new", True))
    c = compare(base, cur)
    assert [x.test_id for x in c.regressions] == ["a"]
    assert [x.test_id for x in c.fixes] == ["b"]
    assert [x.test_id for x in c.score_drops] == ["c"]
    assert c.new_tests == ["new|v1|m"] and c.removed_tests == ["gone|v1|m"]


def test_compare_across_prompt_versions_matches_by_test_and_provider():
    base = run_of(case("a", True, prompt="v2"), case("b", True, prompt="v2"))
    cur = run_of(case("a", False, prompt="v3"), case("b", True, prompt="v3"))
    c = compare(base, cur)
    assert [(x.test_id, x.prompt_id, x.key) for x in c.regressions] == [("a", "v3", "a|v3|m")]
    assert not c.new_tests and not c.removed_tests
    strict = compare(base, cur, ignore_prompt=False)
    assert not strict.regressions and len(strict.new_tests) == 2


def test_gate_rules():
    run = run_of(case("a", True, latency=100, cost=0.02), case("b", False, latency=900, cost=0.03))
    g = evaluate_gate(Gate(min_pass_rate=0.5, max_cost_usd=0.1, max_p95_latency_ms=1000), run)
    assert g.passed and [n for n, _, _ in g.checks] == ["min_pass_rate", "max_cost_usd", "max_p95_latency_ms"]
    g = evaluate_gate(Gate(min_pass_rate=0.9, max_cost_usd=0.01, max_p95_latency_ms=500), run)
    assert not g.passed and all(not ok for _, ok, _ in g.checks)
    comp = compare(run_of(case("b", True)), run)
    assert not evaluate_gate(Gate(max_regressions=0), run, comp).passed
    assert evaluate_gate(Gate(max_regressions=1), run, comp).passed
    errored = run_of(case("a", False, error="HTTP 500"))
    assert ("no_errors", False) == evaluate_gate(Gate(), errored).checks[0][:2]
    assert evaluate_gate(Gate(), run_of(case("a", True))).passed  # no rules, no errors


def test_save_load_round_trip(tmp_path):
    run = run_of(case("a", True, cost=0.01), case("b", False))
    p = tmp_path / "r.json"
    run.save(p)
    back = RunResult.load(p)
    assert back.results == run.results and back.pass_rate == run.pass_rate
    data = json.loads(p.read_text())
    data["version"] = 99
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        RunResult.load(p)


def test_markdown_report():
    base = run_of(case("a", True))
    run = run_of(case("a", False), case("b|pipe", False, error="x | y"))
    md = markdown(run, evaluate_gate(Gate(min_pass_rate=0.8, max_regressions=0), run, compare(base, run)), compare(base, run))
    assert md.startswith("## ❌ EvalSentry — t")
    assert "| `min_pass_rate` | ❌" in md and "**Regressions**" in md and "- `a` · `v1` · `m`" in md
    assert "x / y" in md  # pipes are escaped so tables do not break
    ok = run_of(case("a", True))
    assert "## ✅" in markdown(ok, evaluate_gate(Gate(), ok))


def test_html_report_is_self_contained_and_escapes_output():
    r = case("a", False)
    r.output = "<script>alert(1)</script>"
    run = run_of(r, case("b", True))
    page = html(run, evaluate_gate(Gate(min_pass_rate=1), run), compare(run_of(case("a", True), case("b", True)), run))
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "Quality gate failed" in page and 'class="cell fail reg"' in page
    assert "http://" not in page and "https://" not in page  # no external assets
