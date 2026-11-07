from __future__ import annotations

import pytest

from evalsentry.cache import Cache
from evalsentry.config import load_suite
from evalsentry.runner import run_suite

SUITE = """
providers:
  - id: good
    type: mock
    price: {input: 1, output: 1}
    responses: [{match: "Nigeria", output: "Abuja"}, {match: "Ghana", output: "Accra"}]
  - id: bad
    type: mock
    default_output: "Lagos"
  - id: j
    type: mock
    default_output: '{"score": 1, "reasoning": "ok"}'
judge: j
prompts:
  - id: p1
    system: "Country: {{ country }}"
    user: "Capital of {{ country }}?"
  - id: p2
    user: "{{ country }} capital, one word"
tests:
  - id: ng
    tags: [west]
    vars: {country: Nigeria, capital: Abuja}
  - id: gh
    vars: {country: Ghana, capital: Accra}
defaults:
  assert: [{type: contains, value: "{{ capital }}"}]
"""


async def test_matrix_excludes_judge_and_grades(write_suite):
    suite = load_suite(write_suite(SUITE))
    run = await run_suite(suite)
    assert len(run.results) == 2 * 2 * 2  # tests x prompts x providers (judge is not a target)
    by = {(r.test_id, r.prompt_id, r.provider_id): r for r in run.results}
    assert by[("ng", "p1", "good")].passed and not by[("ng", "p1", "bad")].passed
    assert by[("ng", "p1", "good")].prompt.startswith("[system]\nCountry: Nigeria")
    assert run.pass_rate == 0.5
    assert {g.provider_id: g.pass_rate for g in run.groups() if g.prompt_id == "p1"} == {"good": 1.0, "bad": 0.0}
    assert run.total_cost and run.total_cost > 0


async def test_filters(write_suite):
    suite = load_suite(write_suite(SUITE))
    run = await run_suite(suite, tag="west", prompts=["p2"], providers_only=["good"])
    assert [(r.test_id, r.prompt_id, r.provider_id) for r in run.results] == [("ng", "p2", "good")]
    assert len((await run_suite(suite, only="gh")).results) == 4
    with pytest.raises(ValueError, match="no tests match"):
        await run_suite(suite, only="zzz")
    with pytest.raises(ValueError, match="unknown prompt/provider"):
        await run_suite(suite, prompts=["nope"])
    with pytest.raises(ValueError, match="nothing to run"):
        await run_suite(suite, providers_only=["j"])


async def test_repeat_and_weighted_threshold(write_suite):
    suite = load_suite(write_suite(SUITE.replace("defaults:\n", "defaults:\n  repeat: 3\n")))
    run = await run_suite(suite, providers_only=["good"], prompts=["p1"])
    assert len(run.results) == 2 * 3 and sorted({r.repeat for r in run.results}) == [0, 1, 2]

    weighted = SUITE.replace("  - id: gh\n    vars: {country: Ghana, capital: Accra}",
                             "  - id: gh\n    threshold: 0.6\n    vars: {country: Ghana, capital: Accra}\n"
                             "    assert: [{type: contains, value: Accra, weight: 3}, {type: contains, value: Ghana}]")
    run = await run_suite(load_suite(write_suite(weighted)), providers_only=["good"], prompts=["p2"], only="gh")
    r = run.results[0]
    assert r.score == pytest.approx(4 / 5) and r.passed  # Accra (w3) + default capital check (w1) pass; "Ghana" (w1) fails


async def test_template_and_provider_errors_are_reported_not_raised(write_suite):
    suite = load_suite(write_suite(SUITE.replace('user: "Capital of {{ country }}?"', 'user: "{{ missing_var }}"')))
    run = await run_suite(suite, prompts=["p1"], providers_only=["good"])
    assert all(not r.passed and r.error.startswith("template error") for r in run.results)

    broken = SUITE.replace("  - id: bad\n    type: mock\n    default_output: \"Lagos\"",
                           "  - id: bad\n    type: openai\n    model: m\n    api_key_env: EVALSENTRY_TEST_UNSET_KEY")
    run = await run_suite(load_suite(write_suite(broken)), providers_only=["bad"], prompts=["p2"])
    assert all("EVALSENTRY_TEST_UNSET_KEY is not set" in r.error for r in run.results)


async def test_cache_round_trip(write_suite, tmp_path):
    suite = load_suite(write_suite(SUITE))
    cache = Cache(tmp_path / "cache")
    first = await run_suite(suite, cache=cache, providers_only=["good"])
    assert cache.hits == 0 and not any(r.cached for r in first.results)
    cache2 = Cache(tmp_path / "cache")
    second = await run_suite(suite, cache=cache2, providers_only=["good"])
    assert cache2.hits == len(second.results) and all(r.cached for r in second.results)
    assert second.total_cost == 0 or second.total_cost is None  # cached calls cost nothing
    assert [r.output for r in first.results] == [r.output for r in second.results]
    assert Cache.key({"a": 1}, [], 0) != Cache.key({"a": 1}, [], 1)  # repeats are cached separately
    disabled = Cache(tmp_path / "off", enabled=False)
    await run_suite(suite, cache=disabled, providers_only=["good"])
    assert not (tmp_path / "off").exists()


async def test_python_provider_example_resolves_next_to_suite():
    from conftest import EXAMPLES

    run = await run_suite(load_suite(EXAMPLES / "invoice-extractor" / "evalsentry.yaml"))
    assert run.pass_rate == 1.0 and len(run.results) == 6
