from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalsentry.assertions import REGISTRY, Context, rouge_l, run_assertion
from evalsentry.providers import Completion


def ctx(**kw) -> Context:
    defaults = dict(vars={"city": "Abuja"}, prompt="What is the capital?", completion=Completion(text="", latency_ms=120, cost_usd=0.002),
                    base_dir=Path("."))
    defaults.update(kw)
    return Context(**defaults)


async def check(cfg: dict, out: str, **kw):
    return await run_assertion(cfg, out, ctx(**kw))


@pytest.mark.parametrize("cfg, out, ok", [
    ({"type": "equals", "value": "Abuja"}, " Abuja\n", True),
    ({"type": "equals", "value": "abuja", "ignore_case": True}, "ABUJA", True),
    ({"type": "equals", "value": "Abuja"}, "Lagos", False),
    ({"type": "contains", "value": "{{ city }}"}, "The capital is Abuja.", True),
    ({"type": "contains", "value": ["Abuja", "FCT"]}, "Abuja", False),
    ({"type": "not_contains", "value": ["Lagos"]}, "Abuja", True),
    ({"type": "not_contains", "value": "lagos", "ignore_case": True}, "LAGOS", False),
    ({"type": "contains_any", "value": ["x", "Abu"]}, "Abuja", True),
    ({"type": "contains_any", "value": ["x", "y"]}, "Abuja", False),
    ({"type": "regex", "pattern": r"^\d{3}-\d{4}$"}, "555-1234", True),
    ({"type": "regex", "pattern": "abuja", "ignore_case": True}, "ABUJA", True),
    ({"type": "starts_with", "value": "Yes"}, "  Yes, you can.", True),
    ({"type": "starts_with", "value": "Yes"}, "No.", False),
    ({"type": "json_valid"}, '```json\n{"a": 1}\n```', True),
    ({"type": "json_valid"}, "{a: 1}", False),
    ({"type": "json_field", "path": "items.1.name", "equals": "b"}, '{"items": [{"name": "a"}, {"name": "b"}]}', True),
    ({"type": "json_field", "path": "total", "equals": "12.5"}, '{"total": 12.5}', True),
    ({"type": "json_field", "path": "missing"}, '{"a": 1}', False),
    ({"type": "json_field", "path": "reply", "contains": "TODAY"}, '{"reply": "Arrives today"}', True),
    ({"type": "length", "min_words": 2, "max_chars": 20}, "two words", True),
    ({"type": "length", "max_words": 2}, "one two three", False),
    ({"type": "latency", "max_ms": 200}, "x", True),
    ({"type": "latency", "max_ms": 100}, "x", False),
    ({"type": "cost", "max_usd": 0.01}, "x", True),
    ({"type": "cost", "max_usd": 0.001}, "x", False),
    ({"type": "refusal"}, "I'm sorry, but I can't help with that.", True),
    ({"type": "refusal", "expect": False}, "Sure, here is how to reset your password.", True),
    ({"type": "refusal", "expect": False}, "As an AI, I cannot comply.", False),
    ({"type": "contains", "value": "Lagos", "negate": True}, "Abuja", True),
])
async def test_builtin_checks(cfg, out, ok):
    r = await check(cfg, out)
    assert r.passed is ok, r.reason
    assert 0.0 <= r.score <= 1.0


async def test_partial_contains_score():
    r = await check({"type": "contains", "value": ["a", "b", "c", "d"]}, "a b")
    assert r.score == 0.5 and "missing" in r.reason


async def test_json_schema_inline_and_file(tmp_path):
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer", "minimum": 1}}}
    assert (await check({"type": "json_schema", "schema": schema}, '{"n": 3}')).passed
    bad = await check({"type": "json_schema", "schema": schema}, '{"n": 0}')
    assert not bad.passed and "at n" in bad.reason
    (tmp_path / "s.json").write_text(json.dumps(schema))
    assert (await check({"type": "json_schema", "schema_file": "s.json"}, '{"n": 1}', base_dir=tmp_path)).passed
    assert not (await check({"type": "json_schema", "schema": schema}, "nope")).passed


async def test_similar_uses_rouge_l():
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == pytest.approx(1.0)
    assert rouge_l("", "x") == 0.0
    r = await check({"type": "similar", "reference": "Abuja is the capital of Nigeria", "threshold": 0.5},
                    "The capital of Nigeria is Abuja")
    assert r.passed and 0.5 <= r.score < 1


async def test_configuration_errors_fail_without_crashing():
    for cfg, msg in [({"type": "contains"}, "needs 'value'"), ({"type": "regex", "pattern": "("}, "invalid regex"),
                     ({"type": "json_schema"}, "needs 'schema'"), ({"type": "llm_judge", "rubric": "x"}, "needs `judge:`")]:
        r = await check(cfg, "out")
        assert not r.passed and msg in r.reason
    r = await check({"type": "cost", "max_usd": 1}, "x", completion=Completion(text=""))
    assert "price" in r.reason
    r = await check({"type": "nope"}, "x")
    assert "unknown assertion type" in r.reason and "contains" in r.reason


async def test_python_check_variants(tmp_path, monkeypatch):
    (tmp_path / "mychecks.py").write_text(
        "def b(o, v): return o == v['city']\n"
        "def t(o, v): return (False, 'nope')\n"
        "def f(o, v): return 0.7\n"
        "def d(o, v): return {'pass': True, 'score': 0.9, 'reason': 'good'}\n"
        "async def a(o, v): return True\n"
        "def boom(o, v): raise RuntimeError('broken check')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert (await check({"type": "python", "function": "mychecks:b"}, "Abuja")).passed
    assert (await check({"type": "python", "function": "mychecks:t"}, "x")).reason == "nope"
    assert (await check({"type": "python", "function": "mychecks:f", "threshold": 0.8}, "x")).passed is False
    d = await check({"type": "python", "function": "mychecks:d"}, "x")
    assert d.passed and d.score == 0.9 and d.reason == "good"
    assert (await check({"type": "python", "function": "mychecks:a"}, "x")).passed
    boom = await check({"type": "python", "function": "mychecks:boom"}, "x")
    assert not boom.passed and "broken check" in boom.reason
    assert "cannot load" in (await check({"type": "python", "function": "nomod:fn"}, "x")).reason


async def test_llm_judge_assertion():
    seen = {}

    async def judge(system: str, prompt: str) -> str:
        seen["prompt"] = prompt
        return '{"reasoning": "fine", "score": 8}'

    r = await check({"type": "llm_judge", "rubric": "Mentions {{ city }}", "threshold": 0.7}, "Abuja", judge=judge)
    assert r.passed and r.score == 0.8 and "Mentions Abuja" in seen["prompt"] and "What is the capital?" in seen["prompt"]

    async def bad_judge(system: str, prompt: str) -> str:
        return "I think it is good"

    r = await check({"type": "llm_judge", "rubric": "x"}, "y", judge=bad_judge)
    assert not r.passed and "judge error" in r.reason


def test_registry_is_complete():
    assert {"equals", "contains", "not_contains", "contains_any", "regex", "starts_with", "json_valid", "json_schema", "json_field",
            "length", "similar", "latency", "cost", "refusal", "python", "llm_judge"} <= set(REGISTRY)
