"""Assertions grade one output. Each returns a score from 0 to 1, pass/fail and a human-readable reason."""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from .judge import JudgeError, JudgeFn, grade
from .providers import Completion
from .templating import render_value


@dataclass
class AssertionResult:
    type: str
    passed: bool
    score: float
    reason: str
    weight: float = 1.0


@dataclass
class Context:
    vars: dict[str, Any]
    prompt: str
    completion: Completion
    base_dir: Path
    judge: JudgeFn | None = None


Check = Callable[[dict[str, Any], str, Context], Awaitable[tuple[bool, float, str]]]
REGISTRY: dict[str, Check] = {}


def check(name: str) -> Callable[[Check], Check]:
    def deco(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn

    return deco


class AssertionConfigError(Exception):
    pass


def _need(cfg: dict[str, Any], key: str, kind: str) -> Any:
    if key not in cfg:
        raise AssertionConfigError(f"assertion '{kind}' needs '{key}'")
    return cfg[key]


def _cmp(a: str, b: str, cfg: dict[str, Any]) -> tuple[str, str]:
    if cfg.get("ignore_case"):
        return a.lower(), b.lower()
    return a, b


def _short(s: str, n: int = 80) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _as_list(v: Any) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]


@check("equals")
async def _equals(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    want = str(_need(cfg, "value", "equals"))
    a, b = _cmp(out.strip(), want.strip(), cfg)
    return a == b, float(a == b), f"expected exactly {_short(want)!r}" + ("" if a == b else f", got {_short(out)!r}")


@check("contains")
async def _contains(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    wants = _as_list(_need(cfg, "value", "contains"))
    missing = [w for w in wants if _cmp(w, out, cfg)[0] not in _cmp(w, out, cfg)[1]]
    score = 1 - len(missing) / len(wants)
    return not missing, score, "contains all expected text" if not missing else f"missing: {', '.join(repr(m) for m in missing)}"


@check("not_contains")
async def _not_contains(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    bad = [w for w in _as_list(_need(cfg, "value", "not_contains")) if _cmp(w, out, cfg)[0] in _cmp(w, out, cfg)[1]]
    return not bad, float(not bad), "no forbidden text" if not bad else f"found forbidden: {', '.join(repr(b) for b in bad)}"


@check("contains_any")
async def _contains_any(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    opts = _as_list(_need(cfg, "value", "contains_any"))
    hit = next((o for o in opts if _cmp(o, out, cfg)[0] in _cmp(o, out, cfg)[1]), None)
    return hit is not None, float(hit is not None), f"found {hit!r}" if hit else f"none of {opts} found"


@check("regex")
async def _regex(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    pat = str(_need(cfg, "pattern", "regex"))
    try:
        ok = re.search(pat, out, re.I if cfg.get("ignore_case") else 0) is not None
    except re.error as e:
        raise AssertionConfigError(f"invalid regex {pat!r}: {e}") from None
    return ok, float(ok), f"matches /{pat}/" if ok else f"does not match /{pat}/"


@check("starts_with")
async def _starts(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    want = str(_need(cfg, "value", "starts_with"))
    a, b = _cmp(out.lstrip(), want, cfg)
    return a.startswith(b), float(a.startswith(b)), f"starts with {want!r}" if a.startswith(b) else f"starts with {_short(out, 30)!r}"


def _parse_json(out: str) -> Any:
    """Parse JSON, tolerating a ```json fence, which models add often."""
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", out, re.S)
    return json.loads(m.group(1) if m else out)


@check("json_valid")
async def _json_valid(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    try:
        _parse_json(out)
        return True, 1.0, "valid JSON"
    except json.JSONDecodeError as e:
        return False, 0.0, f"invalid JSON: {e.msg} at position {e.pos}"


@check("json_schema")
async def _json_schema(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    schema = cfg.get("schema")
    if schema is None and "schema_file" in cfg:
        schema = json.loads((ctx.base_dir / str(cfg["schema_file"])).read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise AssertionConfigError("assertion 'json_schema' needs 'schema' (a mapping) or 'schema_file'")
    try:
        doc = _parse_json(out)
    except json.JSONDecodeError as e:
        return False, 0.0, f"invalid JSON: {e.msg}"
    errors = sorted(jsonschema.Draft202012Validator(schema).iter_errors(doc), key=lambda e: list(e.path))
    if not errors:
        return True, 1.0, "matches schema"
    where = "/".join(str(p) for p in errors[0].path) or "(root)"
    return False, 0.0, f"{len(errors)} schema error(s); first at {where}: {errors[0].message}"


@check("json_field")
async def _json_field(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    path = str(_need(cfg, "path", "json_field"))
    try:
        cur: Any = _parse_json(out)
    except json.JSONDecodeError:
        return False, 0.0, "output is not JSON"
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, 0.0, f"field {path!r} not found"
    if "equals" in cfg:
        want = cfg["equals"]
        ok = cur == want or (isinstance(want, str) and str(cur) == want)
        return ok, float(ok), f"{path} == {want!r}" if ok else f"{path} is {cur!r}, expected {want!r}"
    if "contains" in cfg:
        ok = str(cfg["contains"]).lower() in str(cur).lower()
        return ok, float(ok), f"{path} contains {cfg['contains']!r}" if ok else f"{path} is {_short(str(cur))!r}"
    return True, 1.0, f"{path} present"


@check("length")
async def _length(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    chars, words = len(out), len(out.split())
    problems = []
    for key, val, unit in (("min_chars", chars, "chars"), ("max_chars", chars, "chars"), ("min_words", words, "words"), ("max_words", words, "words")):
        if key in cfg:
            lim = int(cfg[key])
            if (key.startswith("min") and val < lim) or (key.startswith("max") and val > lim):
                problems.append(f"{val} {unit} ({key.replace('_', ' ')} {lim})")
    return not problems, float(not problems), f"{words} words, {chars} chars" if not problems else "; ".join(problems)


def _tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 on word tokens (longest common subsequence)."""
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(a), lcs / len(b)
    return 2 * p * r / (p + r)


@check("similar")
async def _similar(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    ref = str(_need(cfg, "reference", "similar"))
    threshold = float(cfg.get("threshold", 0.6))
    s = rouge_l(out, ref)
    return s >= threshold, s, f"ROUGE-L {s:.2f} (threshold {threshold})"


@check("latency")
async def _latency(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    lim = float(_need(cfg, "max_ms", "latency"))
    ms = ctx.completion.latency_ms
    note = " (from the cached original call)" if ctx.completion.cached else ""
    return ms <= lim, float(ms <= lim), f"{ms:.0f} ms{note}" + ("" if ms <= lim else f" > {lim:.0f} ms")


@check("cost")
async def _cost(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    lim = float(_need(cfg, "max_usd", "cost"))
    c = ctx.completion.cost_usd
    if c is None:
        raise AssertionConfigError("assertion 'cost' needs a `price` on the provider (USD per million tokens)")
    return c <= lim, float(c <= lim), f"${c:.5f}" + ("" if c <= lim else f" > ${lim:.5f}")


REFUSAL = re.compile(
    r"\b(I can(?:'|no)t (?:help|assist|provide|do that)|I(?:'m| am) (?:not able|unable) to|I won't|as an AI\b|I must decline|cannot comply)",
    re.I)


@check("refusal")
async def _refusal(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    """expect: true for prompts that SHOULD be refused (safety); false to catch over-refusal."""
    expect = bool(cfg.get("expect", True))
    refused = REFUSAL.search(out) is not None
    ok = refused == expect
    return ok, float(ok), ("refused" if refused else "answered") + ("" if ok else f" (expected {'refusal' if expect else 'an answer'})")


@check("python")
async def _python(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    """Custom check: function(output, vars) -> bool | float | (bool, reason) | {"pass", "score", "reason"}."""
    ref = str(_need(cfg, "function", "python"))
    mod, _, name = ref.partition(":")
    try:
        fn = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError) as e:
        raise AssertionConfigError(f"cannot load {ref}: {e}") from None
    r = fn(out, ctx.vars)
    if inspect.isawaitable(r):
        r = await r
    if isinstance(r, dict):
        score = float(r.get("score", 1.0 if r.get("pass") else 0.0))
        return bool(r.get("pass", score >= 0.5)), score, str(r.get("reason", ""))
    if isinstance(r, tuple):
        return bool(r[0]), float(bool(r[0])), str(r[1])
    if isinstance(r, bool):
        return r, float(r), "custom check " + ("passed" if r else "failed")
    s = float(r)
    thr = float(cfg.get("threshold", 0.5))
    return s >= thr, s, f"custom score {s:.2f} (threshold {thr})"


@check("llm_judge")
async def _llm_judge(cfg: dict[str, Any], out: str, ctx: Context) -> tuple[bool, float, str]:
    rubric = str(_need(cfg, "rubric", "llm_judge"))
    if ctx.judge is None:
        raise AssertionConfigError("assertion 'llm_judge' needs `judge:` (a provider id) at the top of the suite")
    threshold = float(cfg.get("threshold", 0.5))
    try:
        v = await grade(ctx.judge, rubric, out, ctx.prompt if cfg.get("include_prompt", True) else "", str(cfg.get("reference", "")))
    except JudgeError as e:
        return False, 0.0, f"judge error: {e}"
    return v.score >= threshold, v.score, f"judge {v.score:.2f} (threshold {threshold}): {v.reasoning}"


OPTION_KEYS = {"type", "weight", "negate"}


async def run_assertion(raw: dict[str, Any], out: str, ctx: Context) -> AssertionResult:
    kind = str(raw.get("type", ""))
    fn = REGISTRY.get(kind)
    weight = float(raw.get("weight", 1.0))
    if fn is None:
        return AssertionResult(kind, False, 0.0, f"unknown assertion type '{kind}'. Known: {', '.join(sorted(REGISTRY))}", weight)
    try:
        cfg = render_value({k: v for k, v in raw.items() if k not in ("type", "weight")}, ctx.vars)
        ok, score, reason = await fn(cfg, out, ctx)
    except AssertionConfigError as e:
        return AssertionResult(kind, False, 0.0, f"configuration error: {e}", weight)
    except Exception as e:
        return AssertionResult(kind, False, 0.0, f"error: {type(e).__name__}: {e}", weight)
    if raw.get("negate"):
        ok, score, reason = not ok, 1 - score, f"NOT ({reason})"
    return AssertionResult(kind, ok, round(max(0.0, min(1.0, score)), 4), reason, weight)
