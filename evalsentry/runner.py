"""Runs the test matrix (tests × prompts × providers × repeats) concurrently."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from . import __version__
from .assertions import Context, run_assertion
from .cache import Cache
from .config import Suite, TestCase
from .providers import Completion, Provider, ProviderError, make_provider
from .results import CaseResult, RunResult
from .templating import TemplateError, render

Progress = Callable[[int, int], None]


def select(tests: list[TestCase], only: str = "", tag: str = "") -> list[TestCase]:
    out = tests
    if only:
        out = [t for t in out if only.lower() in t.id.lower() or only.lower() in t.description.lower()]
    if tag:
        out = [t for t in out if tag in t.tags]
    return out


async def run_suite(
    suite: Suite,
    *,
    cache: Cache | None = None,
    only: str = "",
    tag: str = "",
    prompts: list[str] | None = None,
    providers_only: list[str] | None = None,
    concurrency: int | None = None,
    progress: Progress | None = None,
    client: httpx.AsyncClient | None = None,
    name: str = "suite",
) -> RunResult:
    cache = cache or Cache(Path(".evalsentry/cache"), enabled=False)
    tests = select(suite.all_tests(), only, tag)
    if not tests:
        raise ValueError("no tests match the filter")
    started, t0 = datetime.now(UTC), time.perf_counter()
    own_client = client is None
    client = client or httpx.AsyncClient(headers={"user-agent": f"evalsentry/{__version__}"})
    try:
        providers = {p.id: make_provider(p, client) for p in suite.providers}
        targets = [p for p in suite.providers if p.id != suite.judge and (not providers_only or p.id in providers_only)]
        prompt_cfgs = [p for p in suite.prompts if not prompts or p.id in prompts]
        unknown = set(prompts or []) - {p.id for p in suite.prompts} | (set(providers_only or []) - {p.id for p in suite.providers})
        if unknown:
            raise ValueError(f"unknown prompt/provider id(s): {', '.join(sorted(unknown))}")
        if not targets or not prompt_cfgs:
            raise ValueError("nothing to run: no providers or prompts selected (the judge provider is never a test target)")
        judge = _judge_fn(providers[suite.judge], cache) if suite.judge else None
        sem = asyncio.Semaphore(concurrency or suite.concurrency)
        jobs = [(t, pr, pv, r) for t in tests for pr in prompt_cfgs for pv in targets for r in range(suite.defaults.repeat)]
        done = 0

        async def one(job: tuple[TestCase, object, object, int]) -> CaseResult:
            nonlocal done
            t, pr, pv, rep = job
            async with sem:
                res = await _run_case(suite, t, pr, providers[pv.id], rep, cache, judge)  # type: ignore[attr-defined]
            done += 1
            if progress:
                progress(done, len(jobs))
            return res

        results = await asyncio.gather(*(one(j) for j in jobs))
    finally:
        if own_client:
            await client.aclose()
    return RunResult(suite=name, description=suite.description, started_at=started.isoformat(timespec="seconds"),
                     duration_s=round(time.perf_counter() - t0, 2), results=list(results), tool_version=__version__)


def _judge_fn(provider: Provider, cache: Cache):  # type: ignore[no-untyped-def]
    async def judge(system: str, prompt: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        key = Cache.key(provider.cfg.model_dump(), messages, 0)
        c = cache.get(key)
        if c is None:
            c = await provider.complete(messages, {})
            cache.put(key, c)
        return c.text

    return judge


async def _run_case(suite: Suite, test: TestCase, prompt_cfg, provider: Provider, repeat: int, cache: Cache, judge) -> CaseResult:  # type: ignore[no-untyped-def]
    base = {"test_id": test.id, "prompt_id": prompt_cfg.id, "provider_id": provider.cfg.id, "repeat": repeat, "vars": test.vars,
            "description": test.description, "tags": test.tags}
    try:
        user = render(suite.prompt_text(prompt_cfg), test.vars)
        system = render(prompt_cfg.system, test.vars) if prompt_cfg.system else ""
    except TemplateError as e:
        return CaseResult(**base, prompt="", output="", passed=False, score=0.0, assertions=[], error=f"template error: {e}")
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]

    key = Cache.key(provider.cfg.model_dump(), messages, repeat)
    completion = cache.get(key)
    if completion is None:
        try:
            completion = await provider.complete(messages, test.vars)
        except ProviderError as e:
            completion = Completion(text="", error=str(e))
        except Exception as e:
            completion = Completion(text="", error=f"{type(e).__name__}: {e}")
        cache.put(key, completion)

    shown = f"[system]\n{system}\n\n[user]\n{user}" if system else user
    if completion.error:
        return CaseResult(**base, prompt=shown, output="", passed=False, score=0.0, assertions=[], error=completion.error,
                          model=provider.model_name)

    ctx = Context(vars=test.vars, prompt=user, completion=completion, base_dir=suite.base_dir, judge=judge)
    checks = [await run_assertion(a.model_dump(by_alias=True), completion.text, ctx) for a in test.assert_]
    total_w = sum(c.weight for c in checks)
    score = sum(c.score * c.weight for c in checks) / total_w if checks else 1.0
    passed = all(c.passed for c in checks) if test.threshold is None else score >= test.threshold
    return CaseResult(
        **base, prompt=shown, output=completion.text, passed=passed, score=round(score, 4),
        assertions=[{"type": c.type, "passed": c.passed, "score": c.score, "reason": c.reason, "weight": c.weight} for c in checks],
        latency_ms=round(completion.latency_ms, 1), input_tokens=completion.input_tokens, output_tokens=completion.output_tokens,
        cost_usd=completion.cost_usd, model=completion.model, cached=completion.cached,
    )
