"""Run results: what was asked, what came back, how it was graded. Serialised as versioned JSON."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class CaseResult:
    test_id: str
    prompt_id: str
    provider_id: str
    repeat: int
    vars: dict[str, Any]
    prompt: str
    output: str
    passed: bool
    score: float
    assertions: list[dict[str, Any]]
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    cached: bool = False
    error: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.test_id}|{self.prompt_id}|{self.provider_id}"


@dataclass
class GroupStats:
    prompt_id: str
    provider_id: str
    cases: int
    passed: int
    pass_rate: float
    mean_score: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    errors: int
    flaky: list[str]


@dataclass
class RunResult:
    suite: str
    description: str
    started_at: str
    duration_s: float
    results: list[CaseResult]
    version: int = SCHEMA_VERSION
    tool_version: str = ""

    # ---------------------------------------------------------------- aggregates
    def groups(self) -> list[GroupStats]:
        keys = sorted({(r.prompt_id, r.provider_id) for r in self.results})
        return [stats(self.results, p, m) for p, m in keys]

    @property
    def pass_rate(self) -> float:
        verdicts = case_verdicts(self.results)
        return sum(verdicts.values()) / len(verdicts) if verdicts else 0.0

    @property
    def total_cost(self) -> float | None:
        costs = [r.cost_usd for r in self.results if r.cost_usd is not None and not r.cached]
        return round(sum(costs), 6) if costs else None

    @property
    def p95_latency_ms(self) -> float:
        return percentile([r.latency_ms for r in self.results if not r.error], 95)

    # ---------------------------------------------------------------- persistence
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported results version {data.get('version')!r}")
        data["results"] = [CaseResult(**r) for r in data["results"]]
        return cls(**data)


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def case_verdicts(results: list[CaseResult]) -> dict[str, bool]:
    """One verdict per (test, prompt, provider). With repeats, a case passes only if EVERY repeat passed."""
    out: dict[str, bool] = {}
    for r in results:
        out[r.key] = out.get(r.key, True) and r.passed
    return out


def stats(results: list[CaseResult], prompt_id: str, provider_id: str) -> GroupStats:
    rs = [r for r in results if r.prompt_id == prompt_id and r.provider_id == provider_id]
    verdicts = case_verdicts(rs)
    by_key: dict[str, set[bool]] = {}
    for r in rs:
        by_key.setdefault(r.test_id, set()).add(r.passed)
    costs = [r.cost_usd for r in rs if r.cost_usd is not None and not r.cached]
    lat = [r.latency_ms for r in rs if not r.error]
    passed = sum(verdicts.values())
    return GroupStats(
        prompt_id=prompt_id, provider_id=provider_id, cases=len(verdicts), passed=passed,
        pass_rate=round(passed / len(verdicts), 4) if verdicts else 0.0,
        mean_score=round(sum(r.score for r in rs) / len(rs), 4) if rs else 0.0,
        p50_latency_ms=percentile(lat, 50), p95_latency_ms=percentile(lat, 95),
        cost_usd=round(sum(costs), 6) if costs else None,
        input_tokens=sum(r.input_tokens for r in rs), output_tokens=sum(r.output_tokens for r in rs),
        errors=sum(1 for r in rs if r.error),
        flaky=sorted(t for t, outcomes in by_key.items() if len(outcomes) > 1),
    )
