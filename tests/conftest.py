from __future__ import annotations

from pathlib import Path

import pytest

from evalsentry.providers import Completion
from evalsentry.results import CaseResult, RunResult

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"


def case(test_id: str, passed: bool, *, prompt: str = "v1", provider: str = "m", score: float | None = None, repeat: int = 0,
         latency: float = 100.0, cost: float | None = None, error: str = "") -> CaseResult:
    return CaseResult(test_id=test_id, prompt_id=prompt, provider_id=provider, repeat=repeat, vars={}, prompt="p", output="o",
                      passed=passed, score=(1.0 if passed else 0.0) if score is None else score, assertions=[],
                      latency_ms=latency, cost_usd=cost, error=error)


def run_of(*results: CaseResult) -> RunResult:
    return RunResult(suite="t", description="", started_at="2025-01-01T00:00:00+00:00", duration_s=1.0, results=list(results),
                     tool_version="test")


@pytest.fixture
def completion() -> Completion:
    return Completion(text="", latency_ms=120.0, input_tokens=100, output_tokens=50, cost_usd=0.001)


@pytest.fixture
def write_suite(tmp_path: Path):
    def _write(text: str, name: str = "evalsentry.yaml") -> Path:
        p = tmp_path / name
        p.write_text(text, encoding="utf-8")
        return p

    return _write
