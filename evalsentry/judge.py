"""LLM-as-judge grading, and calibration of the judge against human labels."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

JUDGE_SYSTEM = """You are a strict, impartial evaluator of AI assistant outputs.
Grade ONLY against the rubric. Do not reward length, confidence or style unless the rubric asks for it.
If the output is empty, off-topic or refuses without reason, it fails.
Reply with JSON only: {"reasoning": "<one or two sentences>", "score": <number from 0 to 1>}"""


def judge_prompt(rubric: str, output: str, task: str = "", reference: str = "") -> str:
    parts = [f"<rubric>\n{rubric.strip()}\n</rubric>"]
    if task:
        parts.append(f"<task_given_to_the_assistant>\n{task.strip()}\n</task_given_to_the_assistant>")
    if reference:
        parts.append(f"<reference_answer>\n{reference.strip()}\n</reference_answer>")
    parts.append(f"<output_to_grade>\n{output.strip()}\n</output_to_grade>")
    parts.append("Think about whether the output meets the rubric, then give the JSON.")
    return "\n\n".join(parts)


@dataclass
class Verdict:
    score: float
    reasoning: str
    raw: str = ""


class JudgeError(Exception):
    pass


def parse_verdict(raw: str) -> Verdict:
    """Accept JSON (possibly fenced or surrounded by prose). Score may be 0–1, 0–10, 0–100 or a pass/fail flag."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise JudgeError(f"judge did not return JSON: {raw[:120]!r}")
    try:
        data: dict[str, Any] = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise JudgeError(f"judge returned invalid JSON: {m.group(0)[:120]!r}") from None
    score = data.get("score")
    if isinstance(score, bool) or score is None:
        if "pass" not in data and score is None:
            raise JudgeError("judge JSON has no score")
        score = 1.0 if (data.get("pass") if score is None else score) else 0.0
    try:
        s = float(score)
    except (TypeError, ValueError):
        raise JudgeError(f"judge score is not a number: {score!r}") from None
    if s > 1:
        s = s / 10 if s <= 10 else s / 100
    return Verdict(score=max(0.0, min(1.0, s)), reasoning=str(data.get("reasoning") or data.get("reason") or ""), raw=raw)


JudgeFn = Callable[[str, str], Awaitable[str]]  # (system, prompt) -> raw text


async def grade(judge: JudgeFn, rubric: str, output: str, task: str = "", reference: str = "") -> Verdict:
    return parse_verdict(await judge(JUDGE_SYSTEM, judge_prompt(rubric, output, task, reference)))


# ---------------------------------------------------------------- calibration


@dataclass
class Calibration:
    n: int
    accuracy: float
    precision: float
    recall: float
    kappa: float
    confusion: dict[str, int]  # tp, fp, fn, tn — "positive" means PASS
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """Landis & Koch bands for Cohen's kappa."""
        k = self.kappa
        if k >= 0.8:
            return "almost perfect agreement — the judge can be trusted for this rubric"
        if k >= 0.6:
            return "substantial agreement — usable, spot-check failures"
        if k >= 0.4:
            return "moderate agreement — tighten the rubric before relying on it"
        return "poor agreement — do not use this judge as a gate"


def agreement(human: list[bool], judge: list[bool]) -> tuple[float, float, float, float, dict[str, int]]:
    if len(human) != len(judge) or not human:
        raise ValueError("need the same, non-zero number of human and judge labels")
    tp = sum(h and j for h, j in zip(human, judge, strict=True))
    tn = sum((not h) and (not j) for h, j in zip(human, judge, strict=True))
    fp = sum((not h) and j for h, j in zip(human, judge, strict=True))
    fn = sum(h and (not j) for h, j in zip(human, judge, strict=True))
    n = len(human)
    po = (tp + tn) / n
    pe = ((tp + fn) / n) * ((tp + fp) / n) + ((tn + fp) / n) * ((tn + fn) / n)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return po, precision, recall, kappa, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _truthy(v: Any) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in ("1", "true", "pass", "yes", "y", "good")


async def calibrate(judge: JudgeFn, rows: list[dict[str, Any]], rubric: str, threshold: float = 0.5) -> Calibration:
    """rows need `output` and `human` (pass/fail), and may have `task` and `reference`."""
    human, judged, dis = [], [], []
    for i, r in enumerate(rows, 1):
        if "output" not in r or "human" not in r:
            raise ValueError(f"calibration row {i} needs 'output' and 'human' columns")
        v = await grade(judge, r.get("rubric") or rubric, str(r["output"]), str(r.get("task", "")), str(r.get("reference", "")))
        h, j = _truthy(r["human"]), v.score >= threshold
        human.append(h)
        judged.append(j)
        if h != j:
            dis.append({"row": i, "human": "pass" if h else "fail", "judge": "pass" if j else "fail", "score": v.score,
                        "reasoning": v.reasoning, "output": str(r["output"])[:200]})
    acc, p, rec, k, conf = agreement(human, judged)
    return Calibration(n=len(rows), accuracy=round(acc, 3), precision=round(p, 3), recall=round(rec, 3), kappa=round(k, 3),
                       confusion=conf, disagreements=dis)
