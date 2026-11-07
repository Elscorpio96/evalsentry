"""Suite configuration: a YAML file describing providers, prompts, test cases and the quality gate."""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ConfigError(Exception):
    """The suite file is invalid. The message is written for the person who wrote the file."""


class Price(BaseModel):
    """USD per million tokens. Set it from your provider's price list; EvalSentry ships no prices."""

    model_config = ConfigDict(extra="forbid")
    input: float = Field(ge=0)
    output: float = Field(ge=0)


class MockRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match: str  # regex searched in the rendered prompt
    output: str


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$")
    type: Literal["anthropic", "openai", "mock", "python", "http"]
    model: str = ""
    base_url: str = ""
    api_key_env: str = ""  # name of the environment variable holding the key — keys never go in the file
    temperature: float | None = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=64000)
    timeout_s: float = Field(default=60, gt=0, le=600)
    price: Price | None = None
    # python provider
    function: str = ""  # "package.module:function"
    # http provider
    url: str = ""
    body: dict[str, Any] | None = None  # JSON body template; strings are rendered with {{ prompt }} and test vars
    headers: dict[str, str] = Field(default_factory=dict)
    response_path: str = ""  # dotted path to the text in the JSON response, e.g. "data.answer"
    # mock provider
    responses: list[MockRule] = Field(default_factory=list)
    default_output: str = ""
    latency_ms: float = 0

    @model_validator(mode="after")
    def _check(self) -> ProviderConfig:
        need = {"anthropic": ["model"], "openai": ["model"], "python": ["function"], "http": ["url"]}.get(self.type, [])
        missing = [f for f in need if not getattr(self, f)]
        if missing:
            raise ValueError(f"provider '{self.id}' of type {self.type} needs: {', '.join(missing)}")
        if self.type == "python" and ":" not in self.function:
            raise ValueError(f"provider '{self.id}': function must look like 'module.path:function_name'")
        return self


class PromptConfig(BaseModel):
    """A prompt template. Use {{ var }} placeholders filled from each test's vars."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$")
    system: str = ""
    user: str = ""
    file: str = ""  # alternative to `user`: path to a template file, relative to the suite file

    @model_validator(mode="after")
    def _one_source(self) -> PromptConfig:
        if bool(self.user) == bool(self.file):
            raise ValueError(f"prompt '{self.id}': give exactly one of `user` or `file`")
        return self


class AssertionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")  # each assertion type has its own options
    type: str
    weight: float = Field(default=1.0, gt=0)


class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    description: str = ""
    vars: dict[str, Any] = Field(default_factory=dict)
    assert_: list[AssertionConfig] = Field(default_factory=list, alias="assert")
    tags: list[str] = Field(default_factory=list)
    threshold: float | None = Field(default=None, ge=0, le=1)  # weighted score needed to pass; default: every assertion passes


class Gate(BaseModel):
    """CI quality gate. Any rule that fails makes `evalsentry run` exit with code 1."""

    model_config = ConfigDict(extra="forbid")
    min_pass_rate: float | None = Field(default=None, ge=0, le=1)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_p95_latency_ms: float | None = Field(default=None, gt=0)
    max_regressions: int | None = Field(default=None, ge=0)  # compared with --baseline


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    assert_: list[AssertionConfig] = Field(default_factory=list, alias="assert")
    repeat: int = Field(default=1, ge=1, le=20)


class Suite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    providers: list[ProviderConfig] = Field(min_length=1)
    prompts: list[PromptConfig] = Field(min_length=1)
    tests: list[TestCase] = Field(default_factory=list)
    dataset: str = ""  # CSV or JSONL; each row becomes a test (columns -> vars)
    defaults: Defaults = Field(default_factory=Defaults)
    judge: str = ""  # provider id used for llm_judge assertions
    gate: Gate = Field(default_factory=Gate)
    concurrency: int = Field(default=4, ge=1, le=64)
    base_dir: Path = Field(default=Path("."), exclude=True)

    @field_validator("providers", "prompts")
    @classmethod
    def _unique_ids(cls, v: list[Any]) -> list[Any]:
        ids = [x.id for x in v]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate ids: {', '.join(dupes)}")
        return v

    @model_validator(mode="after")
    def _judge_exists(self) -> Suite:
        if self.judge and self.judge not in {p.id for p in self.providers}:
            raise ValueError(f"judge '{self.judge}' is not one of the providers")
        return self

    def provider(self, pid: str) -> ProviderConfig:
        return next(p for p in self.providers if p.id == pid)

    def prompt_text(self, p: PromptConfig) -> str:
        if p.user:
            return p.user
        path = (self.base_dir / p.file).resolve()
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"prompt '{p.id}': cannot read {p.file}: {e.strerror}") from None

    def all_tests(self) -> list[TestCase]:
        """Inline tests plus dataset rows, each with a unique id and the default assertions appended."""
        tests = list(self.tests)
        if self.dataset:
            tests += [TestCase(id=str(row.get("id") or ""), description=str(row.get("description") or ""), tags=_tags(row.get("tags")), vars=row)
                      for row in load_rows(self.base_dir / self.dataset)]
        out, seen = [], set()
        for i, t in enumerate(tests, 1):
            tid = t.id or _slug(t.description) or f"case-{i}"
            base, n = tid, 2
            while tid in seen:
                tid, n = f"{base}-{n}", n + 1
            seen.add(tid)
            out.append(t.model_copy(update={"id": tid, "assert_": [*t.assert_, *self.defaults.assert_]}))
        if not out:
            raise ConfigError("the suite has no tests: add `tests:` or a `dataset:`")
        return out


def _tags(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    return [t.strip() for t in str(v or "").split(",") if t.strip()]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ConfigError(f"dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        rows = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ConfigError(f"{path.name} line {n}: invalid JSON ({e.msg})") from None
        return rows
    raise ConfigError(f"dataset must be .csv or .jsonl, got {path.name}")


_ENV = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(text: str) -> str:
    """${VAR} or ${VAR:-default} in the YAML, so URLs and models can come from the environment."""
    def sub(m: re.Match[str]) -> str:
        val = os.environ.get(m.group(1))
        if val is None:
            if m.group(2) is not None:
                return m.group(2)
            raise ConfigError(f"environment variable {m.group(1)} is not set")
        return val

    return _ENV.sub(sub, text)


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    try:
        raw = yaml.safe_load(_expand_env(path.read_text(encoding="utf-8")))
    except OSError as e:
        raise ConfigError(f"cannot read {path}: {e.strerror}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain a mapping at the top level")
    try:
        suite = Suite.model_validate({**raw, "base_dir": path.parent.resolve()})
    except ValueError as e:
        raise ConfigError(f"{path.name}: {_friendly(e)}") from None
    # `python` providers and assertions name "module:function"; resolve modules next to the suite file.
    if str(suite.base_dir) not in sys.path:
        sys.path.insert(0, str(suite.base_dir))
    return suite


def _friendly(e: ValueError) -> str:
    errs = getattr(e, "errors", None)
    if not callable(errs):
        return str(e)
    parts = []
    for err in errs()[:5]:
        loc = ".".join(str(x) for x in err["loc"] if x != "__root__")
        parts.append(f"{loc}: {err['msg'].removeprefix('Value error, ')}" if loc else err["msg"].removeprefix("Value error, "))
    return "; ".join(parts)
