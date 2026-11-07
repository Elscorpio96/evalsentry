"""Model providers. Each turns a list of chat messages into a Completion with usage, latency and cost."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import ProviderConfig
from .templating import render_value

Messages = list[dict[str, str]]


@dataclass
class Completion:
    text: str
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    model: str = ""
    cached: bool = False
    error: str = ""


class ProviderError(Exception):
    pass


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


class Provider:
    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None, max_retries: int = 4) -> None:
        self.cfg = cfg
        self.client = client
        self.max_retries = max_retries

    @property
    def model_name(self) -> str:
        return self.cfg.model or self.cfg.type

    async def complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        start = time.perf_counter()
        c = await self._complete(messages, vars)
        if not c.latency_ms:
            c.latency_ms = (time.perf_counter() - start) * 1000
        if self.cfg.price and c.cost_usd is None:
            c.cost_usd = (c.input_tokens * self.cfg.price.input + c.output_tokens * self.cfg.price.output) / 1_000_000
        c.model = c.model or self.model_name
        return c

    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        raise NotImplementedError

    def _key(self) -> str:
        if not self.cfg.api_key_env:
            return ""
        key = os.environ.get(self.cfg.api_key_env, "")
        if not key:
            raise ProviderError(f"provider '{self.cfg.id}': environment variable {self.cfg.api_key_env} is not set")
        return key

    async def _post(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        """POST with retries: exponential backoff with jitter, honouring Retry-After."""
        assert self.client is not None
        last = ""
        for attempt in range(self.max_retries + 1):
            try:
                r = await self.client.post(url, headers=headers, json=body, timeout=self.cfg.timeout_s)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = f"{type(e).__name__}"
            else:
                if r.status_code < 400:
                    try:
                        data: dict[str, Any] = r.json()
                    except ValueError:
                        raise ProviderError(f"provider '{self.cfg.id}': response was not JSON") from None
                    return data
                last = f"HTTP {r.status_code}: {_redact(r.text[:300], headers)}"
                if r.status_code not in RETRY_STATUS:
                    raise ProviderError(f"provider '{self.cfg.id}': {last}")
                retry_after = r.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():
                    await asyncio.sleep(min(float(retry_after), 60))
                    continue
            if attempt < self.max_retries:
                await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random() / 2))  # noqa: S311 - jitter, not security
        raise ProviderError(f"provider '{self.cfg.id}': gave up after {self.max_retries + 1} attempts ({last})")


def _redact(text: str, headers: dict[str, str]) -> str:
    for k, v in headers.items():
        if k.lower() in ("authorization", "x-api-key") and v:
            text = text.replace(v.removeprefix("Bearer "), "***")
    return text


class AnthropicProvider(Provider):
    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        body: dict[str, Any] = {"model": self.cfg.model, "max_tokens": self.cfg.max_tokens,
                                "messages": [m for m in messages if m["role"] != "system"]}
        if system:
            body["system"] = system
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        url = (self.cfg.base_url or "https://api.anthropic.com").rstrip("/") + "/v1/messages"
        data = await self._post(url, {"x-api-key": self._key(), "anthropic-version": "2023-06-01"}, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return Completion(text=text, input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0),
                          model=data.get("model", self.cfg.model))


class OpenAIProvider(Provider):
    """OpenAI and every compatible server: Ollama, vLLM, LM Studio, OpenRouter, Together, Groq..."""

    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        body: dict[str, Any] = {"model": self.cfg.model, "messages": messages, "max_tokens": self.cfg.max_tokens}
        if self.cfg.temperature is not None:
            body["temperature"] = self.cfg.temperature
        key = self._key()
        headers = {"authorization": f"Bearer {key}"} if key else {}
        url = (self.cfg.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        data = await self._post(url, headers, body)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise ProviderError(f"provider '{self.cfg.id}': unexpected response shape") from None
        usage = data.get("usage") or {}
        return Completion(text=text, input_tokens=usage.get("prompt_tokens", 0), output_tokens=usage.get("completion_tokens", 0),
                          model=data.get("model", self.cfg.model))


class HTTPProvider(Provider):
    """Test any HTTP endpoint — e.g. your own deployed RAG or agent API."""

    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        ctx = {**vars, "prompt": messages[-1]["content"], "messages": messages}
        body = render_value(self.cfg.body if self.cfg.body is not None else {"prompt": "{{ prompt }}"}, ctx)
        headers = {k: _expand(v) for k, v in self.cfg.headers.items()}
        data = await self._post(self.cfg.url, headers, body)
        return Completion(text=_dig(data, self.cfg.response_path) if self.cfg.response_path else json.dumps(data))


class PythonProvider(Provider):
    """Call your own function — an agent, a RAG pipeline, a parser. It receives (prompt, vars) and returns
    a string, or a dict with "output" and optional "input_tokens", "output_tokens", "cost_usd"."""

    def __init__(self, cfg: ProviderConfig, client: httpx.AsyncClient | None = None, max_retries: int = 0) -> None:
        super().__init__(cfg, client, max_retries)
        mod, _, name = cfg.function.partition(":")
        try:
            self.fn = getattr(importlib.import_module(mod), name)
        except (ImportError, AttributeError) as e:
            raise ProviderError(f"provider '{cfg.id}': cannot load {cfg.function} ({e})") from None

    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        prompt = messages[-1]["content"]
        result = self.fn(prompt, vars)
        if inspect.isawaitable(result):
            result = await result
        else:
            await asyncio.sleep(0)
        if isinstance(result, dict):
            out = result.get("output", "")
            return Completion(text=out if isinstance(out, str) else json.dumps(out), input_tokens=int(result.get("input_tokens", 0)),
                              output_tokens=int(result.get("output_tokens", 0)), cost_usd=result.get("cost_usd"))
        return Completion(text=result if isinstance(result, str) else json.dumps(result))


class MockProvider(Provider):
    """Deterministic, offline responses: for demos, tests, and developing a suite before paying for calls."""

    async def _complete(self, messages: Messages, vars: dict[str, Any]) -> Completion:
        prompt = "\n".join(m["content"] for m in messages)
        if self.cfg.latency_ms:
            await asyncio.sleep(self.cfg.latency_ms / 1000)
        out = next((r.output for r in self.cfg.responses if re.search(r.match, prompt, re.S | re.I)), self.cfg.default_output)
        return Completion(text=out, latency_ms=self.cfg.latency_ms or 0.0, input_tokens=max(1, len(prompt) // 4),
                          output_tokens=max(1, len(out) // 4))


def _expand(v: str) -> str:
    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), v)


def _dig(data: Any, path: str) -> str:
    cur = data
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise ProviderError(f"response_path '{path}' not found in the response")
    return cur if isinstance(cur, str) else json.dumps(cur)


PROVIDERS: dict[str, type[Provider]] = {
    "anthropic": AnthropicProvider, "openai": OpenAIProvider, "http": HTTPProvider, "python": PythonProvider, "mock": MockProvider,
}


def make_provider(cfg: ProviderConfig, client: httpx.AsyncClient | None) -> Provider:
    return PROVIDERS[cfg.type](cfg, client)
