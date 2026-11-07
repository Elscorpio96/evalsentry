from __future__ import annotations

import httpx
import pytest
import respx

from evalsentry import providers as P
from evalsentry.config import ProviderConfig
from evalsentry.providers import ProviderError, make_provider

MSGS = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(P.asyncio, "sleep", fake_sleep)
    return slept


def cfg(**kw) -> ProviderConfig:
    return ProviderConfig(**{"id": "x", **kw})


@respx.mock
async def test_anthropic_request_shape_usage_and_cost(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    route = respx.post("https://api.anthropic.com/v1/messages").respond(json={
        "model": "claude-test", "content": [{"type": "text", "text": "Hello"}], "usage": {"input_tokens": 1000, "output_tokens": 200}})
    async with httpx.AsyncClient() as c:
        p = make_provider(cfg(type="anthropic", model="claude-test", api_key_env="TEST_KEY", price={"input": 3, "output": 15}), c)
        out = await p.complete(MSGS, {})
    req = route.calls[0].request
    body = __import__("json").loads(req.content)
    assert req.headers["x-api-key"] == "sk-secret" and req.headers["anthropic-version"]
    assert body["system"] == "Be brief." and body["messages"] == [{"role": "user", "content": "Hi"}]
    assert out.text == "Hello" and out.input_tokens == 1000 and out.output_tokens == 200
    assert out.cost_usd == pytest.approx((1000 * 3 + 200 * 15) / 1e6)


@respx.mock
async def test_openai_compatible_base_url_and_no_key():
    route = respx.post("http://localhost:11434/v1/chat/completions").respond(json={
        "choices": [{"message": {"content": "Hey"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    async with httpx.AsyncClient() as c:
        out = await make_provider(cfg(type="openai", model="llama", base_url="http://localhost:11434/v1/"), c).complete(MSGS, {})
    assert out.text == "Hey" and out.input_tokens == 5 and out.cost_usd is None
    assert "authorization" not in route.calls[0].request.headers


@respx.mock
async def test_retries_on_429_with_retry_after_then_succeeds(no_sleep):
    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=[
        httpx.Response(429, headers={"retry-after": "2"}, json={}),
        httpx.Response(503, json={}),
        httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]}),
    ])
    async with httpx.AsyncClient() as c:
        out = await make_provider(cfg(type="openai", model="m"), c).complete(MSGS, {})
    assert out.text == "ok"
    assert no_sleep[0] == 2.0 and len(no_sleep) == 2


@respx.mock
async def test_gives_up_after_max_retries():
    respx.post("https://api.openai.com/v1/chat/completions").respond(500, text="boom")
    async with httpx.AsyncClient() as c:
        with pytest.raises(ProviderError, match="gave up after 5 attempts"):
            await make_provider(cfg(type="openai", model="m"), c).complete(MSGS, {})


@respx.mock
async def test_client_errors_are_not_retried_and_key_is_redacted(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-very-secret")
    route = respx.post("https://api.openai.com/v1/chat/completions").respond(401, text="invalid key sk-very-secret")
    async with httpx.AsyncClient() as c:
        with pytest.raises(ProviderError) as e:
            await make_provider(cfg(type="openai", model="m", api_key_env="TEST_KEY"), c).complete(MSGS, {})
    assert route.call_count == 1
    assert "sk-very-secret" not in str(e.value) and "***" in str(e.value)


async def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    async with httpx.AsyncClient() as c:
        with pytest.raises(ProviderError, match="NOPE_KEY is not set"):
            await make_provider(cfg(type="anthropic", model="m", api_key_env="NOPE_KEY"), c).complete(MSGS, {})


@respx.mock
async def test_http_provider_body_template_and_response_path(monkeypatch):
    monkeypatch.setenv("SVC_TOKEN", "t0k")
    route = respx.post("https://rag.example.test/ask").respond(json={"data": {"answer": "42", "sources": [1]}})
    p = cfg(type="http", url="https://rag.example.test/ask", body={"q": "{{ prompt }}", "user": "{{ uid }}"},
            headers={"authorization": "Bearer ${SVC_TOKEN}"}, response_path="data.answer")
    async with httpx.AsyncClient() as c:
        out = await make_provider(p, c).complete(MSGS, {"uid": "u1"})
    assert out.text == "42"
    req = route.calls[0].request
    assert __import__("json").loads(req.content) == {"q": "Hi", "user": "u1"} and req.headers["authorization"] == "Bearer t0k"
    with pytest.raises(ProviderError, match="not found"):
        P._dig({"a": 1}, "a.b")


async def test_python_provider(tmp_path, monkeypatch):
    (tmp_path / "myapp.py").write_text(
        "def answer(prompt, vars): return prompt.upper()\n"
        "async def structured(prompt, vars): return {'output': {'k': vars['k']}, 'input_tokens': 3, 'cost_usd': 0.5}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    out = await make_provider(cfg(type="python", function="myapp:answer"), None).complete(MSGS, {})
    assert out.text == "HI"
    out = await make_provider(cfg(type="python", function="myapp:structured"), None).complete(MSGS, {"k": 1})
    assert out.text == '{"k": 1}' and out.input_tokens == 3 and out.cost_usd == 0.5
    with pytest.raises(ProviderError, match="cannot load"):
        make_provider(cfg(type="python", function="myapp:missing"), None)


async def test_mock_provider_rules_and_default():
    p = make_provider(cfg(type="mock", responses=[{"match": "capital.*nigeria", "output": "Abuja"}], default_output="?"), None)
    assert (await p.complete([{"role": "user", "content": "The capital of\nNigeria?"}], {})).text == "Abuja"
    assert (await p.complete([{"role": "user", "content": "other"}], {})).text == "?"
