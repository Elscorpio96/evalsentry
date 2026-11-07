"""Sandboxed Jinja2 rendering for prompts and assertion values."""

from __future__ import annotations

from typing import Any

from jinja2 import StrictUndefined, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True)


class TemplateError(Exception):
    pass


def render(template: str, vars: dict[str, Any]) -> str:
    if "{{" not in template and "{%" not in template:
        return template
    try:
        return _env.from_string(template).render(**vars)
    except UndefinedError as e:
        raise TemplateError(f"{e.message}. Available variables: {', '.join(sorted(vars)) or 'none'}") from None
    except Exception as e:
        raise TemplateError(str(e)) from None


def render_value(value: Any, vars: dict[str, Any]) -> Any:
    """Render strings anywhere inside lists/dicts."""
    if isinstance(value, str):
        return render(value, vars)
    if isinstance(value, list):
        return [render_value(v, vars) for v in value]
    if isinstance(value, dict):
        return {k: render_value(v, vars) for k, v in value.items()}
    return value
