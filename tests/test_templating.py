import pytest

from evalsentry.templating import TemplateError, render, render_value


def test_render_and_passthrough():
    assert render("Hi {{ name }}", {"name": "Ada"}) == "Hi Ada"
    assert render("no placeholders {", {}) == "no placeholders {"


def test_missing_variable_lists_available():
    with pytest.raises(TemplateError, match="Available variables: a, b"):
        render("{{ c }}", {"a": 1, "b": 2})


def test_sandbox_blocks_attribute_escape():
    with pytest.raises(TemplateError):
        render("{{ ''.__class__.__mro__[1].__subclasses__() }}", {})


def test_render_value_nested():
    assert render_value({"a": ["{{ x }}", 3], "b": {"c": "{{ x }}!"}}, {"x": "y"}) == {"a": ["y", 3], "b": {"c": "y!"}}
