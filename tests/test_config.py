from __future__ import annotations

import pytest

from evalsentry.config import ConfigError, load_suite

MIN = """
providers: [{id: m, type: mock, default_output: hi}]
prompts: [{id: p, user: "Say {{ x }}"}]
tests: [{description: "Say hello", vars: {x: hello}}]
"""


def test_minimal_suite_loads_and_assigns_ids(write_suite):
    s = load_suite(write_suite(MIN))
    tests = s.all_tests()
    assert [t.id for t in tests] == ["say-hello"]
    assert s.prompt_text(s.prompts[0]) == "Say {{ x }}"


def test_duplicate_test_ids_get_suffixes(write_suite):
    s = load_suite(write_suite(MIN.replace('tests: [{description: "Say hello", vars: {x: hello}}]',
                                           "tests: [{description: a}, {description: a}, {}]")))
    assert [t.id for t in s.all_tests()] == ["a", "a-2", "case-3"]


def test_defaults_are_appended_to_every_test(write_suite):
    s = load_suite(write_suite(MIN + "defaults:\n  assert: [{type: length, max_words: 5}]\n"))
    assert [a.type for a in s.all_tests()[0].assert_] == ["length"]


@pytest.mark.parametrize("bad, message", [
    (MIN.replace("type: mock", "type: nope"), "providers.0.type"),
    (MIN.replace('user: "Say {{ x }}"', 'user: "a", file: b.txt'), "exactly one of `user` or `file`"),
    (MIN + "judge: missing\n", "judge"),
    (MIN.replace("providers: [{id: m, type: mock, default_output: hi}]",
                 "providers: [{id: m, type: mock}, {id: m, type: mock}]"), "duplicate ids: m"),
    (MIN.replace("providers: [{id: m, type: mock, default_output: hi}]", "providers: [{id: a, type: anthropic}]"), "needs: model"),
    (MIN.replace("providers: [{id: m, type: mock, default_output: hi}]", "providers: [{id: f, type: python, function: nocolon}]"),
     "module.path:function_name"),
    (MIN + "unknown_key: 1\n", "unknown_key"),
    ("- just\n- a list\n", "mapping"),
    ("providers: [\n", "not valid YAML"),
])
def test_invalid_suites_give_friendly_errors(write_suite, bad, message):
    with pytest.raises(ConfigError) as e:
        load_suite(write_suite(bad))
    assert message in str(e.value)


def test_no_tests_is_an_error(write_suite):
    s = load_suite(write_suite(MIN.replace('tests: [{description: "Say hello", vars: {x: hello}}]', "")))
    with pytest.raises(ConfigError, match="no tests"):
        s.all_tests()


def test_env_expansion_with_default(write_suite, monkeypatch):
    monkeypatch.setenv("MODEL_X", "from-env")
    s = load_suite(write_suite(MIN.replace("default_output: hi", "default_output: ${MODEL_X}, model: ${UNSET_VAR:-fallback}")))
    assert s.providers[0].default_output == "from-env"
    assert s.providers[0].model == "fallback"


def test_env_expansion_missing_variable(write_suite, monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET", raising=False)
    with pytest.raises(ConfigError, match="DEFINITELY_UNSET is not set"):
        load_suite(write_suite(MIN.replace("default_output: hi", "default_output: ${DEFINITELY_UNSET}")))


def test_dataset_rows_become_tests(write_suite, tmp_path):
    (tmp_path / "rows.csv").write_text("id,x,tags\nfirst,a,\"smoke, fast\"\n,b,\n", encoding="utf-8")
    (tmp_path / "rows.jsonl").write_text('{"x": "c"}\n\n{"x": "d"}\n', encoding="utf-8")
    s = load_suite(write_suite(MIN + "dataset: rows.csv\n"))
    tests = s.all_tests()
    assert [t.id for t in tests] == ["say-hello", "first", "case-3"]
    assert tests[1].tags == ["smoke", "fast"] and tests[1].vars["x"] == "a"
    s2 = load_suite(write_suite(MIN + "dataset: rows.jsonl\n"))
    assert [t.vars["x"] for t in s2.all_tests()[1:]] == ["c", "d"]


def test_dataset_errors(write_suite, tmp_path):
    (tmp_path / "bad.jsonl").write_text('{"x": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="line 2"):
        load_suite(write_suite(MIN + "dataset: bad.jsonl\n")).all_tests()
    with pytest.raises(ConfigError, match="not found"):
        load_suite(write_suite(MIN + "dataset: nope.csv\n")).all_tests()


def test_prompt_file_relative_to_suite(write_suite, tmp_path):
    (tmp_path / "p.txt").write_text("From file {{ x }}", encoding="utf-8")
    s = load_suite(write_suite(MIN.replace('user: "Say {{ x }}"', "file: p.txt")))
    assert s.prompt_text(s.prompts[0]) == "From file {{ x }}"
    s2 = load_suite(write_suite(MIN.replace('user: "Say {{ x }}"', "file: missing.txt")))
    with pytest.raises(ConfigError, match="cannot read missing.txt"):
        s2.prompt_text(s2.prompts[0])


def test_all_examples_load():
    from conftest import EXAMPLES

    for path in EXAMPLES.glob("*/evalsentry.yaml"):
        assert load_suite(path).all_tests(), path
