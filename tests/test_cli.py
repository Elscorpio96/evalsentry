from __future__ import annotations

import json

from conftest import EXAMPLES

from evalsentry.cli import main

SUPPORT = str(EXAMPLES / "support-bot" / "evalsentry.yaml")


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as e:
        assert e.code == 0
    assert "evalsentry" in capsys.readouterr().out


def test_init_creates_a_runnable_suite(tmp_path):
    assert main(["init", str(tmp_path)]) == 0
    assert main(["init", str(tmp_path)]) == 2  # refuses to overwrite
    cfg = str(tmp_path / "evalsentry.yaml")
    # The starter suite deliberately has a case the mock cannot answer (Kenya), so the gate fails.
    assert main(["run", "-c", cfg, "--no-cache", "-q"]) == 1


def test_run_writes_reports_and_github_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    out, page, md = tmp_path / "r.json", tmp_path / "r.html", tmp_path / "r.md"
    code = main(["run", "-c", SUPPORT, "--prompt", "v2", "--no-cache", "-q", "-o", str(out), "--html", str(page), "--md", str(md),
                 "--github"])
    assert code == 0
    data = json.loads(out.read_text())
    assert data["suite"] == "support-bot" and len(data["results"]) == 6
    assert "<!doctype html>" in page.read_text() and "## ✅ EvalSentry" in md.read_text()
    assert summary.read_text() == md.read_text() + "\n"


def test_prompt_regression_fails_the_gate(tmp_path):
    base = tmp_path / "v2.json"
    assert main(["run", "-c", SUPPORT, "--prompt", "v2", "--no-cache", "-q", "-o", str(base)]) == 0
    # v3 still passes 5/6 (above min_pass_rate 0.8) but breaks the JSON contract: max_regressions catches it.
    assert main(["run", "-c", SUPPORT, "--prompt", "v3", "--no-cache", "-q", "--baseline", str(base), "-o", str(tmp_path / "v3.json")]) == 1
    assert main(["compare", str(base), str(tmp_path / "v3.json")]) == 1
    assert main(["compare", str(base), str(tmp_path / "v3.json"), "--allow", "1"]) == 0


def test_missing_baseline_is_a_warning_not_an_error(tmp_path):
    assert main(["run", "-c", SUPPORT, "--prompt", "v2", "--no-cache", "-q", "--baseline", str(tmp_path / "none.json")]) == 0


def test_usage_errors_exit_2(tmp_path, capsys):
    assert main(["run", "-c", str(tmp_path / "missing.yaml")]) == 2
    assert main(["run", "-c", SUPPORT, "--prompt", "nope", "--no-cache", "-q"]) == 2
    assert "unknown prompt" in capsys.readouterr().err


def test_calibrate_example(tmp_path):
    data = str(EXAMPLES / "support-bot" / "judge-labels.csv")
    rubric = "Applies Mkt's 30-day refund policy correctly and never asks for card details."
    assert main(["calibrate", "-c", SUPPORT, "--data", data, "--rubric", rubric, "--no-cache"]) == 0
    assert main(["calibrate", "-c", SUPPORT, "--data", data, "--rubric", rubric, "--no-cache", "--min-kappa", "0.9"]) == 1
    assert main(["calibrate", "-c", SUPPORT, "--data", data, "--no-cache"]) == 2  # no rubric
