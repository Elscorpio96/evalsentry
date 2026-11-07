"""Command line: evalsentry init | run | compare | calibrate."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from importlib import resources
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .cache import Cache
from .compare import Comparison, compare, evaluate_gate
from .config import ConfigError, load_rows, load_suite
from .judge import Calibration, calibrate
from .providers import Provider, ProviderError, make_provider
from .reports import html, markdown
from .results import RunResult
from .runner import run_suite
from .templating import TemplateError

console = Console(stderr=True)
EXIT_OK, EXIT_GATE, EXIT_USAGE = 0, 1, 2


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _summary(run: RunResult) -> Table:
    t = Table(title=f"{run.suite} — pass rate {_pct(run.pass_rate)}", title_justify="left", header_style="bold")
    for col, j in (("Prompt", "left"), ("Provider", "left"), ("Passed", "right"), ("Pass rate", "right"), ("Score", "right"),
                   ("p95 ms", "right"), ("Cost", "right"), ("Flaky", "left")):
        t.add_column(col, justify=j)  # type: ignore[arg-type]
    for g in run.groups():
        color = "green" if g.pass_rate == 1 else "yellow" if g.pass_rate >= 0.8 else "red"
        t.add_row(g.prompt_id, g.provider_id, f"{g.passed}/{g.cases}", f"[{color}]{_pct(g.pass_rate)}[/]", f"{g.mean_score:.2f}",
                  f"{g.p95_latency_ms:.0f}", "—" if g.cost_usd is None else f"${g.cost_usd:.4f}", ", ".join(g.flaky))
    return t


def _failures(run: RunResult, limit: int = 10) -> None:
    fails = [r for r in run.results if not r.passed]
    for r in fails[:limit]:
        why = r.error or "; ".join(a["reason"] for a in r.assertions if not a["passed"])
        console.print(f"  [red]✕[/] [bold]{r.test_id}[/] [dim]{r.prompt_id} · {r.provider_id}[/] — {why[:200]}")
    if len(fails) > limit:
        console.print(f"  [dim]… and {len(fails) - limit} more (see the HTML report)[/]")


def _ids(v: str | None) -> list[str] | None:
    return [x.strip() for x in v.split(",") if x.strip()] if v else None


def _suite_name(config: str) -> str:
    """examples/support-bot/evalsentry.yaml -> support-bot; nightly.yaml -> nightly."""
    p = Path(config).resolve()
    return p.parent.name if p.stem == "evalsentry" else p.stem


def cmd_run(a: argparse.Namespace) -> int:
    suite = load_suite(a.config)
    cache = Cache(Path(a.cache_dir), enabled=not a.no_cache)
    with Progress(TextColumn("[bold]Evaluating[/]"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console,
                  transient=True, disable=a.quiet) as prog:
        task = prog.add_task("run", total=None)
        run = asyncio.run(run_suite(suite, cache=cache, only=a.only, tag=a.tag, concurrency=a.concurrency, name=a.name or _suite_name(a.config),
                                    prompts=_ids(a.prompt), providers_only=_ids(a.provider),
                                    progress=lambda d, n: prog.update(task, completed=d, total=n)))
    comparison: Comparison | None = None
    if a.baseline:
        if Path(a.baseline).exists():
            comparison = compare(RunResult.load(a.baseline), run)
        else:
            console.print(f"[yellow]Baseline {a.baseline} not found — skipping regression check.[/]")
    gate = evaluate_gate(suite.gate, run, comparison)

    if not a.quiet:
        console.print(_summary(run))
        if cache.hits:
            console.print(f"[dim]{cache.hits} response(s) served from cache ({a.cache_dir}); use --no-cache to call providers again.[/]")
        _failures(run)
        if comparison:
            console.print(f"Baseline: {_pct(comparison.pass_rate_before)} → {_pct(comparison.pass_rate_after)}; "
                          f"[red]{len(comparison.regressions)} regression(s)[/], [green]{len(comparison.fixes)} fixed[/]")
            for c in comparison.regressions:
                console.print(f"  [red]↓[/] {c.test_id} [dim]{c.prompt_id} · {c.provider_id}[/]")
        for name, ok, detail in gate.checks:
            console.print(f"{'[green]✓' if ok else '[red]✕'}[/] gate {name}: {detail}")
        console.print("[bold green]Quality gate passed[/]" if gate.passed else "[bold red]Quality gate failed[/]")

    if a.out:
        run.save(a.out)
    if a.html:
        Path(a.html).write_text(html(run, gate, comparison), encoding="utf-8")
        if not a.quiet:
            console.print(f"[dim]HTML report: {a.html}[/]")
    md = markdown(run, gate, comparison)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    if a.github and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md + "\n")
    return EXIT_OK if gate.passed else EXIT_GATE


def cmd_compare(a: argparse.Namespace) -> int:
    c = compare(RunResult.load(a.baseline), RunResult.load(a.current))
    print(f"Pass rate: {_pct(c.pass_rate_before)} → {_pct(c.pass_rate_after)}")
    for label, items in (("Regressions", c.regressions), ("Fixed", c.fixes), ("Score dropped (still passing)", c.score_drops)):
        if items:
            print(f"\n{label}:")
            for ch in items:
                print(f"  {ch.test_id}  [{ch.prompt_id} · {ch.provider_id}]  score {ch.before:.2f} → {ch.after:.2f}")
    if c.new_tests:
        print(f"\nNew: {len(c.new_tests)} case(s)")
    if c.removed_tests:
        print(f"Removed: {len(c.removed_tests)} case(s)")
    return EXIT_GATE if len(c.regressions) > a.allow else EXIT_OK


def cmd_calibrate(a: argparse.Namespace) -> int:
    suite = load_suite(a.config)
    if not suite.judge:
        raise ConfigError("set `judge:` in the suite to the provider you want to calibrate")
    rows = load_rows(Path(a.data))
    if not a.rubric and not all("rubric" in r for r in rows):
        raise ConfigError("give --rubric or a 'rubric' column in the data")
    cache = Cache(Path(a.cache_dir), enabled=not a.no_cache)

    async def go() -> Calibration:
        async with httpx.AsyncClient() as client:
            provider: Provider = make_provider(suite.provider(suite.judge), client)

            async def judge(system: str, prompt: str) -> str:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
                key = Cache.key(provider.cfg.model_dump(), msgs, 0)
                c = cache.get(key)
                if c is None:
                    c = await provider.complete(msgs, {})
                    cache.put(key, c)
                return c.text

            return await calibrate(judge, rows, a.rubric, a.threshold)

    cal = asyncio.run(go())
    t = Table(title=f"Judge calibration — {suite.judge} on {cal.n} human-labelled outputs", title_justify="left")
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    for k, v in (("Accuracy", f"{cal.accuracy:.1%}"), ("Precision (pass)", f"{cal.precision:.1%}"), ("Recall (pass)", f"{cal.recall:.1%}"),
                 ("Cohen's kappa", f"{cal.kappa:.2f}")):
        t.add_row(k, v)
    console.print(t)
    cf = cal.confusion
    console.print(f"Confusion: judge pass & human pass {cf['tp']} · judge pass & human fail {cf['fp']} · "
                  f"judge fail & human pass {cf['fn']} · both fail {cf['tn']}")
    console.print(f"[bold]{cal.verdict}[/]")
    for d in cal.disagreements[:10]:
        console.print(f"  row {d['row']}: human {d['human']}, judge {d['judge']} ({d['score']:.2f}) — {d['reasoning'][:150]}")
    return EXIT_OK if cal.kappa >= a.min_kappa else EXIT_GATE


def cmd_init(a: argparse.Namespace) -> int:
    target = Path(a.dir)
    target.mkdir(parents=True, exist_ok=True)
    dest = target / "evalsentry.yaml"
    if dest.exists() and not a.force:
        console.print(f"[red]{dest} already exists[/] (use --force to overwrite)")
        return EXIT_USAGE
    dest.write_text(resources.files("evalsentry").joinpath("starter.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"Created {dest}. Try:  [bold]evalsentry run -c {dest} --html report.html[/]")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalsentry", description="Regression tests and CI quality gates for LLM apps.")
    p.add_argument("--version", action="version", version=f"evalsentry {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a suite and apply the quality gate")
    r.add_argument("-c", "--config", default="evalsentry.yaml")
    r.add_argument("-o", "--out", help="write results JSON (use it later as a --baseline)")
    r.add_argument("--html", help="write a self-contained HTML report")
    r.add_argument("--md", help="write a Markdown summary")
    r.add_argument("--github", action="store_true", help="append the summary to the GitHub Actions job summary")
    r.add_argument("--baseline", help="results JSON from a previous run, to detect regressions")
    r.add_argument("--only", default="", help="run tests whose id or description contains this text")
    r.add_argument("--tag", default="", help="run tests with this tag")
    r.add_argument("--prompt", help="only these prompt ids (comma-separated)")
    r.add_argument("--provider", help="only these provider ids (comma-separated)")
    r.add_argument("--concurrency", type=int)
    r.add_argument("--name", default="", help="name shown in reports (default: config file name)")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--cache-dir", default=".evalsentry/cache")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="compare two results files")
    c.add_argument("baseline")
    c.add_argument("current")
    c.add_argument("--allow", type=int, default=0, help="number of regressions to tolerate")
    c.set_defaults(fn=cmd_compare)

    k = sub.add_parser("calibrate", help="measure how well the LLM judge agrees with human labels")
    k.add_argument("-c", "--config", default="evalsentry.yaml")
    k.add_argument("--data", required=True, help="CSV/JSONL with output, human (pass/fail), optional task, reference, rubric")
    k.add_argument("--rubric", default="")
    k.add_argument("--threshold", type=float, default=0.5)
    k.add_argument("--min-kappa", type=float, default=0.6, help="exit 1 if agreement is below this")
    k.add_argument("--no-cache", action="store_true")
    k.add_argument("--cache-dir", default=".evalsentry/cache")
    k.set_defaults(fn=cmd_calibrate)

    i = sub.add_parser("init", help="create a starter evalsentry.yaml")
    i.add_argument("dir", nargs="?", default=".")
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_init)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (ConfigError, TemplateError, ProviderError, ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]error:[/] {e}")
        return EXIT_USAGE
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
