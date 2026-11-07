import pytest

from evalsentry.judge import JudgeError, agreement, calibrate, judge_prompt, parse_verdict


@pytest.mark.parametrize("raw, score", [
    ('{"reasoning": "ok", "score": 0.75}', 0.75),
    ('```json\n{"reasoning": "ok", "score": 7}\n```', 0.7),
    ('Here you go: {"score": 85, "reason": "good"} thanks', 0.85),
    ('{"pass": true}', 1.0),
    ('{"score": false}', 0.0),
    ('{"score": -3}', 0.0),
])
def test_parse_verdict(raw, score):
    assert parse_verdict(raw).score == pytest.approx(score)


@pytest.mark.parametrize("raw", ["no json here", "{not json}", '{"reasoning": "x"}', '{"score": "high"}'])
def test_parse_verdict_errors(raw):
    with pytest.raises(JudgeError):
        parse_verdict(raw)


def test_judge_prompt_sections():
    p = judge_prompt("R", "O", task="T", reference="REF")
    assert p.index("<rubric>") < p.index("<task_given") < p.index("<reference_answer>") < p.index("<output_to_grade>")
    assert "<task_given" not in judge_prompt("R", "O")


def test_agreement_matches_hand_computed_kappa():
    human = [True, True, True, True, False, False, False, False, True, False]
    judge = [True, True, True, False, False, False, False, True, True, False]
    acc, p, r, k, conf = agreement(human, judge)
    assert conf == {"tp": 4, "fp": 1, "fn": 1, "tn": 4}
    assert acc == 0.8 and p == 0.8 and r == 0.8
    assert k == pytest.approx(0.6)  # po 0.8, pe 0.5


def test_agreement_edge_cases():
    assert agreement([True, True], [True, True])[3] == 1.0
    with pytest.raises(ValueError):
        agreement([], [])


async def test_calibrate_reports_disagreements():
    async def judge(system, prompt):
        return '{"score": 0.9, "reasoning": "looks right"}' if "GOOD" in prompt else '{"score": 0.1, "reasoning": "bad"}'

    rows = [{"output": "GOOD answer", "human": "pass"}, {"output": "bad answer", "human": "fail"},
            {"output": "GOOD but wrong", "human": "fail"}, {"output": "fine", "human": "yes"}]
    cal = await calibrate(judge, rows, "rubric")
    assert cal.n == 4 and cal.accuracy == 0.5
    assert [d["row"] for d in cal.disagreements] == [3, 4]
    assert "agreement" in cal.verdict
    with pytest.raises(ValueError, match="row 1"):
        await calibrate(judge, [{"output": "x"}], "r")
