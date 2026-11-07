"""Custom assertion: the extracted total must equal the expected total to the cent."""

import json


def total_matches(output: str, vars: dict) -> tuple[bool, str]:
    got = json.loads(output).get("total")
    want = float(vars["total"])
    if got is None:
        return False, "no total extracted"
    return abs(float(got) - want) < 0.005, f"total {got} (expected {want})"
