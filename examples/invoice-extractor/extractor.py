"""A small rule-based invoice field extractor — the "system under test" for this example.

It stands in for whatever you build: a RAG chain, an agent, a fine-tuned model behind an API. EvalSentry calls
`extract(prompt, vars)` through the `python` provider and grades the JSON it returns.
"""

from __future__ import annotations

import re
from datetime import datetime

CURRENCIES = {"₦": "NGN", "NGN": "NGN", "N": "NGN", "$": "USD", "USD": "USD", "£": "GBP", "GBP": "GBP", "€": "EUR", "EUR": "EUR",
              "GH₵": "GHS", "GHS": "GHS", "KES": "KES"}
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")


def _date(text: str) -> str | None:
    m = re.search(r"(?:invoice date|date of issue|issued|date)\s*[:\-]?\s*([0-9A-Za-z ,/\-]{6,20}?\d{2,4})\b", text, re.I)
    if not m:
        return None
    raw = m.group(1).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _total(text: str) -> tuple[float | None, str | None]:
    # Prefer the line that says total due / amount due / grand total; fall back to "total".
    for label in (r"(?:total due|amount due|grand total|balance due)", r"total"):
        for m in re.finditer(label + r"\s*[:\-]?\s*(GH₵|[₦$£€]|NGN|USD|GBP|EUR|GHS|KES)?\s*([0-9][0-9,]*(?:\.\d{1,2})?)\s*(NGN|USD|GBP|EUR|GHS|KES)?",
                             text, re.I):
            if re.search(r"sub\s*-?total", text[max(0, m.start() - 4): m.start() + 9], re.I):
                continue  # skip "Subtotal"
            cur = m.group(1) or m.group(3)
            return float(m.group(2).replace(",", "")), CURRENCIES.get((cur or "").upper() if cur and cur.isalpha() else cur or "")
    return None, None


def extract(prompt: str, vars: dict) -> dict:
    text = vars.get("invoice", prompt)
    number = re.search(r"invoice\s*(?:no\.?|number|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,})", text, re.I)
    vendor = re.search(r"^\s*(?:from|vendor|supplier|billed by)\s*[:\-]\s*(.+)$", text, re.I | re.M)
    total, currency = _total(text)
    out = {
        "invoice_number": number.group(1) if number else None,
        "vendor": vendor.group(1).strip() if vendor else None,
        "date": _date(text),
        "total": total,
        "currency": currency,
    }
    return {"output": out, "input_tokens": len(text) // 4, "output_tokens": 30}
