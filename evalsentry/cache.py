"""On-disk response cache: re-running a suite after editing assertions costs nothing and is deterministic."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .providers import Completion


class Cache:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.dir = directory
        self.enabled = enabled
        self.hits = 0

    @staticmethod
    def key(provider: dict[str, Any], messages: list[dict[str, str]], repeat: int) -> str:
        # The repeat index is part of the key so `repeat: 3` really samples three times.
        relevant = {k: provider.get(k) for k in ("type", "model", "base_url", "temperature", "max_tokens", "function", "url", "body",
                                                  "responses", "default_output")}
        blob = json.dumps({"p": relevant, "m": messages, "r": repeat}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Completion | None:
        if not self.enabled:
            return None
        f = self.dir / key[:2] / f"{key}.json"
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        self.hits += 1
        return Completion(**{**data, "cached": True})

    def put(self, key: str, c: Completion) -> None:
        if not self.enabled or c.error:
            return
        f = self.dir / key[:2] / f"{key}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps({**asdict(c), "cached": False}), encoding="utf-8")
        tmp.replace(f)
