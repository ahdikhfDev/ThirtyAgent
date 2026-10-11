"""Token accounting, turn budgets, and tracing wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Usage:
    """Rolls up token usage and model calls for a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    turns: int = 0
    by_model: dict = field(default_factory=dict)
    turn_start: dict = field(default_factory=dict)

    def record_message(self, usage_metadata: dict | None, model: str | None = None) -> None:
        if not usage_metadata:
            return
        self.calls += 1
        self.turn_start.setdefault("in", 0)
        self.turn_start.setdefault("out", 0)
        self.turn_start["in"] += int(usage_metadata.get("input_tokens") or 0)
        self.turn_start["out"] += int(usage_metadata.get("output_tokens") or 0)

        if model:
            bucket = self.by_model.setdefault(model, {"calls": 0, "in": 0, "out": 0})
            bucket["calls"] += 1
            bucket["in"] += int(usage_metadata.get("input_tokens") or 0)
            bucket["out"] += int(usage_metadata.get("output_tokens") or 0)

    def end_turn(self) -> dict:
        """Fold the current turn into the totals and return its slice."""
        turn = {
            "in": self.turn_start.get("in", 0),
            "out": self.turn_start.get("out", 0),
            "calls": self.calls,
        }
        self.input_tokens += turn["in"]
        self.output_tokens += turn["out"]
        self.turns += 1
        self.turn_start.clear()
        return turn

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def reset(self) -> None:
        self.__init__()


def tracing_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2"))


def tracing_status() -> str | None:
    """A short badge for the TUI, or None when tracing is off."""
    if not tracing_enabled():
        return None
    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")
    return f"trace: {project}" if project else "trace: on"


def enable_tracing() -> str:
    """Turn tracing on for this process if a key is available.

    ponytail: relies on LANGSMITH_API_KEY / LANGSMITH_TRACING already being in
    the environment. Never prompts for a key and never writes one to disk.
    """
    if not os.environ.get("LANGSMITH_API_KEY") and not os.environ.get("LANGCHAIN_API_KEY"):
        return "no LANGSMITH_API_KEY set"

    project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT")
    if not project:
        return "no LANGSMITH_PROJECT set"

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    return project


def budget_middleware(max_calls: int | None):
    """Cap model calls per turn so a runaway loop cannot drain the account."""
    if not max_calls or max_calls < 1:
        return None
    from langchain.agents.middleware import ModelCallLimitMiddleware

    return ModelCallLimitMiddleware(run_limit=max_calls, exit_behavior="end")


def pii_middleware(kinds: tuple[str, ...] = ("email", "credit_card", "ip"), strategy: str = "redact"):
    """Strip secrets before anything leaves the machine."""
    from langchain.agents.middleware import PIIMiddleware

    return [PIIMiddleware(kind, strategy=strategy) for kind in kinds]
