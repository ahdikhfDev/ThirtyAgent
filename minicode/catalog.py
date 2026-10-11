"""The palette's contents: every command and every skill, grouped."""

from __future__ import annotations

from pathlib import Path

from .palette import Entry

# (text, label, group, shortcut)
COMMANDS: list[tuple[str, str, str, str]] = [
    ("/skills", "Skills (pick one)", "Prompt", ""),
    ("/skill:", "/skill:<name>  use a named skill", "Prompt", ""),
    ("/todos", "Stash prompt", "Prompt", ""),
    ("/resume", "Switch session", "Session", ""),
    ("/sessions", "Browse sessions", "Session", ""),
    ("/new", "New session", "Session", ""),
    ("/clear", "Clear conversation", "Session", ""),
    ("/rename", "Rename session", "Session", ""),
    ("/export", "Export session", "Session", ""),
    ("/model", "Switch model", "Model", ""),
    ("/mcp", "Plugins & MCP servers", "Capabilities", ""),
    ("/memory", "Loaded context files", "Capabilities", ""),
    ("/plan", "Plan mode (read-only)", "Work", ""),
    ("/usage", "Token usage this session", "Cost & safety", ""),
    ("/budget", "Cap model calls per turn", "Cost & safety", ""),
    ("/pii", "Redact PII before sending", "Cost & safety", ""),
    ("/rubric", "Grade the agent against a rubric", "Cost & safety", ""),
    ("/trace", "LangSmith tracing", "Cost & safety", ""),
    ("/retry", "Re-run the last turn", "Cost & safety", ""),
    ("/help", "Help", "System", ""),
    ("/exit", "Exit", "System", "ctrl+d"),
]


def _discover(root: Path) -> list:
    """A skill-scan failure must not stop the palette from opening."""
    from .skills import discover

    try:
        return discover(root)
    except (OSError, ValueError):
        return []


def skill_entries(root: Path) -> list[Entry]:
    """Skills only -- what the /skills picker shows."""
    return [
        Entry(
            text=f"/skill:{s.name}",
            label=s.name,
            group="Skills",
            shortcut=s.scope,
            detail=" ".join((s.description or "").split()),
        )
        for s in _discover(root)
    ]


def entries(root: Path) -> list[Entry]:
    """Commands plus one entry per discovered skill."""
    out = [Entry(text=t, label=l, group=g, shortcut=s) for t, l, g, s in COMMANDS]
    out.extend(
        Entry(
            text=f"/skill:{s.name}",
            label=f"/skill:{s.name}",
            group="Skills",
            shortcut=s.scope,
            detail=" ".join((s.description or "").split()),
        )
        for s in _discover(root)
    )
    return out


def commands_only() -> list[Entry]:
    return [Entry(text=t, label=l, group=g, shortcut=s) for t, l, g, s in COMMANDS]
