"""ThirtyAgent theme: purple + yellow on a deep violet-black.

Single source of truth so the TUI and the palette never drift apart.
"""

from __future__ import annotations

from pathlib import Path

from prompt_toolkit.styles import Style
from rich.theme import Theme

BRAND = "ThirtyAgent"

# core palette
PURPLE = "#a78bfa"
PURPLE_DEEP = "#8b5cf6"
PURPLE_DARK = "#6d28d9"
YELLOW = "#fcd34d"
YELLOW_DEEP = "#fbbf24"
BG = "#16121f"
FG = "#e6e1f0"
MUTED = "#8b839f"
DIM = "#5f566f"
BORDER = "#4c3d6b"

RICH_THEME = Theme(
    {
        "brand": f"bold {PURPLE}",
        "brand.dim": f"{PURPLE_DEEP}",
        "accent": f"bold {YELLOW}",
        "accent.dim": YELLOW_DEEP,
        "muted": MUTED,
        "dim": DIM,
        "text": FG,
        "border": BORDER,
        "model": f"bold {YELLOW}",
        "ok": "#5fd7af",
        "warn": YELLOW,
        "bad": "#ff6b81",
        "trace": PURPLE,
        "panel.border": BORDER,
        "prompt": f"bold {PURPLE}",
        "tool": YELLOW,
        "group": f"bold {PURPLE}",
        # tool cards: border + body tint by lifecycle, dim on success
        "card.pending": PURPLE,
        "card.running": YELLOW,
        "card.success": DIM,
        "card.error": "#ff6b81",
        "card.title": YELLOW,
        "card.ok": "#5fd7af",
        "card.divider": BORDER,
    }
)

# palette group accents -- purple family, yellow reserved for Cost & safety
GROUP_COLORS = {
    "Prompt": PURPLE,
    "Skills": "#c4b5fd",
    "Session": "#ddd6fe",
    "Model": YELLOW,
    "Capabilities": "#b8a3e8",
    "Work": "#9d8bd4",
    "Cost & safety": YELLOW_DEEP,
    "System": MUTED,
}


def group_class(group: str) -> str:
    """prompt_toolkit class for a group header.

    Class names cannot contain '#' or '&', so "Cost & safety" renders as
    "group.costsafety". Header tints have to live here rather than inline:
    render() used to emit e.g. "title #c4b5fd", which prompt_toolkit parses
    right-to-left and rejects with ValueError: Wrong color format 'title'.
    """
    return "group." + "".join(c for c in group.lower() if c.isalnum())


PTK_STYLE = Style.from_dict(
    {
        "base": f"bg:{BG} {FG}",
        "title": f"bold {PURPLE}",
        "esc": MUTED,
        "item": FG,
        "sel": f"bg:{PURPLE} bg:{BG} bold",
        "hint": MUTED,
        "selhint": f"bg:{PURPLE} {BG}",
        "cursor": f"reverse {YELLOW}",
        "frame": BORDER,
        **{group_class(g): f"bold {c}" for g, c in GROUP_COLORS.items()},
    }
)


def config_dir() -> Path:
    """Where state lives. Prefers the new name, still reads the old one."""
    new = Path.home() / ".thirtyagent"
    old = Path.home() / ".minicode"
    if new.exists() or not old.exists():
        return new
    return old
