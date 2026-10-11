from __future__ import annotations

import re
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

# name -> (one-line hint shown next to the suggestion)
COMMANDS: dict[str, str] = {
    "/help": "show every command",
    "/exit": "quit",
    "/new": "start a fresh session",
    "/sessions": "list saved sessions here",
    "/resume": "resume a session id",
    "/clear": "clear conversation, keep session",
    "/rename": "name this session",
    "/export": "write the session entry as JSON",
    "/plan": "toggle read-only plan mode",
    "/todos": "show the task list",
    "/memory": "show loaded context files",
    "/model": "show, list, or switch model",
    "/skills": "list discovered skills",
    "/mcp": "manage MCP servers",
    "/rubric": "self-grade against criteria",
}

# command -> (subcommands, which dynamic sources make sense after it)
SUBCOMMANDS: dict[str, dict[str, str]] = {
    "/model": {
        "list": "show all aliases",
        "add": "add alias, base_url, model, key",
        "rm": "remove an alias",
    },
    "/mcp": {
        "add": "add a server",
        "test": "connect once and report",
        "rm": "remove a server",
    },
}

# (command, kind) -> dynamic sources offered in its argument slot
ARG_SOURCES: dict[str, tuple[str, ...]] = {
    "/model": ("models",),
    "/mcp": ("mcp_servers",),
    "/resume": ("sessions",),
}

SOURCES = ("models", "mcp_servers", "sessions")

ALIASES = {"quit": "/exit", "q": "/exit", "skills": "/skill:", "model": "/model"}

_TOKEN = re.compile(r"(\S*)$")


class SlashCompleter(Completer):
    """Completes commands, /skill:<name>, and subcommands. Falls back to nothing
    on free text so normal typing is never interrupted."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path.cwd()

    # -- candidate sources -------------------------------------------------

    def skills(self) -> dict[str, str]:
        from .skills import discover

        try:
            return {s.name: s.description or "(no description)" for s in discover(self.root)}
        except Exception:  # noqa: BLE001 - completion must never crash the prompt
            return {}

    def models(self) -> dict[str, str]:
        from .config import discover_models

        try:
            return {
                a: f"{s.get('model', '?')}  {s.get('base_url', '')}" for a, s in discover_models().items()
            }
        except Exception:  # noqa: BLE001
            return {}

    def mcp_servers(self) -> dict[str, str]:
        from .mcp import connections

        try:
            return {n: s.get("url") or s.get("command", "") for n, s in connections().items()}
        except Exception:  # noqa: BLE001
            return {}

    def sessions(self) -> dict[str, str]:
        from .sessions import Store
        from .theme import config_dir

        try:
            rows = Store(config_dir() / "sessions.db").list(self.root, limit=10)
            return {r["id"]: (r["title"] or "(untitled)") for r in rows}
        except Exception:  # noqa: BLE001
            return {}

    # -- completion --------------------------------------------------------

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return

        m = _TOKEN.search(text)
        token = m.group(1) if m else ""
        # prompt_toolkit wants a relative offset: 0 inserts at the cursor.
        back = -len(token)

        # "/skill:<partial>" -- the case worth getting right.
        if token.startswith("/skill:") or (token == "/skill" and not text.endswith(":")):
            body = token[len("/skill:") :] if token.startswith("/skill:") else ""
            for name, desc in sorted(self.skills().items()):
                if name.startswith(body):
                    yield Completion(
                        f"/skill:{name}",
                        start_position=back,
                        display=f"/skill:{name}",
                        display_meta=(desc.splitlines()[0][:48] if desc else ""),
                    )
            return

        # bare command, possibly with a space already typed
        if " " in text and not text.endswith(" "):
            return

        head = text.strip()
        after_space = " " in text or text.endswith(" ")

        if after_space:
            for sub, hint in SUBCOMMANDS.get(head, {}).items():
                yield Completion(sub, start_position=0, display_meta=hint[:48])
            for kind in ARG_SOURCES.get(head, ()):
                for name, meta in sorted(getattr(self, kind)().items()):
                    if name.startswith(token.lstrip("/")):
                        yield Completion(name, start_position=back, display_meta=meta[:60])
            return

        # dynamic values are also valid first tokens (/model big, not just /model)
        for kind in SOURCES:
            for name, meta in sorted(getattr(self, kind)().items()):
                if name.startswith(token.lstrip("/")):
                    yield Completion(
                        name,
                        start_position=back,
                        display_meta=f"{kind.removesuffix('_servers').removesuffix('s')}  {meta}"[:60],
                    )

        for name, hint in sorted(COMMANDS.items()):
            if name.startswith(token):
                yield Completion(name, start_position=back, display_meta=hint[:48])
