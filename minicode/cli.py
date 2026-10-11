from __future__ import annotations

import logging
import re
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from rich.console import Console, Group, RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import build_agent, find_memory
from .redact import SecretRedactor
from .sessions import (
    Store,
    derive_title,
    leaf_name,
    new_session_id,
)
from .sessions import (
    shorten_paths as _shorten_paths,
)
from .theme import BRAND, RICH_THEME, config_dir
from .usage import Usage, enable_tracing, tracing_enabled, tracing_status

# Masks .env secret values from anything the agent prints or stores.
_redactor = SecretRedactor()


def use_utf8_console() -> None:
    """Windows consoles default to CP437, where box-drawing and block glyphs
    (─ ▏ ▸ ● …) do not exist and arrive as mojibake like 'a-<8f>'. Switch the
    code page to UTF-8 before a single character is written, and let Python
    replace anything genuinely unencodable rather than crash on it.
    """
    import sys

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001 - not a real console, or locked down
            return _reconfigure_streams(sys)
    _reconfigure_streams(sys)


def _reconfigure_streams(sys) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            continue


app = typer.Typer(add_completion=False)
console = Console(theme=RICH_THEME)


def sync_width() -> int:
    """Pin Rich's width to prompt_toolkit's, not the OS's.

    prompt_toolkit owns the live prompt and is the authority on how wide the
    screen is; Rich asks the OS separately and can disagree by a column or a
    lot. Panels rendered at the stale width leave borders that stop short of
    their own content when the window is resized. Re-reading on every render
    means Rich always agrees with the prompt it shares a line with.
    """
    try:
        from prompt_toolkit.application import get_app

        out = get_app().output
        size = out.get_size() if out is not None else None
        # set height too: Console.size only honours a pinned width when the
        # height is set as well, otherwise it silently re-reads the OS size.
        # legacy_windows is subtracted back out on read, so add it here or the
        # panels come up a column short.
        if size is not None and size.columns:
            console.width = size.columns + console.legacy_windows
            console.height = (size.rows or 24) + console.legacy_windows
    except Exception:  # noqa: BLE001,S110 - no running app, fall back to OS size
        pass
    return console.width


EXIT_WORDS = {"/exit", "/quit", "/q"}

# opencode-style folding: 3 lines folded, 12 expanded (omp's preview limits).
COLLAPSED_LINES = 3
EXPANDED_LINES = 12
_expand_all = False


def set_expand_all(on: bool) -> None:
    global _expand_all
    _expand_all = on


def is_expanded() -> bool:
    return _expand_all


def _fold(body: str) -> list[str]:
    """Peek at the end, which is where the answer is; drop the top with a marker."""
    lines = body.splitlines()
    if len(lines) <= (EXPANDED_LINES if _expand_all else COLLAPSED_LINES):
        return lines

    keep = EXPANDED_LINES if _expand_all else COLLAPSED_LINES
    hidden = len(lines) - keep
    tail = lines[-keep:] if not _expand_all else lines[:keep]
    return [f"… {hidden} earlier lines [Ctrl+O: Expand]", *tail]


HELP = """\
[bold]general[/bold]
  /help              this text

[bold]session[/bold]
  /new                start a fresh session
  /sessions           list saved sessions for this directory
  /resume [id]        resume a session (default: latest here)
  /rename <title>     name the current session
  /export <path>      write the session index entry as JSON
  /clear              clear conversation, keep the session

[bold]model[/bold]
  /model              show current model, provider, endpoint
  /model <alias>      switch model
  /model list         list configured aliases
  /model add <alias> <base_url> <model> [api_key]

[bold]capabilities[/bold]
  /skills             pick a skill (arrow keys) and use it
  /skills list        just list them
  /skill:<name>       load a skill for the next turn
  /mcp                list configured MCP servers and their tools
  /mcp add <name> -- <cmd...>    stdio server
  /mcp add <name> --url <url>    http server
  /mcp test <name>    connect once and report
  /mcp rm <name>      remove a server

[bold]cost & safety[/bold]
  /usage              tokens and model calls this session
  /budget <n|off>     cap model calls per turn
  /pii on|off         redact emails/cards/IPs before they leave the machine
  /rubric <criteria>  self-grade and iterate until it passes (/rubric off)
  /trace [on|off]     LangSmith tracing
  /retry              re-run the last turn

[bold]work[/bold]
  /plan [prompt]      plan mode (read-only) — no edits, no shell
  /todos              show the current task list
  /memory             show which context files are loaded
  /exit               quit

[bold]keys[/bold]
  ctrl+k              open the command palette (search, arrow keys, enter)
  tab / enter         accept the inline suggestion
  esc                 clear the current line
  ctrl+d              quit

[bold]the agent can[/bold]
  ls read_file write_file edit_file delete glob grep   files under --cwd
  execute                                             shell commands
  fetch_url                                           web pages
  task                                                 delegate to a subagent
"""

_EXIT_SUFFIX = re.compile(r"\[Command (?:succeeded|failed)[^\]]*\]")


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _summarize(name: str, args: dict) -> str:
    """One human line per tool. Raw kwargs dumps are unreadable.

    Paths are reduced to their leaf: a Windows profile path is ~55 chars of
    noise that used to wrap mid-word and push the whole card off the screen.
    """
    if name in {"read_file", "write_file", "edit_file", "delete"}:
        return leaf_name(args.get("file_path", "?")) or "?"
    if name == "execute":
        return " ".join(str(args.get("command", "?")).split())
    if name == "ls":
        return leaf_name(args.get("path", "/")) or "/"
    if name == "glob":
        return str(args.get("pattern", "?"))
    if name == "grep":
        return f"{str(args.get('pattern', '?')).strip()} in {leaf_name(args.get('path')) or '.'}"
    if name == "task":
        return str(args.get("description", "?"))
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in args.items())[:100]


_TOOL_KIND = {
    "read_file": "read",
    "ls": "read",
    "glob": "read",
    "grep": "read",
    "write_file": "write",
    "edit_file": "edit",
    "delete": "delete",
    "execute": "bash",
    "fetch_url": "fetch",
    "task": "agent",
}


_GUTTER = "  "


def _card_title(name: str, args: dict) -> Text:
    """Zen header: `▸ read  path/to/file`. No box, just a marker and a colour."""
    kind = _TOOL_KIND.get(name, name)
    title = Text()
    title.append(_GUTTER, style="card.border")
    title.append("▸ ", style="card.title")
    title.append(kind, style="card.title")
    title.append("  ", style="card.title")
    title.append(_redactor.text(_shorten_paths(_summarize(name, args))), style="muted")
    return title


def _tool_card(name: str, args: dict, result: Text | None, border: str) -> None:
    """Zen-style tool line: a coloured header, then indented output.

    No box. opencode/omp leave finished work flat and let colour carry the
    lifecycle, because a bordered panel per tool call turns a long run into a
    wall of rectangles.
    """
    lines: list[Text] = [_card_title(name, args)]
    if result is not None and result.plain.strip():
        for line in result.plain.splitlines():
            # keep output inside the text column, clear of the sidebar
            lines.append(Text(_GUTTER * 2 + line, style="muted", overflow="fold"))
    console.print(Group(*lines), width=text_column_width())


class _NullStatus:
    """Same surface as rich Status, prints nothing (used when stdout is not a tty)."""

    def start(self) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def stop(self) -> None:
        pass


def _entries(body: str) -> list[str]:
    """ls/grep return a flat list of paths; render a count, not the whole dump."""
    items = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(items) == 1 and items[0].startswith("["):
        import ast

        try:
            parsed = ast.literal_eval(items[0])
        except (ValueError, SyntaxError):
            parsed = items
        items = [str(x) for x in parsed]
    return items


def _tool_result_lines(msg) -> list[Text]:
    """Result body, folded unless ctrl+o is on."""
    body = _redactor.text(_text(msg.content)).strip()
    exit_code = (getattr(msg, "artifact", None) or {}).get("exit_code")
    failed = getattr(msg, "status", "") == "error"

    if exit_code is not None or failed:
        head = "failed" if failed else f"exit {exit_code}"
        tail = " ".join(_EXIT_SUFFIX.sub("", body).split())[:90]
        return [
            Text.assemble(
                (head, "bad" if failed or exit_code else "card.ok"), ("  ", "muted"), (tail, "muted")
            )
        ]

    if msg.name in {"ls", "grep"}:
        items = _entries(body)
        if not items:
            return [Text("nothing found", style="muted")]
        preview = ", ".join(Path(i).name or i for i in items[:4])
        more = f" +{len(items) - 4} more" if len(items) > 4 else ""
        return [Text.assemble((preview, "muted"), (more, "muted"))]

    if not body:
        return []

    return [
        Text(ln, style="muted italic" if ln.startswith("… ") else "muted")
        for ln in _fold(body)[: EXPANDED_LINES + 1]
    ]


def _state_for(msg) -> str:
    code = (getattr(msg, "artifact", None) or {}).get("exit_code")
    if getattr(msg, "status", "") == "error" or code:
        return "error"
    return "success"


def _drive(agent, config: dict, payload, usage=None) -> dict | None:
    """Run one graph step, rendering as it goes.

    ponytail: assistant text renders per completed message, not per token --
    per-token markdown re-parses and flickers on every chunk. Upgrade path:
    rich.live with a throttled Markdown renderable.
    """
    pending: dict | None = None
    buf = ""
    # tool_call_id -> (name, args) so a card can be printed when its result lands
    inflight: dict[str, tuple[str, dict]] = {}
    # rich's Status repaints on a timer, which leaks frames when stdout is piped.
    spinner = (
        console.status("[dim]thinking…[/dim]", spinner="dots", spinner_style="model")
        if console.is_terminal
        else _NullStatus()
    )
    spinner.start()

    def flush():
        nonlocal buf
        if buf.strip():
            console.print(Markdown(buf.strip()), width=text_column_width())
        buf = ""

    try:
        for mode, chunk in agent.stream(payload, config=config, stream_mode=["updates", "messages"]):
            if mode == "messages":
                # tool results also arrive here; the `tools` node renders them instead
                if chunk[0].type == "AIMessageChunk":
                    buf += _text(chunk[0].content)
                continue

            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                elif node == "model":
                    for m in (update or {}).get("messages", []):
                        if usage is not None:
                            usage.record_message(
                                getattr(m, "usage_metadata", None),
                                getattr(m, "response_metadata", {}).get("model_name"),
                            )
                        for call in getattr(m, "tool_calls", None) or []:
                            inflight[call["id"]] = (call["name"], call["args"])
                            flush()
                            spinner.update(f"[dim]{call['name']} running…[/dim]")
                elif node == "tools":
                    for msg in (update or {}).get("messages", []):
                        name, args = inflight.pop(getattr(msg, "tool_call_id", ""), (msg.name, {}))
                        flush()
                        _tool_card(name, args, Text("\n").join(_tool_result_lines(msg)), _state_for(msg))
        flush()
    finally:
        spinner.stop()
        console.print()

    return pending


def _ask_decisions(payload: dict) -> dict:
    decisions = []
    for req in payload["action_requests"]:
        body = Text()
        body.append(f"{req['name']}\n", style="accent")
        body.append(_summarize(req["name"], req["args"]), style="model")
        console.print(Panel(body, title="[warn]needs approval[/warn]", border_style="warn"))

        if typer.confirm("run it?", default=False):
            decisions.append({"type": "approve"})
        elif typer.confirm("reply instead?", default=False):
            decisions.append({"type": "respond", "message": typer.prompt("message")})
        else:
            decisions.append({"type": "reject", "message": "the user rejected this call"})
    return {"decisions": decisions}


class Session:
    """Holds the live agent plus the sqlite checkpointer it shares."""

    def __init__(self, root: Path, session_id: str, store: Store, *, approve: bool, model=None):
        self.root = root
        self.id = session_id
        self.store = store
        self.approve = approve
        self.plan_mode = False
        self.model = model
        self.mcp_tools: list = []
        self.mcp_dead: list[str] = []
        self.usage = Usage()
        self.max_model_calls: int | None = None
        self.redact_pii = False
        self.rubric: str | None = None
        self.last_prompt: str | None = None
        self._stack = ExitStack()
        self.checkpointer = self._stack.enter_context(SqliteSaver.from_conn_string(str(store.path)))

        from .config import resolve

        self.model_alias = None
        if model is None:
            self.model_spec = resolve(None)
        else:
            self.model_spec = {"alias": "injected", "model": type(model).__name__, "base_url": "—"}

    def config(self) -> dict:
        return {"configurable": {"thread_id": self.id}}

    def rebuild(self):
        self.agent = build_agent(
            self.root,
            approve=self.approve and not self.plan_mode,
            checkpointer=self.checkpointer,
            plan_mode=self.plan_mode,
            model=self.model,
            model_alias=self.model_alias,
            mcp_tools=self.mcp_tools,
            max_model_calls=self.max_model_calls,
            redact_pii=self.redact_pii,
            rubric=self.rubric,
        )

    def try_rebuild(self) -> bool:
        """Rebuild, reporting failure instead of killing the session.

        A bad model alias or a broken config must not take the TUI down; the
        previous agent stays usable.
        """
        try:
            self.rebuild()
            return True
        except (SystemExit, ValueError, TypeError, AttributeError, KeyError, OSError) as e:
            detail = (str(e) or type(e).__name__).strip().splitlines()[0][:110]
            console.print(Text(f"  cannot apply that: {detail}", style="bad"))
            console.print(Text("  previous settings kept", style="muted"))
            return False

    def close(self):
        self._stack.close()


_session = None


class _DropSpecNits(logging.Filter):
    """Deepagents warns on every turn that a skill's `name` differs from its
    directory. It is a cosmetic spec nit and the skill still loads, so it is
    noise in an interactive session. Genuine "Skipping ..." warnings still show.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "does not follow Agent Skills specification" not in record.getMessage()


# Must attach to the logger that actually emits. A filter on a parent logger is
# NOT applied to records from child loggers during propagation.
_SKILLS_LOGGERS = ("deepagents.middleware.skills", "deepagents")
for _name in _SKILLS_LOGGERS:
    _lg = logging.getLogger(_name)
    _lg.addFilter(_DropSpecNits())
    for _h in _lg.handlers:
        _h.addFilter(_DropSpecNits())


def _pick_skill(root: Path) -> str | None:
    """Arrow-key skill picker.

    Reuses the command palette rather than prompt_toolkit's radiolist_dialog,
    which shows radio buttons instead of names and floods the whole screen.
    """
    from .catalog import skill_entries
    from .palette import show_palette

    items = skill_entries(root)
    if not items:
        return None
    picked = show_palette(items, title="Skills")
    return picked.removeprefix("/skill:") or None if picked else None


def _palette_bindings(root: Path):
    """ctrl+k opens the palette, ctrl+o folds/unfolds tool output.

    Passed into PromptSession; do NOT mutate `session.app.key_bindings` --
    that is a _MergedKeyBindings and has no add().

    The handler is async and goes through run_in_terminal: a key handler already
    runs inside prompt_toolkit's event loop, and Application.run() calls
    asyncio.run(), which refuses to nest.
    """
    from prompt_toolkit.application.run_in_terminal import run_in_terminal
    from prompt_toolkit.key_binding import KeyBindings

    from .catalog import entries
    from .palette import show_palette

    kb = KeyBindings()

    @kb.add("c-k")
    async def open_palette(event) -> None:
        buf = event.current_buffer
        try:
            picked = await run_in_terminal(lambda: show_palette(entries(root)))
        except Exception as e:  # noqa: BLE001 - a popup failure must not kill the TUI
            console.print(Text(f"  palette unavailable: {type(e).__name__}", style="bad"))
            return
        if picked:
            buf.text = picked
            buf.cursor_position = len(picked)
            buf.validate_and_handle()

    @kb.add("c-o")
    def toggle_fold(event) -> None:
        set_expand_all(not is_expanded())
        state = "expanded" if is_expanded() else "folded"
        event.app.output.write(f"\x1b[2K\r  tool output {state} (ctrl+o)\n")

    return kb


def _ctrl_k(kb):
    """The handler bound to c-k. prompt_toolkit stores Binding objects."""
    for b in kb.bindings:
        if "c-k" in (getattr(b, "keys", None) or []):
            return b.handler
    raise AssertionError("ctrl+k is not bound")


def _make_session(root: Path):
    """Seam so tests can verify wiring without a real Windows console."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.application import get_app
    from prompt_toolkit.history import FileHistory

    from .complete import SlashCompleter

    def on_resize(_event) -> None:
        """Dragging the window edge must not leave half-width panels behind."""
        sync_width()

    session = PromptSession(
        history=FileHistory(str(config_dir() / "history")),
        completer=SlashCompleter(root),
        complete_while_typing=True,
        key_bindings=_palette_bindings(root),
    )
    handler = getattr(get_app(), "event_handler", None)
    if handler is not None:  # DummyApplication has none outside a real session
        handler.add_handler("SizeChanged", on_resize)
    return session


def _read_prompt(root: Path) -> str:
    """Prompt with completion when we have a real terminal, plain read otherwise.

    prompt_toolkit is the only completion lib that works on Windows (readline
    does not exist there), and its Win32 output needs a real console -- hence
    the split, so piping still works.
    """
    global _session
    if not console.is_terminal:
        return console.input("[bold cyan]▌ [/bold cyan]")
    if _session is None:
        _session = _make_session(root)
    return _session.prompt("▌ ")


def _pick(items: list[dict], title: str) -> str | None:
    """One popup for every list in the TUI. Returns the chosen text, or None."""
    if not items:
        return None
    from .palette import Entry, show_palette

    try:
        picked = show_palette([Entry(**e) for e in items], title=title)
    except (RuntimeError, OSError) as e:
        # asyncio.run() cannot nest, and a dumb terminal has no raw mode.
        console.print(Text(f"  popup unavailable: {str(e).splitlines()[0][:60]}", style="bad"))
        console.print(Text("  pass 'list' to print instead", style="muted"))
        return None
    if not picked:
        console.print(Text("  cancelled", style="muted"))
    return picked


def _model_items() -> list[dict]:
    from .config import active_alias, discover_models

    models = discover_models()
    try:
        current = active_alias()
    except (OSError, ValueError):
        current = None
    return [
        {
            "text": alias,
            "label": f"{'▸' if alias == current else ' '} {alias}",
            "group": "Model",
            "shortcut": str(spec.get("model", "?")),
            "detail": str(spec.get("base_url", "")),
        }
        for alias, spec in sorted(models.items())
    ]


def _mcp_items() -> list[dict]:
    from . import mcp as mcpmod

    return [
        {
            "text": name,
            "label": f"  {name}",
            "group": "MCP servers",
            "shortcut": "test",
            "detail": str(spec.get("url") or " ".join([spec.get("command", "")] + spec.get("args", []))),
        }
        for name, spec in sorted(mcpmod.connections().items())
    ]


def _session_items(session) -> list[dict]:
    return [
        {
            "text": r["id"],
            "label": f"  {r['id']}",
            "group": "Sessions",
            "shortcut": datetime.fromtimestamp(r["updated_at"]).strftime("%m-%d %H:%M"),
            "detail": (r["title"] or "").strip() or "(untitled)",
        }
        for r in session.store.list(session.root, limit=15)
    ]


def _title_for(session: Session, prompt: str) -> str | None:
    """First prompt names the session; after that, leave it alone.

    Returning None keeps an explicit /rename, and keeps a resumed session
    showing the title it was given rather than the latest stray message.
    """
    existing = {r["id"]: (r["title"] or "") for r in session.store.list(session.root, limit=100)}
    if existing.get(session.id):
        return None
    return derive_title(prompt)


WIDE_BREAKPOINT = 92
SIDEBAR_COLS = 26
SIDEBAR_GAP = 3


def text_column_width() -> int:
    """Width available to chat and the input frame.

    Must match what _render_split actually reserves (sidebar plus the gap), or
    the frame gets rendered wider than its column and the right rule is cut.
    """
    total = sync_width()
    if total >= WIDE_BREAKPOINT + SIDEBAR_COLS:
        return total - SIDEBAR_COLS - SIDEBAR_GAP
    return total


CONTEXT_WINDOW = 128_000  # ponytail: typical chat window; /model shows the real one


def _pct(usage) -> int:
    return min(100, int(usage.total_tokens * 100 / CONTEXT_WINDOW))


def _sidebar(session: Session) -> RenderableType:
    """Zen's right rail: context, MCP, todos. Plain text, no boxes."""
    u = session.usage
    col = Text()
    col.append("Context\n", style="accent")
    for label, value in (
        ("tokens", f"{u.total_tokens:,}"),
        ("used", f"{_pct(u)}%"),
        ("calls", str(u.calls)),
    ):
        col.append(f"{label:<9}", style="muted")
        col.append(f"{value}\n", style="model")

    servers = sorted(session.mcp_tools)
    if servers:
        col.append("\nMCP\n", style="accent")
        for name in servers[:8]:
            col.append("● ", style="ok")
            col.append(f"{name}\n", style="muted")
    for name in sorted(session.mcp_dead):
        col.append("● ", style="bad")
        col.append(f"{name} failed\n", style="muted")

    todos = _todo_lines(session)
    if todos:
        col.append("\n▼ Todo\n", style="accent")
        col.append("\n".join(todos), style="muted")
    return col


def _todo_lines(session: Session) -> list[str]:
    """Read the agent's todo list straight off its state, best effort."""
    try:
        state = session.agent.get_state(session.config()).values
    except Exception:  # noqa: BLE001 - state may not exist yet
        return []
    todos = state.get("todos") or []
    out = []
    for t in todos[:8]:
        mark = {"completed": "[x]", "in_progress": "[~]"}.get(t.get("status"), "[ ]")
        out.append(f"{mark} {t.get('content', '')}")
    return out


def _render_split(left: RenderableType, right: RenderableType, total: int, gap: int = SIDEBAR_GAP) -> None:
    """Two columns sharing one line, Zen style. No box characters.

    Table.grid rather than Columns: Columns decides the column count itself and
    divides by zero when the measured item width does not divide the terminal.
    """
    side = SIDEBAR_COLS
    main = max(20, total - side - gap)
    grid = Table.grid(padding=(0, gap))
    grid.add_column(no_wrap=True, width=main, overflow="fold")
    grid.add_column(no_wrap=True, width=side, overflow="fold")
    grid.add_row(left, right)
    console.print(grid)


def _header(session: Session) -> None:
    spec = session.model_spec or {}
    head = Text()
    head.append(BRAND, style="brand")
    head.append("  ·  ", style="muted")
    head.append(spec.get("model", "?"), style="model")
    head.append("  ", style="muted")
    head.append(f"({spec.get('alias', 'no model')})", style="muted")
    head.append("\n", style="muted")
    title = next(
        (r["title"] for r in session.store.list(session.root, limit=100) if r["id"] == session.id),
        "",
    )
    if title:
        head.append(title, style="accent")
        head.append("\n", style="muted")
    head.append(str(session.root), style="muted")

    badges = []
    if session.plan_mode:
        badges.append(("PLAN MODE", "accent"))
    if tracing_status():
        badges.append((tracing_status(), "trace"))
    if session.max_model_calls:
        badges.append((f"budget {session.max_model_calls}/turn", "warn"))
    if session.redact_pii:
        badges.append(("pii:redact", "warn"))
    if session.mcp_dead:
        badges.append((f"MCP down: {', '.join(session.mcp_dead)}", "bad"))
    if badges:
        head.append("\n", style="muted")
        for i, (txt, style) in enumerate(badges):
            if i:
                head.append(" · ", style="muted")
            head.append(txt, style=style)

    # no rail here: the input frame carries it, right next to the prompt
    console.print(head)


def _input_frame(session: Session) -> None:
    """Zen's input area: a thin rule naming the model, then the prompt gutter.

    Deliberately not a box. A box needs its right edge after the text, and the
    text length is unknown until the user submits, so the frame always ended up
    ragged ("│ ▌ hi" with no closing bar). A trailing rule cannot do that.
    """
    spec = session.model_spec or {}
    label = f" {BRAND} · {spec.get('model', '?')} "
    width = text_column_width()
    label = label[: max(4, width - 1)]
    rule = Text(label + "─" * max(0, width - len(label)), style="muted")

    total = sync_width()
    if total >= WIDE_BREAKPOINT + SIDEBAR_COLS:
        _render_split(rule, _sidebar(session), total)
    else:
        console.print(rule)


def _statusbar(session: Session, mode: str = "") -> None:
    """Zen's bottom rail: key hints on the left, token meter on the right."""
    u = session.usage
    width = sync_width()

    left = Text(" " + (mode or "ctrl+k commands"), style="muted")

    meter = Text()
    if u.total_tokens:
        meter.append(f"{u.total_tokens / 1000:.1f}k ", style="muted")
        meter.append(f"({_pct(u)}%)", style="accent")

    right = Text()
    right.append_text(meter)
    tail = "  ctrl+o expand"
    # the meter is the load-bearing part; the shortcut is a nicety
    if left.cell_len + meter.cell_len + len(tail) + 4 <= width:
        right.append(tail, style="muted")

    prefix = 2
    gap = max(1, width - prefix - left.cell_len - right.cell_len)
    line = Text(" " * prefix)
    line.append_text(left)
    line.append(" " * gap)
    line.append_text(right)
    console.print(line)


def _cmd_model(rest: str, session: Session) -> None:
    from .config import SECRET, discover_models, load, save

    models = discover_models()
    if not rest or rest == "list":
        if not models:
            console.print(Text("  no models configured", style="muted"))
            return
        current = session.model_spec.get("alias") if session.model_spec else None
        for alias, spec in sorted(models.items()):
            mark = "▸" if alias == current else " "
            console.print(
                Text.assemble(
                    (f" {mark} ", "bold green" if alias == current else "dim"),
                    (f"{alias:<14}", "bold cyan"),
                    (f"{spec.get('model', '?'):<24}", ""),
                    (spec.get("base_url", ""), "dim"),
                )
            )
        return

    parts = rest.split()
    if parts[0] == "add":
        if len(parts) < 4:
            console.print(Text("  usage: /model add <alias> <base_url> <model> [api_key]", style="warn"))
            return
        alias, base, model = parts[1], parts[2], parts[3]
        cfg = load()
        cfg.setdefault("models", {})[alias] = {
            "base_url": base,
            "model": model,
            SECRET: parts[4] if len(parts) > 4 else "",
        }
        save(cfg)
        console.print(Text(f"  added {alias}", style="ok"))
        return

    if parts[0] in {"rm", "remove"}:
        cfg = load()
        if parts[1:] and parts[1] in cfg.get("models", {}):
            del cfg["models"][parts[1]]
            save(cfg)
            console.print(Text(f"  removed {parts[1]}", style="ok"))
        else:
            console.print(Text("  usage: /model rm <alias>", style="warn"))
        return

    if parts[0] not in models:
        console.print(Text(f"  unknown model {parts[0]!r} — try /model list", style="warn"))
        return

    cfg = load()
    cfg["active"] = parts[0]
    save(cfg)
    session.model_alias = parts[0]
    session.model_spec = {"alias": parts[0], **models[parts[0]]}
    session.mcp_tools, session.mcp_dead = _load_mcp()
    session.try_rebuild()
    _header(session)


def _cmd_skills(session: Session) -> None:
    from .skills import discover

    found = discover(session.root)
    if not found:
        console.print(Text("  no skills found — add <user-dir>/skills/<name>/SKILL.md", style="muted"))
        return
    for s in found:
        console.print(
            Text.assemble(
                (f"  {s.name:<22}", "bold cyan"),
                (f"[{s.scope}] ", "dim"),
                (s.description[:60], "dim"),
            )
        )
    console.print(Text("  invoke with /skill:<name>", style="muted italic"))


def _cmd_mcp(rest: str, session: Session) -> None:
    import shlex

    from . import mcp

    servers = mcp.connections()
    if not rest or rest == "list":
        if not servers:
            console.print(Text("  no MCP servers configured", style="muted"))
            return
        loaded = {t.name for t in (session.mcp_tools or [])}
        for name, spec in sorted(servers.items()):
            kind = spec.get("url") or " ".join([spec.get("command", "")] + spec.get("args", []))
            console.print(Text.assemble(("  - ", "dim"), (f"{name:<16}", "bold cyan"), (kind[:56], "dim")))
        if loaded:
            console.print(Text(f"  {len(loaded)} tool(s) attached", style="ok"))
        for name, err in session.mcp_dead:
            console.print(Text(f"  x {name}: {err}", style="bad"))
        return

    argv = shlex.split(rest)
    sub, args = argv[0], argv[1:]

    if sub == "add":
        if not args:
            console.print(Text("  usage: /mcp add <name> -- <cmd...> | --url <url>", style="warn"))
            return
        name, rest_args = args[0], args[1:]
        if "--url" in rest_args:
            i = rest_args.index("--url")
            spec = {"url": rest_args[i + 1], "transport": "streamable_http"}
            for tok in ("--token",):
                if tok in rest_args:
                    spec["headers"] = {"Authorization": f"Bearer {rest_args[rest_args.index(tok) + 1]}"}
        elif "--" in rest_args:
            cmd = rest_args[rest_args.index("--") + 1 :]
            if not cmd:
                console.print(Text("  usage: /mcp add <name> -- <cmd...>", style="warn"))
                return
            spec = {"command": cmd[0], "args": cmd[1:]}
        else:
            console.print(Text("  usage: /mcp add <name> -- <cmd...> | --url <url>", style="warn"))
            return
        mcp.add(name, spec)
        ok, msg = mcp.probe(name, spec)
        console.print(
            Text(f"  added {name} — {'ok: ' + msg if ok else 'FAILED: ' + msg}", "green" if ok else "red")
        )
        session.mcp_tools, session.mcp_dead = _load_mcp()
        session.try_rebuild()
    elif sub == "test":
        if len(args) != 1:
            console.print(Text("  usage: /mcp test <name>", style="warn"))
        else:
            ok, msg = mcp.probe(args[0], servers[args[0]])
            console.print(Text(f"  {'✓' if ok else '✗'} {args[0]}: {msg}", "green" if ok else "red"))
    elif sub in {"rm", "remove"}:
        if len(args) != 1:
            console.print(Text("  usage: /mcp rm <name>", style="warn"))
            return
        console.print(
            Text(
                f"  {'removed ' + args[0] if mcp.remove(args[0]) else 'no such server'}",
                "green" if args[0] in servers else "yellow",
            )
        )
        session.mcp_tools, session.mcp_dead = _load_mcp()
        session.try_rebuild()
    else:
        console.print(Text(f"  unknown /mcp subcommand {sub!r}", style="warn"))


def _load_mcp():
    """Returns (tools, [(name, error)]). Never raises -- a dead server is not fatal."""
    from . import mcp

    try:
        tools, dead = mcp.load_tools()
        return tools, dead
    except Exception as e:  # noqa: BLE001
        return [], [("all", str(e).splitlines()[0][:60])]


def _cmd_usage(session: Session) -> None:
    u = session.usage
    if not u.turns and not u.calls:
        console.print(Text("  no usage yet", style="muted"))
        return

    rows = [
        ("model calls", str(u.calls)),
        ("input tokens", f"{u.input_tokens:,}"),
        ("output tokens", f"{u.output_tokens:,}"),
        ("total tokens", f"{u.total_tokens:,}"),
        ("turns", str(u.turns)),
    ]
    table = Table(box=None, show_header=False, padding=(0, 2))
    for k, v in rows:
        table.add_row(Text(k, style="muted"), Text(v, style="accent"))

    if u.by_model:
        table.add_row("", "")
        for name, b in sorted(u.by_model.items()):
            table.add_row(
                Text(name, style="model"), Text(f"{b['calls']} calls · {b['in']:,} in · {b['out']:,} out")
            )

    console.print(Panel(table, title="[model]usage[/model]", border_style="panel.border", padding=(0, 1)))


def _cmd_budget(rest: str, session: Session) -> None:
    arg = rest.strip().lower()
    if not arg or arg in {"status", "show"}:
        if session.max_model_calls:
            console.print(Text(f"  max {session.max_model_calls} model calls per turn", style="muted"))
        else:
            console.print(Text("  no budget set (unlimited model calls per turn)", style="muted"))
        return
    if arg in {"off", "none", "0"}:
        session.max_model_calls = None
        session.try_rebuild()
        console.print(Text("  budget cleared", style="ok"))
        return
    if not arg.isdigit() or int(arg) < 1:
        console.print(Text("  usage: /budget <n>   or   /budget off", style="warn"))
        return
    session.max_model_calls = int(arg)
    session.try_rebuild()
    console.print(Text(f"  budget set to {arg} model calls per turn", style="ok"))


def _cmd_pii(rest: str, session: Session) -> None:
    arg = rest.strip().lower()
    if arg in {"", "status"}:
        console.print(
            Text(
                f"  pii redaction is {'on' if session.redact_pii else 'off'}",
                style="muted",
            )
        )
        return
    session.redact_pii = arg == "on"
    session.try_rebuild()
    console.print(Text(f"  pii redaction {'on' if session.redact_pii else 'off'}", style="ok"))


def _cmd_rubric(rest: str, session: Session) -> None:
    """Opt-in self-evaluation loop: the agent re-runs until the rubric passes.

    Costs extra model calls, so it stays off until asked for by name.
    """
    arg = rest.strip()
    if arg.lower() in {"", "status", "show"}:
        if session.rubric:
            console.print(Text(f"  rubric: {session.rubric}", style="muted"))
        else:
            console.print(Text("  no rubric set", style="muted"))
        return

    if arg.lower() in {"off", "none", "clear"}:
        session.rubric = None
        session.try_rebuild()
        console.print(Text("  rubric cleared", style="ok"))
        return

    session.rubric = arg
    if not session.try_rebuild():
        session.rubric = None
        return
    console.print(Text("  rubric set - the agent grades itself and iterates (up to 3x)", style="ok"))
    console.print(Text(f"  {arg}", style="muted"))


def _cmd_trace(rest: str, session: Session) -> None:
    arg = rest.strip().lower()
    if arg in {"", "status"}:
        console.print(Text(f"  tracing {'on' if tracing_enabled() else 'off'}", style="muted"))
        return
    if arg == "off":
        import os as _os

        _os.environ.pop("LANGSMITH_TRACING", None)
        _os.environ.pop("LANGCHAIN_TRACING_V2", None)
        session.try_rebuild()
        console.print(Text("  tracing off", style="ok"))
        return
    result = enable_tracing()
    if "no " in result:
        console.print(Text(f"  cannot enable tracing: {result}", style="warn"))
    else:
        session.try_rebuild()
        console.print(Text(f"  tracing on → project {result}", style="ok"))
        console.print(Text("    view runs at https://smith.langchain.com", style="muted"))


def _cmd_todos(agent, config) -> None:
    state = agent.get_state(config).values
    todos = state.get("todos") or []
    if not todos:
        console.print(Text("  no tasks yet", style="muted"))
        return
    marks = {"completed": ("✓", "green"), "in_progress": ("▸", "yellow"), "pending": ("·", "dim")}
    for t in todos:
        mark, style = marks.get(t.get("status"), ("·", "dim"))
        console.print(Text.assemble((f"  {mark} ", style), (t.get("content", ""), style)))


def _cmd_memory(root: Path) -> None:
    files = find_memory(root)
    if not files:
        console.print(Text("  no context files — add AGENTS.md", style="muted"))
        return
    for f in files:
        console.print(Text.assemble(("  · ", "dim"), (Path(f).name, "cyan")))


def _cmd_sessions(store: Store, cwd: Path) -> None:
    rows = store.list(cwd)
    if not rows:
        console.print(Text("  no saved sessions here", style="muted"))
        return
    for r in rows:
        when = datetime.fromtimestamp(r["updated_at"]).strftime("%m-%d %H:%M")
        title = (r["title"] or "").strip() or "(untitled)"
        console.print(
            Text.assemble(
                (f"  {r['id']}  ", "cyan"),
                (f"{when}  ", "dim"),
                (title[:60], ""),
            )
        )


def _run_command(line: str, session: Session) -> str | None:
    """Handle a slash command. Returns a prompt to run, or None."""
    head, _, rest = line.partition(" ")
    rest = rest.strip()
    cmd = head.lower()

    if cmd in {"/help", "/?"}:
        # HELP is Rich markup, not Markdown -- Markdown() would print [bold] raw.
        console.print(Panel(Text.from_markup(HELP), border_style="panel.border", padding=(0, 1)))
    elif cmd == "/model":
        # bare /model opens the picker; "/model list" keeps the plain listing
        if rest.lower() == "list":
            _cmd_model(rest, session)
        else:
            picked = _pick(_model_items(), "Models")
            if picked:
                _cmd_model(picked, session)
    elif cmd == "/sessions":
        picked = _pick(_session_items(session), "Sessions")
        if picked:
            session.id = picked
            session.try_rebuild()
            _header(session)
    elif cmd == "/skills":
        # "list" keeps the old behaviour for pipes; bare /skills opens the picker.
        if rest.lower() == "list" or not console.is_terminal:
            _cmd_skills(session)
        else:
            name = _pick_skill(session.root)
            if name:
                return f"Use the {name} skill for this task."
            console.print(Text("  cancelled", style="muted"))
    elif cmd.startswith("/skill:"):
        name = cmd.split(":", 1)[1]
        return f"Use the {name} skill for this task.\n\n{rest}" if rest else f"Use the {name} skill."
    elif cmd == "/mcp":
        if rest:
            _cmd_mcp(rest, session)
        else:
            picked = _pick(_mcp_items(), "MCP servers")
            if picked:
                _cmd_mcp(f"test {picked}", session)
    elif cmd == "/usage":
        _cmd_usage(session)
    elif cmd == "/budget":
        _cmd_budget(rest, session)
    elif cmd == "/pii":
        _cmd_pii(rest, session)
    elif cmd == "/rubric":
        _cmd_rubric(rest, session)
    elif cmd == "/trace":
        _cmd_trace(rest, session)
    elif cmd == "/retry":
        if session.last_prompt:
            return session.last_prompt
        console.print(Text("  nothing to retry", style="warn"))
    elif cmd == "/plan":
        session.plan_mode = not session.plan_mode
        session.try_rebuild()
        _header(session)
        return rest or None
    elif cmd == "/todos":
        _cmd_todos(session.agent, session.config())
    elif cmd == "/memory":
        _cmd_memory(session.root)
    elif cmd == "/sessions":
        _cmd_sessions(session.store, session.root)
    elif cmd == "/rename":
        if not rest:
            console.print(Text("  usage: /rename <title>", style="warn"))
        else:
            session.store.rename(session.id, rest)
            console.print(Text(f"  renamed to {rest!r}", style="ok"))
    elif cmd == "/export":
        out = Path(rest) if rest else session.root / f"{session.id}.json"
        session.store.export(session.id, out)
        console.print(Text(f"  wrote {out}", style="ok"))
    elif cmd == "/clear":
        session.agent.update_state(session.config(), {"messages": []})
        console.print(Text("  cleared", style="ok"))
    elif cmd == "/new":
        session.id = new_session_id()
        session.store.create(session.id, session.root)
        session.try_rebuild()
        _header(session)
    elif cmd == "/resume":
        target = rest or session.store.latest(session.root)
        if not target:
            console.print(Text("  no sessions to resume", style="warn"))
        elif target == session.id:
            console.print(Text("  already here", style="muted"))
        else:
            session.id = target
            session.try_rebuild()
            _header(session)
    else:
        _suggest(cmd)

    return None


def _suggest(cmd: str) -> None:
    """'unknown command' is useless on its own; show what they probably meant."""
    from .catalog import COMMANDS
    from .palette import fuzzy

    known = [t for t, *_ in COMMANDS] + [f"/skill:{s}" for s in _skill_names()]
    scored = sorted((s, n) for n, s in ((n, fuzzy(cmd, k)) for n, k in enumerate(known)) if s is not None)
    close = [known[n] for _, n in scored[:3]]

    console.print(Text(f"  unknown command {cmd}", style="warn"))
    if close:
        console.print(Text("  did you mean: ", style="muted") + Text("  ".join(close), style="model"))
    console.print(Text("  ctrl+k opens the palette", style="muted italic"))


def _skill_names() -> list[str]:
    try:
        from .skills import discover

        return [s.name for s in discover(Path.cwd())]
    except (OSError, ValueError):
        return []


@app.command()
def run(
    task: str | None = typer.Argument(None, help="Task to run once, then exit."),
    cwd: Path = typer.Option(Path.cwd(), "--cwd", help="Directory the agent may touch."),
    approve: bool = typer.Option(
        False, "--approve", help="Pause and ask before every write/edit/delete/shell command."
    ),
    resume: bool = typer.Option(False, "--resume", "-c", help="Continue the latest session here."),
) -> None:
    """Coding agent in your terminal."""
    use_utf8_console()
    root = cwd.resolve()
    # Import cost plus building the model and opening the checkpointer takes
    # ~5s; without this line the terminal just sits blank and the first thing the
    # user does is type into a prompt that is not listening yet.
    console.print(Text(f" {BRAND}  starting…", style="brand"))
    store = Store(config_dir() / "sessions.db")
    session_id = (store.latest(root) if resume else None) or new_session_id()
    store.create(session_id, root, title=derive_title(task) if task else "")

    session = Session(root, session_id, store, approve=approve)
    session.mcp_tools, session.mcp_dead = _load_mcp()
    session.try_rebuild()
    _header(session)

    prompt = task
    try:
        while True:
            if prompt is None:
                # the rail and the prompt rule are adjacent, so the context
                # numbers the user reads are the ones for this turn
                _input_frame(session)
                try:
                    prompt = _read_prompt(root).strip()
                except (EOFError, KeyboardInterrupt):
                    console.print()
                    return
                _statusbar(session)
                if not prompt:
                    continue
                if prompt.lower() in EXIT_WORDS:
                    return
                if prompt.startswith("/"):
                    nxt = _run_command(prompt, session)
                    # None means "handled, go back to the prompt". Clearing it
                    # first matters: `continue` alone would leave the old text
                    # in `prompt` and the loop would send it to the model.
                    prompt = nxt
                    if prompt is None:
                        continue

            try:
                store.touch(session.id, root, title=_title_for(session, prompt))
                session.last_prompt = prompt
                payload = {"messages": [{"role": "user", "content": prompt}]}
                pending = _drive(session.agent, session.config(), payload, usage=session.usage)
                while pending:
                    pending = _drive(
                        session.agent,
                        session.config(),
                        Command(resume=_ask_decisions(pending)),
                        usage=session.usage,
                    )
                session.usage.end_turn()
            except KeyboardInterrupt:
                # Esc/Ctrl-C stops the turn but keeps the session (omp: Escape stops a turn)
                console.print(Text("  stopped", style="warn"))
            except Exception as e:  # noqa: BLE001 - a provider/transport failure must
                # not take down the REPL; rate limits and network blips happen
                detail = str(e).splitlines()[0][:110] if str(e) else type(e).__name__
                console.print(Text(f"  request failed: {detail}", style="bad"))

            if task is not None:
                return
            prompt = None
    finally:
        session.close()


if __name__ == "__main__":
    app()
