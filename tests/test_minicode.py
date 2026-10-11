"""Runnable check: agent wiring, tools present, and interrupt resume contract.

Run: uv run pytest -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class ScriptedModel(GenericFakeChatModel):
    """Replays a fixed script of AIMessages, then stops."""

    script: list[AIMessage] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        msg = self.script.pop(0) if self.script else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        msg = self.script.pop(0) if self.script else AIMessage(content="done")
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=msg.content, tool_calls=msg.tool_calls, id=msg.id)
        )


def scripted(*script: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(script), script=list(script))


def _agent(tmp_path: Path, approve: bool):
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"file_path": "/x.txt", "content": "hi"}, "id": "1"}],
        ),
        AIMessage(content="wrote it"),
    )
    return create_deep_agent(
        model=model,
        backend=LocalShellBackend(root_dir=str(tmp_path)),
        checkpointer=InMemorySaver(),
        interrupt_on={"write_file": True} if approve else None,
    )


CFG = {"configurable": {"thread_id": "t"}}


def _capture_drive(agent, payload) -> str:
    """Render through a real console so Panels/Text come out as text, not objects."""
    import io

    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    buf = io.StringIO()
    real = cli.console
    cli.console = Console(file=buf, width=100, force_terminal=False, theme=RICH_THEME)
    try:
        cli._drive(agent, CFG, payload)
    finally:
        cli.console = real
    return buf.getvalue()


def _tool_msg(name: str, content: str, **kw) -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id="1", **kw)


def test_write_tool_lands_on_disk(tmp_path: Path):
    agent = _agent(tmp_path, approve=False)
    agent.invoke({"messages": [{"role": "user", "content": "write x.txt"}]}, CFG)
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_interrupt_blocks_write_until_resume(tmp_path: Path):
    agent = _agent(tmp_path, approve=True)
    config = CFG

    result = agent.invoke({"messages": [{"role": "user", "content": "write x.txt"}]}, config)
    interrupts = result.get("__interrupt__") or []
    assert interrupts, f"expected an interrupt before the write ran; got keys {list(result)}"
    assert not (tmp_path / "x.txt").exists(), "write ran before approval"

    req = interrupts[0].value
    print("INTERRUPT PAYLOAD:", req)

    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_reject_leaves_disk_untouched(tmp_path: Path):
    agent = _agent(tmp_path, approve=True)
    agent.invoke({"messages": [{"role": "user", "content": "write x.txt"}]}, CFG)
    agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "nope"}]}), CFG)
    assert not (tmp_path / "x.txt").exists()


def test_execute_tool_is_available(tmp_path: Path):
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {"command": "echo hello"}, "id": "1"}],
        ),
        AIMessage(content="ok"),
    )
    agent = create_deep_agent(
        model=model,
        backend=LocalShellBackend(root_dir=str(tmp_path), inherit_env=True),
        checkpointer=InMemorySaver(),
    )
    names = set(agent.get_graph().nodes.keys())
    assert "tools" in names, f"no tool node in {names}"

    result = agent.invoke({"messages": [{"role": "user", "content": "run echo"}]}, CFG)
    outputs = " ".join(str(m.content) for m in result["messages"] if getattr(m, "type", "") == "tool")
    assert "hello" in outputs, f"execute tool output missing: {outputs}"


def test_build_model_requires_config(monkeypatch):
    from minicode import agent as agent_mod

    for var in (
        "MINICODE_BASE_URL",
        "OPENAI_BASE_URL",
        "MINICODE_API_KEY",
        "OPENAI_API_KEY",
        "MINICODE_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit) as exc:
        agent_mod.build_model()
    assert "MINICODE_BASE_URL" in str(exc.value)


def test_build_model_reads_custom_endpoint(monkeypatch):
    from minicode import agent as agent_mod

    monkeypatch.setenv("MINICODE_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("MINICODE_API_KEY", "test-key")
    monkeypatch.setenv("MINICODE_MODEL", "my-model")
    m = agent_mod.build_model()
    assert m.model_name == "my-model"
    assert str(m.openai_api_base).rstrip("/").endswith("/v1")


def test_cli_drive_surfaces_interrupt(tmp_path: Path):
    """The REPL must actually see the pause, else --approve silently auto-runs."""
    agent = _agent(tmp_path, approve=True)

    from minicode.cli import _drive

    payload = _drive(agent, CFG, {"messages": [{"role": "user", "content": "write x.txt"}]})
    assert payload is not None, "stream did not surface the interrupt"
    assert payload["action_requests"][0]["name"] == "write_file"
    assert not (tmp_path / "x.txt").exists()


def test_cli_summaries_are_readable():
    """Raw kwargs dumps in the TUI are the failure mode this guards."""
    from minicode.cli import _summarize

    assert _summarize("read_file", {"file_path": "/a/b.py", "offset": 1, "limit": 20}) == "b.py"
    assert _summarize("execute", {"command": "npm   run\n  test"}) == "npm run test"
    assert _summarize("grep", {"pattern": "TODO", "path": "/src"}) == "TODO in src"
    assert _summarize("write_file", {"file_path": "/x", "content": "y" * 500}) == "x"


def test_tool_cards_do_not_inhale_windows_profile_paths():
    """Regression: a real Windows profile path is ~55 chars with spaces in it
    ("Ahdi Khalida Fathir"), which wrapped mid-word and shoved the card off the
    right edge. Cards must show the leaf only."""
    from minicode.cli import _card_title, _summarize

    long = r"C:\Users\Ahdi Khalida Fathir\AppData\Local\Temp\opencode\minicode\cli.py"
    assert _summarize("read_file", {"file_path": long}) == "cli.py"

    card = _card_title("read_file", {"file_path": long})
    assert card.cell_len <= 40, f"card too wide: {card.cell_len}"
    assert "Users" not in card.plain and "AppData" not in card.plain


def test_leaf_name_handles_spaces_and_separators():
    from minicode.sessions import leaf_name

    assert leaf_name(r"C:\Users\Ahdi Khalida Fathir\proj\file.txt") == "file.txt"
    assert leaf_name("/usr/local/bin/tool") == "tool"
    assert leaf_name("bare") == "bare"
    assert leaf_name("") == ""


def test_cli_renders_tool_results(tmp_path: Path):
    """execute output must reach the terminal, not vanish into the graph."""
    model = scripted(
        AIMessage(
            content="",
            tool_calls=[{"name": "execute", "args": {"command": "echo hi"}, "id": "1"}],
        ),
        AIMessage(content="done"),
    )
    agent = create_deep_agent(
        model=model,
        backend=LocalShellBackend(root_dir=str(tmp_path)),
        checkpointer=InMemorySaver(),
    )

    blob = _capture_drive(agent, {"messages": [{"role": "user", "content": "test"}]})
    assert "bash" in blob, f"tool call not rendered: {blob!r}"
    assert "exit 0" in blob, f"tool result not rendered: {blob!r}"
    assert "Command succeeded" not in blob, f"redundant exit suffix leaked: {blob!r}"


def test_cli_collapses_list_results():
    """ls returning 40 paths must render a count, not 40 lines."""
    from minicode.cli import _tool_result_lines

    names = "[" + ", ".join(f"'/dir/file{i}.py'" for i in range(40)) + "]"
    line = _tool_result_lines(_tool_msg("ls", names))[0].plain
    assert "file0.py" in line
    assert "+36 more" in line, line
    assert line.count("\n") == 0, f"result leaked newlines: {line!r}"


def test_cli_marks_failed_tools():
    from minicode.cli import _tool_result_lines

    err = _tool_result_lines(_tool_msg("execute", "boom", status="error"))[0].plain
    assert "failed" in err, err

    bad = _tool_result_lines(_tool_msg("execute", "1 test failed", artifact={"exit_code": 1}))[0].plain
    assert "exit 1" in bad, bad

    good = _tool_result_lines(_tool_msg("execute", "ok", artifact={"exit_code": 0}))[0].plain
    assert "exit 0" in good, good


def test_long_tool_output_folds_by_default():
    """opencode-style: a 300-line result must not flood the scrollback."""
    from minicode.cli import COLLAPSED_LINES, _tool_result_lines, set_expand_all

    body = "\n".join(f"line {i}" for i in range(300))
    try:
        set_expand_all(False)
        folded = _tool_result_lines(_tool_msg("read_file", body))
        assert len(folded) <= COLLAPSED_LINES + 1, len(folded)
        assert any("Ctrl+O: Expand" in t.plain for t in folded), [t.plain for t in folded]

        set_expand_all(True)
        full = _tool_result_lines(_tool_msg("read_file", body))
        assert len(full) > COLLAPSED_LINES, "ctrl+o did not expand"
    finally:
        set_expand_all(False)


def test_tool_cards_use_kinds_not_raw_names():
    from minicode.cli import _card_title

    assert _card_title("read_file", {"file_path": "/a.py"}).plain.strip().startswith("▸ read")
    assert _card_title("execute", {"command": "ls"}).plain.strip().startswith("▸ bash")
    assert _card_title("edit_file", {"file_path": "/b"}).plain.strip().startswith("▸ edit")
    assert _card_title("write_file", {"file_path": "/b"}).plain.strip().startswith("▸ write")
    assert _card_title("fetch_url", {"url": "u"}).plain.strip().startswith("▸ fetch")


def test_card_border_colour_tracks_state():
    from minicode.cli import _state_for

    assert _state_for(_tool_msg("execute", "ok", artifact={"exit_code": 0})) == "success"
    assert _state_for(_tool_msg("execute", "no", artifact={"exit_code": 1})) == "error"
    assert _state_for(_tool_msg("execute", "x", status="error")) == "error"


def _card_lines(name, args, result, border="success", width=100):
    """Render a tool card offscreen and return its visible lines."""
    import io

    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    buf = io.StringIO()
    old = cli.console
    cli.console = Console(file=buf, width=width, force_terminal=False, theme=RICH_THEME)
    from rich.text import Text

    body = None if result is None else Text(result)
    # sync_width reads the live terminal, which the harness does not control.
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(cli, "sync_width", lambda: width)
    try:
        cli._tool_card(name, args, body, border)
    finally:
        mp.undo()
        cli.console = old
    return buf.getvalue().splitlines()


def test_tool_card_has_no_box_border():
    """Zen/omp render tool calls flat. A bordered panel per call turns a long
    run into a wall of rectangles."""
    lines = _card_lines("write", {"file_path": "a.py"}, "Updated file a.py")
    for line in lines:
        assert not set(line) & set("╭╮╰╯│├─"), f"box drawing in tool output: {line!r}"


def test_tool_card_starts_with_a_marker():
    lines = _card_lines("bash", {"command": "pytest -q"}, "8 passed")
    assert "▸" in lines[0], lines[0]
    assert "bash" in lines[0], lines[0]


def test_tool_card_indents_output_under_the_header():
    """Output hangs off the header, it does not start at column zero."""
    lines = _card_lines("bash", {"command": "pytest -q"}, "8 passed")
    body = lines[1:]
    assert body, "expected output lines"
    for line in body:
        assert line.startswith("    "), f"not indented: {line!r}"


@pytest.mark.parametrize("width", [60, 80, 100, 132, 200])
def test_tool_card_reflows_without_overflowing(width):
    lines = _card_lines("bash", {"command": "pytest -q"}, "exit 0  8 passed", width=width)
    for line in lines:
        assert len(line.rstrip()) <= width, f"overflows at {width}: {len(line.rstrip())}"


def test_long_output_wraps_inside_the_column():
    lines = _card_lines("bash", {"command": "x"}, "z" * 500, width=80)
    for line in lines:
        assert len(line.rstrip()) <= 80, f"long output overflowed: {len(line.rstrip())}"


def test_empty_result_still_renders_the_header_only():
    lines = _card_lines("bash", {"command": "x"}, None)
    assert len(lines) == 1, lines
    assert "bash" in lines[0]


def test_help_renders_markup_instead_of_printing_it(capsys):
    """Regression: HELP is Rich markup but was rendered through Markdown(),
    which does not parse markup, so /help printed a literal [bold]."""
    import io

    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    buf = io.StringIO()
    old = cli.console
    cli.console = Console(file=buf, width=100, theme=RICH_THEME)
    try:
        cli.console.print(
            __import__("rich.panel", fromlist=["Panel"]).Panel(
                __import__("rich.text", fromlist=["Text"]).Text.from_markup(cli.HELP),
                border_style="panel.border",
                padding=(0, 1),
            )
        )
    finally:
        cli.console = old
    out = buf.getvalue()
    assert "[bold]" not in out, "markup leaked into output"
    assert "general" in out


def test_help_panel_lines_are_rectangular():
    import io

    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from minicode import cli
    from minicode.theme import RICH_THEME

    for width in (60, 80, 120):
        buf = io.StringIO()
        c = Console(file=buf, width=width, theme=RICH_THEME)
        c.print(Panel(Text.from_markup(cli.HELP), border_style="panel.border", padding=(0, 1)))
        lines = buf.getvalue().splitlines()
        assert {len(x.rstrip()) for x in lines if x.strip()} == {width}, f"ragged at {width}"


class _FakeSize:
    def __init__(self, columns, rows=24):
        self.columns, self.rows = columns, rows


class _FakeOutput:
    def __init__(self, columns):
        self._c = columns

    def get_size(self):
        return _FakeSize(self._c)


def test_sync_width_follows_prompt_toolkit_not_the_os(monkeypatch):
    """Rich asks the OS for width; prompt_toolkit owns the visible prompt.
    When they disagree the panels stop short of their own content."""
    import io

    import prompt_toolkit.application as ptk_app
    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    monkeypatch.setattr(ptk_app, "get_app", lambda: type("A", (), {"output": _FakeOutput(72)})())
    buf = io.StringIO()
    old = cli.console
    cli.console = Console(file=buf, width=120, theme=RICH_THEME)
    try:
        assert cli.sync_width() == 72, "did not adopt the prompt_toolkit width"
    finally:
        cli.console = old


def test_sync_width_survives_no_running_app(monkeypatch):
    import io

    import prompt_toolkit.application as ptk_app
    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    def boom():
        raise RuntimeError("no running application")

    monkeypatch.setattr(ptk_app, "get_app", boom)
    buf = io.StringIO()
    old = cli.console
    cli.console = Console(file=buf, width=100, theme=RICH_THEME)
    try:
        assert cli.sync_width() == 100, "must fall back to the OS size"
    finally:
        cli.console = old


def _render(fn, cols=120, *args, **kwargs):
    """Render a cli draw function offscreen and return its lines."""
    import io

    from rich.console import Console

    from minicode import cli
    from minicode.theme import RICH_THEME

    buf = io.StringIO()
    old = cli.console
    cli.console = Console(file=buf, width=cols, theme=RICH_THEME, force_terminal=False)
    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "sync_width", lambda: cols)
    try:
        fn(*args, **kwargs)
    finally:
        mp.undo()
        cli.console = old
    return buf.getvalue().splitlines()


class _FakeUsage:
    def __init__(self, i=0, o=0, calls=0):
        self.input_tokens, self.output_tokens, self.calls, self.turns = i, o, calls, 1

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


class _FakeSide:
    def __init__(self):
        from minicode.usage import Usage

        self.usage = Usage()
        self.usage.input_tokens = 90_606
        self.usage.output_tokens = 8_000
        self.usage.calls = 4
        self.mcp_tools = {"browsermcp": [], "other": []}
        self.mcp_dead = []
        self.plan_mode = False
        self.max_model_calls = 0
        self.redact_pii = False
        self.model_spec = {"model": "test-model", "alias": "default"}
        self.root = Path(".")
        self.id = "x"

        class _StubStore:
            def list(self, root, limit=100):
                return [{"id": "x", "title": "t"}]

        self.store = _StubStore()


def test_sidebar_shows_context_mcp_and_todos():
    from minicode.cli import _sidebar

    s = _FakeSide()
    out = "".join(str(_sidebar(s)))
    assert "Context" in out
    assert "98,606" in out, out
    assert "browsermcp" in out
    assert "%" in out


def test_sidebar_marks_dead_servers():
    from minicode.cli import _sidebar

    s = _FakeSide()
    s.mcp_dead = ["flaky"]
    out = "".join(str(_sidebar(s)))
    assert "flaky" in out and "failed" in out


def test_sidebar_survives_an_agent_with_no_state():
    """get_state raises before the first turn; the rail must not take the TUI down."""
    from minicode.cli import _sidebar

    class Boom:
        def get_state(self, cfg):
            raise RuntimeError("no checkpoint yet")

    s = _FakeSide()
    s.agent = Boom()
    s.config = dict
    out = "".join(str(_sidebar(s)))
    assert "Context" in out


def test_split_puts_the_rail_on_the_right():
    from rich.text import Text

    from minicode.cli import _render_split

    lines = _render(lambda: _render_split(Text("MAIN COLUMN"), Text("SIDE RAIL"), 120))
    assert "MAIN COLUMN" in lines[0] and "SIDE RAIL" in lines[0]
    assert lines[0].index("MAIN COLUMN") < lines[0].index("SIDE RAIL")


def test_split_does_not_blow_up_on_narrow_terminals():
    """Columns used to divide by zero; Table.grid must just squeeze."""
    from rich.text import Text

    from minicode.cli import _render_split

    for cols in (40, 60, 80, 120):
        lines = _render(lambda c=cols: _render_split(Text("MAIN"), Text("SIDE"), c), cols)
        assert "MAIN" in lines[0], cols
        assert "SIDE" in lines[0], cols


def test_sidebar_only_appears_when_there_is_room():
    from minicode.cli import SIDEBAR_COLS, WIDE_BREAKPOINT

    assert WIDE_BREAKPOINT + SIDEBAR_COLS >= 100, "breakpoint must leave the rail usable"
    from minicode import cli
    from minicode.cli import text_column_width

    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "sync_width", lambda: 80)
    assert text_column_width() == 80, "narrow terminal must use the full width"
    mp.undo()

    mp = pytest.MonkeyPatch()
    mp.setattr(cli, "sync_width", lambda: 160)
    assert text_column_width() < 160, "wide terminal must reserve the rail"
    mp.undo()


def test_input_frame_is_a_clean_rule_not_a_ragged_box():
    """Regression: the boxed input always looked ragged -- '│ ▌ hi' had no right
    edge because the text length is unknown until submit. Now it's a rule."""
    import minicode.cli as C

    lines = _render(C._input_frame, 120, _FakeSide())
    rule = lines[0]
    assert "╭" not in rule and "╮" not in rule, f"box should be gone: {rule!r}"
    assert "ThirtyAgent" in rule
    assert all(len(ln) <= 120 for ln in lines), "frame overflowed the terminal"


def test_input_frame_carries_the_sidebar_on_wide_terminals():
    import minicode.cli as C

    lines = _render(C._input_frame, 120, _FakeSide())
    joined = "\n".join(lines)
    assert "Context" in joined, "sidebar should sit beside the input frame"
    assert "tokens" in joined


def test_header_does_not_duplicate_the_sidebar():
    """The rail moved to the input frame; rendering it in both places doubled it."""
    import minicode.cli as C

    joined = "\n".join(_render(C._header, 120, _FakeSide()))
    assert "Context" not in joined, "header still renders its own rail"


def test_bare_pickers_have_one_entry_point(monkeypatch, tmp_path: Path):
    """Regression: /mcp opened a picker but /mcp-pick also existed as a hidden
    duplicate, and /mcp list printed raw rows instead of the picker."""
    import minicode.cli as C

    picked: list[str] = []
    monkeypatch.setattr(C, "_pick", lambda items, title: picked.append(title) or None)
    monkeypatch.setattr(C, "_mcp_items", list)

    class _S:
        mcp_tools: list = []
        mcp_dead: list = []

    C._run_command("/mcp", _S())
    assert picked == ["MCP servers"], f"bare /mcp should open the picker, got {picked}"

    # the old hidden alias must be gone: it should fall through as unknown
    picked.clear()
    C._run_command("/mcp-pick", _S())
    assert picked == [], f"/mcp-pick should be unknown now, opened {picked}"


def test_turn_errors_become_a_message_not_a_traceback():
    """The REPL only caught KeyboardInterrupt, so an upstream 429 raised out of
    app() and printed a traceback instead of continuing the session."""
    import inspect

    import minicode.cli as C

    src = inspect.getsource(C.run)
    # the turn body must have a general handler
    assert "except Exception" in src, "turn errors are not caught"
    assert "request failed" in src, "turn errors are not reported to the user"


def test_statusbar_puts_hints_left_and_meter_right():
    from minicode.cli import _statusbar

    lines = _render(_statusbar, 120, _FakeSide())
    bar = lines[-1]
    assert "ctrl+k commands" in bar
    assert "ctrl+p" not in bar, "ctrl+p is not a binding; advertising it is a lie"
    assert "ctrl+o expand" in bar


def test_statusbar_shows_the_token_meter():
    from minicode.cli import CONTEXT_WINDOW, _statusbar

    lines = _render(_statusbar, 120, _FakeSide())
    bar = lines[-1]
    assert "98.6k" in bar, bar
    want = int(_FakeSide().usage.total_tokens * 100 / CONTEXT_WINDOW)
    assert f"({want}%)" in bar, bar


def test_statusbar_never_overflows():
    from minicode.cli import _statusbar

    for width in (40, 60, 100):
        bar = _render(_statusbar, width, _FakeSide())[-1]
        assert len(bar.rstrip()) <= width, f"overflows at {width}"


def test_statusbar_survives_an_empty_session():
    from minicode.cli import _statusbar
    from minicode.usage import Usage

    class Empty:
        usage = Usage()

    bar = _render(_statusbar, 120, Empty())[-1]
    assert "ctrl+k" in bar


def test_system_prompt_teaches_virtual_paths(tmp_path: Path):
    """Regression: the model had no idea files are addressed as /src/a.py, so it
    guessed C:\\... and every file tool call was rejected."""
    from minicode.agent import PLAN_PROMPT, SYSTEM_PROMPT

    for prompt in (SYSTEM_PROMPT, PLAN_PROMPT):
        assert "virtual path" in prompt.lower(), prompt[:200]
        assert "/" in prompt


def test_system_prompt_warns_against_drive_letters(tmp_path: Path):
    from minicode.agent import SYSTEM_PROMPT

    low = SYSTEM_PROMPT.lower()
    assert "drive" in low or "c:\\\\" in low or "c:/" in low
    assert "ls" in low, "should tell the model how to discover the project"


def test_plan_prompt_also_teaches_virtual_paths(tmp_path: Path):
    from minicode.agent import PLAN_PROMPT

    assert "virtual path" in PLAN_PROMPT.lower(), PLAN_PROMPT


def test_root_dir_confines_filesystem(tmp_path: Path):
    backend = LocalShellBackend(root_dir=str(tmp_path))
    outside = tmp_path.parent / "escape.txt"
    for bad in ("../escape.txt", str(outside)):
        with pytest.raises(ValueError):
            backend.write(bad, "nope")
    assert not outside.exists()
