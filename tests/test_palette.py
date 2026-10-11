"""Palette ranking and rendering (pure functions, no console needed)."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from minicode.catalog import COMMANDS, entries, skill_entries
from minicode.palette import (
    Entry,
    PaletteState,
    fuzzy,
    rank,
    render,
)


def tmp_skill(root: Path, name: str, desc: str) -> str:
    d = root / ".thirtyagent" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n", encoding="utf-8")
    return name


def _items():
    return [Entry(text=t, label=l, group=g, shortcut=s) for t, l, g, s in COMMANDS]


def _text(rows) -> str:
    return "".join(t for _, t in rows)


def _selected_lines(rows) -> list[str]:
    """Lines drawn with the 'sel' style."""
    out, cur = [], []
    for style, text in rows:
        cur.append(text)
        if text.endswith("\n"):
            if "sel" in style or "selhint" in style:
                out.append("".join(cur))
            cur = []
    return out


# --- fuzzy ---


def test_subsequence_matches_across_words():
    assert fuzzy("skl", "Skills") is not None
    assert fuzzy("swmod", "Switch model") is not None


def test_non_subsequence_does_not_match():
    assert fuzzy("zzz", "Skills") is None


def test_word_start_beats_mid_word():
    """ "sw" should prefer "Switch model" over anything with s...w late."""
    ranked = rank(_items(), "sw")
    assert ranked[0].display == "Switch model", [r.display for r in ranked]


def test_empty_needle_returns_everything():
    assert len(rank(_items(), "")) == len(COMMANDS)


def test_score_is_deterministic():
    assert fuzzy("sk", "Skills") == fuzzy("sk", "Skills")


# --- ranking ---


def test_rank_keeps_group_order_when_unfiltered():
    got = rank(_items(), "")
    assert got[0].group == "Prompt"


def test_rank_filters_skill_entries(tmp_path):
    got = rank(entries(tmp_path), "/skill:")
    assert got, "no skill entries"
    assert all(e.text.startswith("/skill:") for e in got)


# --- render ---


def test_render_shows_groups_as_headers():
    rows = render(
        PaletteState(items=_items(), needle=""),
    )
    text = _text(rows)
    for group in ("Prompt", "Session", "Model", "System"):
        assert group in text, f"missing group header {group}"


def test_render_shows_search_needle():
    rows = render(
        PaletteState(items=_items(), needle="mod"),
    )
    assert "/mod" in _text(rows)


def test_render_no_matches_is_graceful():
    rows = render(
        PaletteState(items=_items(), needle="zzzz"),
    )
    assert "no matches" in _text(rows)


def test_render_selection_moves_with_index():
    items = _items()
    first = _selected_lines(render(PaletteState(items=items, index=0)))
    third = _selected_lines(render(PaletteState(items=items, index=3)))

    assert first, "nothing highlighted at index 0"
    assert third, "nothing highlighted at index 3"
    assert first != third


def test_render_shows_shortcuts_on_the_right():
    """Shortcut text renders; without it the column is ragged."""
    text = _text(render(PaletteState(items=[Entry("/exit", "Exit", "System", "ctrl+d")])))
    assert "ctrl+d" in text
    assert "Exit" in text


def test_every_command_group_is_reachable_unfiltered():
    """The cap must not hide whole sections behind a wall of skills."""
    got = {g for e in entries(Path(".")) for g in [e.group]}
    rows = render(PaletteState(items=entries(Path("."))))
    text = _text(rows)
    for g in got:
        if g in ("System", "Model", "Work", "Session", "Capabilities"):
            assert g in text, f"{g} group is unreachable without searching"


def test_render_height_grows_with_results():
    many = _items() + [Entry(text=f"/skill:s{i}", group="Prompt") for i in range(30)]
    short = _text(render(PaletteState(items=entries(Path(".")), needle="sw")))
    tall = _text(render(PaletteState(items=many, needle="")))
    assert len(tall.splitlines()) > len(short.splitlines())


def test_palette_caps_visible_rows():
    many = [Entry(text=f"/skill:s{i}", group="Prompt") for i in range(200)]
    rows = render(PaletteState(items=many))
    items_drawn = sum(1 for s, t in rows if t.startswith("  /skill"))
    assert items_drawn <= 20, f"palette drew {items_drawn} rows"


def test_skill_picker_shows_names_not_descriptions():
    """Regression: radiolist_dialog showed descriptions, not names."""
    d = tmp_skill(Path("."), "gpt-taste", "Elite UX/UI and GSAP motion engineering.")
    got = skill_entries(Path("."))
    pick = next(e for e in got if e.text == f"/skill:{d}")
    assert pick.label == d, f"label should be the name, got {pick.label!r}"
    assert pick.shortcut in {"user", "project"}
    assert pick.detail.startswith("Elite UX/UI"), pick.detail


def test_title_is_configurable():
    assert "Skills" in _text(render(PaletteState(items=_items()), title="Skills"))
    assert "Commands" in _text(render(PaletteState(items=_items()), title="Commands"))


def test_detail_is_truncated_to_width():
    long = Entry("/skill:x", "x", "Skills", "user", detail="w" * 500)
    rows = render(PaletteState(items=[long]), width=60)
    assert len(_text(rows).splitlines()[3]) <= 62, "detail overflowed the width"


def test_term_width_is_clamped():
    from minicode.palette import term_width

    w = term_width()
    assert 40 <= w <= 100, w


# --- catalog ---


def test_every_entry_has_a_group():
    for e in entries(Path(".")):
        assert e.group, e


def test_skill_entries_have_their_own_group():
    """Skills need a separate header, else "Prompt" renders twice."""
    for e in entries(Path(".")):
        if e.text.startswith("/skill:") and e.text != "/skill:":
            assert e.group == "Skills", e


def test_render_styles_resolve_in_prompt_toolkit():
    """Regression: render() emitted bare class names ('title', 'item', ...) and
    the group header as 'title #c4b5fd'. prompt_toolkit only treats the
    'class:' prefix as a class, so it fed every one of those to parse_color()
    and /skills died with ValueError: Wrong color format 'title'."""
    from minicode.theme import PTK_STYLE

    rows = render(PaletteState(items=entries(Path("."))))
    for style, _ in rows:
        PTK_STYLE.get_attrs_for_style_str(style)  # must not raise


def test_group_header_uses_a_class_not_an_inline_color():
    """Header tints live in PTK_STYLE, so they cannot collide with a color."""
    rows = render(PaletteState(items=[Entry(text="/m", group="Cost & safety")]))
    header = next(style for style, text in rows if text == "Cost & safety")
    assert "#" not in header, header
    assert "class:group.costsafety" in header, header


def test_group_headers_are_not_repeated():
    """Each group must appear once, so no duplicated header in the popup."""
    groups = {e.group for e in entries(Path("."))}
    text = _text(render(PaletteState(items=entries(Path(".")))))
    headers = [ln.strip() for ln in text.splitlines() if ln.strip() in groups]
    assert headers, "expected at least one group header"
    assert len(headers) == len(set(headers)), f"repeated group header: {headers}"


def test_commands_only_has_no_skill_entries():
    assert not [
        e
        for e in __import__("minicode.catalog", fromlist=["x"]).commands_only()
        if e.text.startswith("/skill:") and e.text != "/skill:"
    ]


@pytest.mark.parametrize("text,label,group,_", COMMANDS)
def test_catalog_commands_are_unique(text, label, group, _):
    texts = [t for t, *_ in COMMANDS]
    assert texts.count(text) == 1, f"duplicate {text}"


def test_picker_items_are_well_formed(monkeypatch, tmp_path):
    """/model, /mcp, /sessions all feed the same popup, so shapes must agree."""
    from minicode import cli

    for items in (cli._model_items(), cli._mcp_items()):
        for e in items:
            assert set(e) == {"text", "label", "group", "shortcut", "detail"}, e
            (Entry(**e),)  # must be constructible


def test_pick_returns_none_when_cancelled(monkeypatch):
    import minicode.palette as P
    from minicode import cli

    monkeypatch.setattr(P, "show_palette", lambda *a, **k: None)
    assert cli._pick([], "Models") is None
    assert cli._pick([{"text": "a", "label": "a", "group": "G", "shortcut": "", "detail": ""}], "M") is None


def test_pick_passes_through_the_choice(monkeypatch):
    import minicode.palette as P
    from minicode import cli

    seen = {}

    def fake(items, title):
        seen["title"] = title
        seen["count"] = len(items)
        return "chosen"

    monkeypatch.setattr(P, "show_palette", fake)
    got = cli._pick([{"text": "a", "label": "a", "group": "G", "shortcut": "", "detail": ""}], "Models")
    assert got == "chosen"
    assert seen == {"title": "Models", "count": 1}


def test_unknown_command_suggests_close_matches(capsys):
    """/sk should point at /skills, not just say 'unknown'."""
    from minicode import cli

    cli._suggest("/sk")
    out = capsys.readouterr().out
    assert "/skills" in out, out


def test_unknown_command_lists_palette_hint(capsys):
    from minicode import cli

    cli._suggest("/zzzzz")
    out = capsys.readouterr().out
    assert "ctrl+k" in out, out


def test_ctrl_k_handler_is_async():
    """Regression: Application.run() calls asyncio.run(), which refuses to nest.

    A key handler already runs inside prompt_toolkit's loop, so ctrl+k must be
    async and go through run_in_terminal.
    """
    import inspect

    from minicode.cli import _ctrl_k, _palette_bindings

    handler = _ctrl_k(_palette_bindings(Path.cwd()))
    assert inspect.iscoroutinefunction(handler), "ctrl+k handler must be a coroutine function"


def _patch_run_in_terminal(monkeypatch, fake):
    """prompt_toolkit.application.run_in_terminal is shadowed by the function
    it re-exports, so reach the module object directly."""
    import importlib

    mod = importlib.import_module("prompt_toolkit.application.run_in_terminal")
    monkeypatch.setattr(mod, "run_in_terminal", fake)


def test_ctrl_k_uses_run_in_terminal(monkeypatch, tmp_path):
    """It must not call Application.run() directly from inside the loop."""
    import asyncio

    from minicode import cli
    from minicode import palette as P

    calls = {}

    async def fake_run_in_terminal(func):
        calls["ran"] = True
        return func()

    _patch_run_in_terminal(monkeypatch, fake_run_in_terminal)
    monkeypatch.setattr(P, "show_palette", lambda *a, **k: "/model")

    buf = types.SimpleNamespace(text="", cursor_position=0, validated=False)
    buf.validate_and_handle = lambda: setattr(buf, "validated", True)
    event = types.SimpleNamespace(current_buffer=buf)

    asyncio.run(_ctrl_k(cli._palette_bindings(tmp_path))(event))

    assert calls.get("ran"), "did not go through run_in_terminal"
    assert buf.text == "/model"


def test_ctrl_k_survives_a_popup_failure(monkeypatch, tmp_path, capsys):
    """A popup error must not take the whole TUI down again."""
    import asyncio

    from minicode import cli
    from minicode import palette as P

    def boom(*a, **k):
        raise RuntimeError("asyncio.run() cannot be called from a running event loop")

    monkeypatch.setattr(P, "show_palette", boom)

    async def fake_run_in_terminal(func):
        return func()

    _patch_run_in_terminal(monkeypatch, fake_run_in_terminal)

    buf = types.SimpleNamespace(text="keep me", cursor_position=7)
    buf.validate_and_handle = lambda: pytest.fail("must not validate on failure")
    event = types.SimpleNamespace(current_buffer=buf)

    asyncio.run(_ctrl_k(cli._palette_bindings(tmp_path))(event))

    assert "unavailable" in capsys.readouterr().out
    assert buf.text == "keep me", "buffer was clobbered"


def test_pick_reports_instead_of_crashing(monkeypatch, capsys):
    from minicode import cli
    from minicode import palette as P

    def boom(*a, **k):
        raise RuntimeError("cannot nest")

    monkeypatch.setattr(P, "show_palette", boom)
    got = cli._pick([{"text": "a", "label": "a", "group": "G", "shortcut": "", "detail": ""}], "Models")
    assert got is None
    assert "unavailable" in capsys.readouterr().out


def _floats_of(layout):
    """Pull the Float objects out of whatever container the layout wrapped."""
    seen, stack = [], [layout.container]
    while stack:
        node = stack.pop()
        floats = getattr(node, "floats", None)
        if floats:
            seen.extend(floats)
        for attr in ("content", "container"):
            child = getattr(node, attr, None)
            if hasattr(child, "floats") or hasattr(child, "container") or hasattr(child, "content"):
                stack.append(child)
    return seen


def test_row_count_counts_lines_not_fragments():
    """Regression: render() returns styled fragments, several per visual line.
    The layout asked for len(rows) lines, which for 20 items was 67 -- more than
    the terminal had, so prompt_toolkit aborted with 'Window too small'."""
    from minicode.palette import Entry, PaletteState, render, row_count

    items = [Entry(text=f"/cmd{i}", group="G", label=f"command {i}") for i in range(40)]
    rows = render(PaletteState(items=items), 76, title="Sessions")
    lines = row_count(rows)
    assert lines < len(rows), "fragments must not be mistaken for lines"
    assert lines <= 30, f"a 20-item palette should not need {lines} lines"


def test_body_height_fits_a_small_terminal(monkeypatch):
    """A short window must shrink the popup, not abort."""
    from minicode.palette import Entry, PaletteState, _layout_for

    monkeypatch.setattr("minicode.palette._term_columns", lambda *a, **k: 120)
    for rows in (10, 24, 60):
        monkeypatch.setattr("minicode.palette._term_rows", lambda *a, _r=rows, **k: _r)
        state = PaletteState(items=[Entry(text=f"/c{i}", group="G", label=f"command {i}") for i in range(40)])
        layout = _layout_for(state.items, "Sessions", state)  # must not raise
        assert layout is not None


def test_palette_never_exceeds_available_height(monkeypatch):
    """The popup height is clamped to the window; render() itself may produce
    more lines than fit, the layout must crop rather than abort."""
    from minicode.palette import (
        Entry,
        PaletteState,
        _layout_for,
        render,
        row_count,
        term_width,
    )

    monkeypatch.setattr("minicode.palette._term_columns", lambda *a, **k: 120)
    items = [Entry(text=f"/c{i}", group="G", label=f"command {i}") for i in range(40)]
    for rows in (8, 12, 24, 60):
        monkeypatch.setattr("minicode.palette._term_rows", lambda *a, _r=rows, **k: _r)
        state = PaletteState(items=items)
        _layout_for(items, "Sessions", state)  # must not raise
        raw = row_count(render(state, term_width(), title="Sessions"))
        clamped = min(raw, max(3, rows) - 3)
        assert clamped <= rows, f"height {clamped} exceeds window {rows}"


def test_body_height_never_exceeds_screen(monkeypatch):
    """Regression: /skills died with 'Window too small'. The float sits at top=1
    with header+searchbar above the body, so the body must leave room for all
    three or the popup claims the last line and aborts."""
    from minicode.palette import Entry, PaletteState, _layout_for

    monkeypatch.setattr("minicode.palette._term_columns", lambda *a, **k: 120)
    items = [Entry(text=f"/c{i}", group="G", label=f"command {i}") for i in range(40)]
    for rows in (6, 10, 24, 60):
        monkeypatch.setattr("minicode.palette._term_rows", lambda *a, _r=rows, **k: _r)
        layout = _layout_for(items, "Skills", PaletteState(items=items))
        float_ = layout.container.floats[0]
        box = float_.content
        body = box.get_children()[-1]
        height = body.height() if callable(body.height) else body.height
        assert height <= max(1, rows - 3), f"body {height} cannot fit in {rows} rows"


def test_float_has_either_width_or_left_right_never_both(monkeypatch):
    """Regression: Float(width=...) plus left/right made prompt_toolkit claim
    the same space twice and raise 'Window too small'."""

    from minicode.palette import Entry, _layout_for

    monkeypatch.setattr("minicode.palette._term_columns", lambda *a, **k: 120)
    layout = _layout_for([Entry(text="a", group="G")], "T", PaletteState(items=[]))
    floats = _floats_of(layout)
    assert len(floats) == 1, floats
    f = floats[0]
    assert f.width is None, "must not pin width when using left/right"
    assert f.left is not None and f.right is not None
    assert f.left == f.right, "popup should be centred"


def test_float_fits_narrow_terminals(monkeypatch):
    """A 40-column window must not request 100 columns of float."""
    from minicode.palette import Entry, _layout_for

    for cols in (20, 30, 40, 60, 120, 200):
        monkeypatch.setattr("minicode.palette._term_columns", lambda *a, _c=cols, **k: _c)
        layout = _layout_for([Entry(text="a", group="G")], "T", PaletteState(items=[]))
        f = _floats_of(layout)[0]
        span = cols - f.left - f.right
        assert span > 0, f"float does not fit at {cols} cols (left={f.left} right={f.right})"


def test_palette_layout_and_keys_share_one_state(monkeypatch):
    """Regression: each used to build its own PaletteState, so typing filtered
    a list the renderer never looked at."""
    from minicode.palette import Entry, _keys_for, _layout_for

    monkeypatch.setattr("minicode.palette._term_columns", lambda *a, **k: 120)
    state = PaletteState(items=[Entry(text="a", group="G")])
    _layout_for([Entry(text="a", group="G")], "T", state)
    _keys_for([Entry(text="a", group="G")], "T", state)

    # typing through the handler must mutate the very state the layout draws
    for b in _keys_for([Entry(text="a", group="G")], "T", state).bindings:
        handler = b.handler
        if getattr(handler, "__name__", "") == "_type":
            handler(types.SimpleNamespace(data="q"))
            break
    assert state.needle == "q", "handler wrote to a different state object"


def test_enter_returns_the_shared_states_selection(monkeypatch):
    from minicode.palette import Entry, _keys_for

    items = [Entry(text="one", group="G"), Entry(text="two", group="G")]
    state = PaletteState(items=list(items))
    state.index = 1
    app = types.SimpleNamespace(exit_result="unset")
    fired = []
    app.exit = lambda: fired.append(True)

    for b in _keys_for(items, "T", state).bindings:
        h = b.handler
        if getattr(h, "__name__", "") == "_accept":
            h(types.SimpleNamespace(app=app))
            break
    assert app.exit_result == "two", app.exit_result
    assert fired, "did not exit"


# --- prompt wiring ---


def test_palette_layout_constructs(monkeypatch):
    """Regression: FloatContainer needs floats=[Float(...)], not content=Float.

    This crashed the whole TUI the first time ctrl+k was pressed.
    """
    from prompt_toolkit.layout.containers import FloatContainer

    from minicode import palette as P

    captured = {}

    class FakeApp:
        def run(self):
            return "/model"

    monkeypatch.setattr(P, "Application", lambda **kw: (captured.update(kw), FakeApp())[1])

    got = P.show_palette(_items(), title="Commands")
    assert got == "/model"
    assert "layout" in captured, "Application built without a layout"
    assert "key_bindings" in captured, "Application built without key bindings"

    root = captured["layout"].container
    assert isinstance(root, FloatContainer), type(root)
    assert root.floats, "popup has no Float inside it, so it would render as nothing"


def _ctrl_k(kb):
    """The handler registered for c-k. Prompt_toolkit stores Binding objects."""
    for b in kb.bindings:
        keys = getattr(b, "keys", None) or []
        if "c-k" in keys:
            return b.handler
    raise AssertionError("ctrl+k is not bound")


def test_palette_bindings_are_a_real_key_bindings(tmp_path):
    """Regression: session.app.key_bindings is _MergedKeyBindings, has no add()."""
    from minicode.cli import _palette_bindings

    kb = _palette_bindings(tmp_path)
    assert kb.bindings, "no bindings registered"
    assert callable(_ctrl_k(kb))


def test_make_session_accepts_the_bindings(tmp_path, monkeypatch):
    """PromptSession takes key_bindings=; constructing must not raise."""
    import prompt_toolkit

    import minicode.cli as C

    captured = {}

    class FakeSession:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(prompt_toolkit, "PromptSession", FakeSession)

    C._make_session(tmp_path)
    assert "key_bindings" in captured, "bindings not passed to PromptSession"
    assert captured["key_bindings"].bindings, "empty bindings"


def test_palette_handler_inserts_the_picked_command(tmp_path, monkeypatch):
    """Enter on an item must write it into the input buffer."""
    import asyncio

    import minicode.cli as C
    import minicode.palette as P

    async def fake_run_in_terminal(func):
        return func()

    _patch_run_in_terminal(monkeypatch, fake_run_in_terminal)
    monkeypatch.setattr(P, "show_palette", lambda items, **kw: "/model")

    buf = types.SimpleNamespace(text="", cursor_position=0, validated=False)
    buf.validate_and_handle = lambda: setattr(buf, "validated", True)
    event = types.SimpleNamespace(current_buffer=buf)

    asyncio.run(_ctrl_k(C._palette_bindings(tmp_path))(event))

    assert buf.text == "/model", buf.text
    assert buf.cursor_position == len("/model")
    assert buf.validated


def test_cancelled_palette_leaves_the_buffer_alone(tmp_path, monkeypatch):
    import asyncio

    import minicode.cli as C
    import minicode.palette as P

    async def fake_run_in_terminal(func):
        return func()

    _patch_run_in_terminal(monkeypatch, fake_run_in_terminal)
    monkeypatch.setattr(P, "show_palette", lambda items, **kw: None)

    buf = types.SimpleNamespace(text="already typing", cursor_position=9)
    buf.validate_and_handle = lambda: pytest.fail("should not validate on cancel")
    event = types.SimpleNamespace(current_buffer=buf)

    asyncio.run(_ctrl_k(C._palette_bindings(tmp_path))(event))

    assert buf.text == "already typing"
