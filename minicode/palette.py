"""Grouped, fuzzy-searchable command palette (opencode-style popup)."""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.containers import Float, FloatContainer
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from .theme import PTK_STYLE, group_class

STYLE = PTK_STYLE

MAX_VISIBLE = 20
DEFAULT_WIDTH = 76


def _term_columns(default: int = 100) -> int:
    try:
        import shutil

        return shutil.get_terminal_size((default, 24)).columns
    except Exception:  # noqa: BLE001
        return default


def _term_rows(default: int = 24) -> int:
    """Rows prompt_toolkit will actually render into.

    shutil alone lied on Windows (it reported a size the renderer never used),
    so the Float asked for more height than the screen had and prompt_toolkit
    aborted with "Window too small".
    """
    rows = _ptk_rows()
    if rows:
        return rows
    return _shutil_rows(default)


def _ptk_rows() -> int:
    """0 when there is no real output to measure."""
    try:
        from prompt_toolkit.application import get_app

        rows = get_app().output.get_size().rows
        return rows if rows and rows > 0 else 0
    except Exception:  # noqa: BLE001 - DummyApplication, no terminal, etc.
        return 0


def _shutil_rows(default: int) -> int:
    try:
        import shutil

        return shutil.get_terminal_size((100, default)).lines
    except Exception:  # noqa: BLE001
        return default


def term_width(default: int = DEFAULT_WIDTH) -> int:
    """Terminal columns, clamped so the popup never wraps or eats the screen."""
    return max(40, min(_term_columns(default), 100) - 4)


@dataclass(frozen=True)
class Entry:
    text: str
    group: str
    label: str = ""
    shortcut: str = ""
    detail: str = ""

    @property
    def display(self) -> str:
        return self.label or self.text


@dataclass
class PaletteState:
    needle: str = ""
    index: int = 0
    items: list = field(default_factory=list)

    def visible(self) -> list[Entry]:
        return rank(self.items, self.needle)[:MAX_VISIBLE]


def fuzzy(needle: str, haystack: str) -> float | None:
    """Subsequence score, lower is better. None means no match."""
    if not needle:
        return 0.0
    n, h = needle.lower(), haystack.lower()
    score, hi, run = 0.0, 0, 0
    for ch in n:
        found = h.find(ch, hi)
        if found < 0:
            return None
        score += found - hi
        if found == 0 or h[found - 1] in " /-_":
            score -= 3.0
        run = run + 1 if found == hi else 0
        score -= run
        hi = found + 1
    return max(score, 0.0) + len(h) * 0.01


def rank(items: list[Entry], needle: str) -> list[Entry]:
    """Filter by fuzzy match; ties keep declaration order."""
    needle = needle.strip()
    if not needle:
        return list(items)
    scored = [(fuzzy(needle, i.display), n, i) for n, i in enumerate(items)]
    scored = [(s, n, i) for s, n, i in scored if s is not None]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [i for _, _, i in scored]


def render(state: PaletteState, width: int = DEFAULT_WIDTH, title: str = "Commands") -> list[tuple[str, str]]:
    """Pure renderer -> [(style, text)]. Unit tested directly."""
    items = state.visible()
    rows: list[tuple[str, str]] = [("class:title", " " + title)]
    rows.append(("class:esc", " " * max(1, width - len(title) - 4) + "esc"))
    rows.append(("class:item", "\n"))
    rows.append(("class:cursor", " /" + state.needle))
    rows.append(("class:item", "\n\n"))

    if not items:
        rows.append(("class:hint", "no matches"))
        return rows

    name_col = max(len(i.display) for i in items) + 3
    short_col = name_col + max((len(i.shortcut) for i in items if i.shortcut), default=0) + 2
    last = None
    for n, e in enumerate(items):
        if e.group != last:
            if last is not None:
                rows.append(("class:item", "\n"))
            rows.append((f"class:title class:{group_class(e.group)}", e.group))
            rows.append(("class:item", "\n"))
            last = e.group

        sel = n == state.index
        item_style = "class:sel" if sel else "class:item"
        hint_style = "class:selhint" if sel else "class:hint"

        rows.append((item_style, "  " + e.display))
        rows.append((item_style, " " * max(1, name_col - len(e.display) - 2)))
        if e.shortcut:
            rows.append((hint_style, e.shortcut))
            rows.append((item_style, " " * max(1, short_col - name_col - len(e.shortcut))))
        if e.detail:
            room = width - short_col - 2
            if room > 8:
                rows.append((hint_style, e.detail[:room]))
        rows.append((item_style, "\n"))

    return rows


def row_count(rows: list[tuple[str, str]]) -> int:
    """How many terminal lines a rendered row list actually occupies.

    render() returns styled fragments, several per visual line -- counting the
    list gave 67 lines for a 20-item palette, and the Float then asked for more
    height than the terminal had, which prompt_toolkit calls "Window too small".
    """
    return sum(text.count("\n") for _, text in rows) + 1


def _ft(rows: list[tuple[str, str]]) -> FormattedText:
    return FormattedText(rows)


def _layout_for(items: list[Entry], title: str, state: PaletteState) -> Layout:
    """The popup, as a Float inside a FloatContainer.

    Takes the shared state: a private one here would render a list the key
    handlers never update.
    """
    width = term_width()

    def header() -> FormattedText:
        return _ft(
            [
                ("class:title", f" {title}"),
                ("class:esc", " " * max(1, width - len(title) - 4) + "esc"),
            ]
        )

    def searchbar() -> FormattedText:
        return _ft([("class:cursor", " /" + state.needle + "â–")])

    def body() -> FormattedText:
        return _ft(render(state, width, title=title))

    def body_height() -> Dimension:
        # prompt_toolkit fails the whole popup when the children's *minimum*
        # heights exceed the space (containers.py: sum_dimensions.min > height).
        # min=1 makes that impossible however the terminal size is reported;
        # preferred still tracks the filtered list and max keeps the box from
        # running past the bottom of the screen.
        room = max(1, _term_rows() - 4)
        return Dimension(
            min=1,
            max=room,
            preferred=min(row_count(render(state, width, title=title)), room),
        )

    box = HSplit(
        [
            Window(content=FormattedTextControl(header), height=1),
            Window(content=FormattedTextControl(searchbar), height=1),
            Window(
                content=FormattedTextControl(body),
                # callable so the box tracks the filtered result set; the clamp is
                # what keeps the float inside the window ("Window too small")
                height=body_height,
                always_hide_cursor=True,
                wrap_lines=False,
            ),
        ]
    )
    # Give the Float either width OR left/right, never both: prompt_toolkit
    # treats them as independent claims on the same space and raises
    # "Window too small" when they overlap. Width alone would also pin it to
    # the left edge, so we derive the width from the available columns here and
    # pass left/right only.
    cols = max(20, _term_columns())
    float_width = min(width, cols - 4)
    inset = max(0, (cols - float_width) // 2)
    return Layout(
        FloatContainer(
            content=Window(),  # background behind the popup
            floats=[
                Float(
                    content=box,
                    top=1,
                    left=inset,
                    right=inset,
                )
            ],
        )
    )


def _keys_for(items: list[Entry], title: str, state: PaletteState) -> KeyBindings:
    """Navigation, filtering, and accept/cancel."""
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event) -> None:
        state.index = max(0, state.index - 1)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event) -> None:
        state.index = min(len(state.visible()) - 1, state.index + 1)

    @kb.add("backspace")
    def _backspace(event) -> None:
        state.needle = state.needle[:-1]
        state.index = 0

    @kb.add("c-w")
    def _clear(event) -> None:
        state.needle = ""
        state.index = 0

    @kb.add("enter")
    def _accept(event) -> None:
        visible = state.visible()
        event.app.exit_result = visible[min(state.index, len(visible) - 1)].text if visible else None
        event.app.exit()

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event) -> None:
        event.app.exit_result = None
        event.app.exit()

    @kb.add("<any>")
    def _type(event) -> None:
        state.needle += event.data
        state.index = 0

    return kb


def show_palette(items: list[Entry], title: str = "Commands") -> str | None:
    """Modal popup. Returns the chosen text, or None if cancelled."""
    if not items:
        return None
    state = PaletteState(items=list(items))
    app = Application(
        layout=_layout_for(items, title, state),
        key_bindings=_keys_for(items, title, state),
        style=STYLE,
        full_screen=False,
        mouse_support=False,
    )
    return app.run()
