"""Slash-command completion."""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from minicode.complete import COMMANDS, SlashCompleter


def _meta(c) -> str:
    """display_meta is a str, a (style, text) tuple, or FormattedText."""
    m = c.display_meta
    if not m:
        return ""
    if isinstance(m, str):
        return m
    if isinstance(m, tuple):
        return str(m[1])
    return "".join(part for _, part in m)


def completions(completer: SlashCompleter, text: str) -> list[tuple[str, str]]:
    return [(c.text, _meta(c)) for c in completer.get_completions(Document(text, len(text)), None)]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Isolated skill + config dirs so the developer's real ones do not leak in."""
    from minicode import skills as S
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(S, "USER_DIRS", ())
    return tmp_path


def _skill(root: Path, name: str, desc: str) -> None:
    d = root / ".thirtyagent" / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {desc}\n---\n", encoding="utf-8")


def test_slash_lists_commands(root):
    got = [t for t, _ in completions(SlashCompleter(root), "/")]
    assert "/model" in got
    assert "/skills" in got
    assert "/plan" in got


def test_prefix_filters_commands(root):
    got = [t for t, _ in completions(SlashCompleter(root), "/mo")]
    assert "/model" in got
    assert "/model".startswith("/mo")
    assert all(t.startswith("/mo") for t in got), got


def test_slash_prefix_matching_is_not_mid_token(root):
    """ "hello /mo" must not offer commands -- completion is line-start only."""
    assert completions(SlashCompleter(root), "hello /mo") == []


def test_plain_text_gives_nothing(root):
    assert completions(SlashCompleter(root), "fix the bug in") == []


def test_skill_completion_lists_every_skill(root):
    _skill(root, "release-check", "Use when releasing.")
    _skill(root, "frontend-design", "Distinctive visual design.")
    got = [t for t, _ in completions(SlashCompleter(root), "/skill:")]
    assert "/skill:release-check" in got
    assert "/skill:frontend-design" in got


def test_skill_completion_filters_by_name(root):
    _skill(root, "release-check", "a")
    _skill(root, "frontend-design", "b")
    got = [t for t, _ in completions(SlashCompleter(root), "/skill:fr")]
    assert got == ["/skill:frontend-design"], got


def test_skill_completion_carries_the_description(root):
    _skill(root, "release-check", "Use when preparing or reviewing a release.")
    got = completions(SlashCompleter(root), "/skill:rel")
    assert got[0][0] == "/skill:release-check"
    assert "release" in got[0][1]


def test_bare_slash_skill_also_completes(root):
    """ "/skill" should still suggest skills, not dead-end."""
    _skill(root, "release-check", "x")
    got = [t for t, _ in completions(SlashCompleter(root), "/skill")]
    assert "/skill:release-check" in got


def test_model_aliases_complete_after_space(root):
    from minicode.config import load, save

    cfg = load()
    cfg["models"] = {
        "big": {"base_url": "u", "api_key": "k", "model": "big-model"},
        "small": {"base_url": "u", "api_key": "k", "model": "small-model"},
    }
    save(cfg)

    got = [t for t, _ in completions(SlashCompleter(root), "/model ")]
    assert "big" in got and "small" in got, got


def test_subcommands_complete_after_a_command(root):
    got = [t for t, _ in completions(SlashCompleter(root), "/mcp ")]
    assert set(got) == {"add", "test", "rm"}, got

    got = [t for t, _ in completions(SlashCompleter(root), "/model ")]
    assert "add" in got and "list" in got


def test_mcp_arg_slot_does_not_offer_model_aliases(root):
    """Scoping: /mcp should not list models, and vice versa."""
    from minicode.config import load, save

    cfg = load()
    cfg["models"] = {"big": {"base_url": "u", "api_key": "k", "model": "m"}}
    cfg["mcp"] = {"filesys": {"command": "npx"}}
    save(cfg)

    mcp_slot = [t for t, _ in completions(SlashCompleter(root), "/mcp ")]
    assert "filesys" in mcp_slot
    assert "big" not in mcp_slot, f"models leaked into /mcp: {mcp_slot}"

    model_slot = [t for t, _ in completions(SlashCompleter(root), "/model ")]
    assert "big" in model_slot
    assert "filesys" not in model_slot, f"mcp servers leaked into /model: {model_slot}"


def test_argument_slot_never_offers_commands(root):
    """Once a command is typed, its arguments are not other commands."""
    got = [t for t, _ in completions(SlashCompleter(root), "/model ")]
    assert not any(t.startswith("/") for t in got), got


def test_completer_survives_a_broken_config(root, monkeypatch):
    """A bad config must not make the prompt unusable."""

    (root / "cfg").mkdir(parents=True, exist_ok=True)
    (root / "cfg" / "config.json").write_text("{not json", encoding="utf-8")
    c = SlashCompleter(root)
    assert c.skills() == {} or isinstance(c.skills(), dict)
    assert completions(c, "/")  # still completes commands


def test_every_command_is_documented():
    """No command in the completer that /help does not mention."""
    from minicode.cli import HELP

    for name in COMMANDS:
        assert name in HELP, f"{name} completes but is missing from /help"


# --- wiring into the prompt ---


def test_prompt_uses_completion_when_tty(monkeypatch, tmp_path):
    """A real terminal must get a prompt_toolkit session, not console.input."""
    import minicode.cli as C

    monkeypatch.setattr(C, "_session", None)
    monkeypatch.setattr(type(C.console), "is_terminal", property(lambda self: True))

    made = {}

    class FakeSession:
        def __init__(self, root):
            made["root"] = root

        def prompt(self, text):
            made["prompt_text"] = text
            return "/skill:release-check"

    monkeypatch.setattr(C, "_make_session", lambda root: FakeSession(root))
    assert C._read_prompt(tmp_path) == "/skill:release-check"
    assert made["root"] == tmp_path
    assert C._session is not None, "session was not cached"


def test_prompt_falls_back_when_piped(monkeypatch, tmp_path):
    """Piped stdin must not try to build a console UI."""
    import minicode.cli as C

    monkeypatch.setattr(type(C.console), "is_terminal", property(lambda self: False))

    def boom(root):
        raise AssertionError("prompt_toolkit must not be used when stdout is a pipe")

    monkeypatch.setattr(C, "_make_session", boom)
    monkeypatch.setattr(C.console, "input", lambda *_: "/plan")

    assert C._read_prompt(tmp_path) == "/plan"
    assert C._session is None, "a session leaked into piped mode"
