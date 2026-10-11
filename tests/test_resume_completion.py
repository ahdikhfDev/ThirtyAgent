"""Session-id completion for /resume.

The bug this covers: complete.py imported CONFIG_DIR from .config, which never
existed there. The ImportError was swallowed by a bare except, so /resume
silently offered nothing, forever, with no test to notice.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.document import Document

from minicode.complete import SlashCompleter
from minicode.sessions import Store


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Isolated config dir so the developer's real sessions.db never leaks in."""
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    return tmp_path


def _session(cfg, store: Store, sid: str, title: str) -> None:
    store.create(sid, cfg, title=title)


def test_sessions_are_listed_for_the_current_cwd(cfg):
    """The regression itself: sessions() must not be permanently empty."""
    store = Store(cfg / "cfg" / "sessions.db")
    _session(cfg, store, "s-1", "first")
    _session(cfg, store, "s-2", "second")

    got = SlashCompleter(cfg).sessions()
    assert set(got) == {"s-1", "s-2"}, got


def test_sessions_from_another_directory_are_excluded(cfg):
    store = Store(cfg / "cfg" / "sessions.db")
    _session(cfg, store, "mine", "here")
    _session(cfg / "elsewhere", store, "theirs", "other")

    got = SlashCompleter(cfg).sessions()
    assert "mine" in got
    assert "theirs" not in got, got


def test_completer_offers_session_ids_after_resume(cfg):
    store = Store(cfg / "cfg" / "sessions.db")
    _session(cfg, store, "20260101-000000-000", "a session")

    text = "/resume "
    got = [c.text for c in SlashCompleter(cfg).get_completions(Document(text, len(text)), None)]
    assert "20260101-000000-000" in got, got


def test_completer_survives_a_missing_db(cfg):
    """No db yet must not raise -- the prompt still has to work."""
    assert SlashCompleter(cfg).sessions() == {}


def test_session_meta_shows_the_title(cfg):
    store = Store(cfg / "cfg" / "sessions.db")
    _session(cfg, store, "s-1", "named session")

    text = "/resume "
    got = {
        c.text: c.display_meta for c in SlashCompleter(cfg).get_completions(Document(text, len(text)), None)
    }
    meta = got["s-1"]
    meta = meta if isinstance(meta, str) else str(meta)
    assert "sessions" in meta or "named session" in meta, meta
