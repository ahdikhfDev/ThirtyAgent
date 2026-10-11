"""/rubric wired end to end.

The bug this covers: Session.rubric was initialised to None, threaded through
rebuild() into build_agent(), and reset to None by /trace -- but nothing ever
set it to a value. agent.py's `if rubric:` branch was unreachable from the CLI.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import minicode.cli as C


class FakeSession:
    """Just enough session for the command handlers and dispatch."""

    def __init__(self):
        self.root = Path.cwd()
        self.id = "s1"
        self.approve = False
        self.plan_mode = False
        self.sandbox = ""
        self.sandbox_setup = ""
        self.model_spec = {"model": "m", "alias": "a", "base_url": "u"}
        self.mcp_tools = []
        self.mcp_dead = []
        self.usage = types.SimpleNamespace()
        self.max_model_calls = None
        self.redact_pii = False
        self.rubric = None
        self.last_prompt = None
        self.rebuilds = 0

    def config(self):
        return {"configurable": {"thread_id": self.id}}

    def try_rebuild(self):
        self.rebuilds += 1
        return True


@pytest.fixture
def no_config(tmp_path, monkeypatch):
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    return tmp_path


def test_rubric_is_registered_everywhere():
    """Completer, catalog and /help must agree, or the command is invisible."""
    from minicode.catalog import COMMANDS
    from minicode.complete import COMMANDS as COMPLETER

    assert "/rubric" in COMPLETER
    assert any(t == "/rubric" for t, *_ in COMMANDS)
    assert "/rubric" in C.HELP


def test_setting_a_rubric_stores_and_rebuilds(no_config):
    s = FakeSession()
    C._cmd_rubric("no bare prints, keep diffs minimal", s)

    assert s.rubric == "no bare prints, keep diffs minimal"
    assert s.rebuilds == 1, "the agent must be rebuilt for the rubric to apply"


def test_rubric_off_clears_it(no_config):
    s = FakeSession()
    C._cmd_rubric("be terse", s)
    C._cmd_rubric("off", s)

    assert s.rubric is None
    assert s.rebuilds == 2


def test_bare_rubric_reports_status_without_rebuilding(no_config):
    s = FakeSession()
    C._cmd_rubric("", s)
    assert s.rubric is None
    assert s.rebuilds == 0


def test_failed_rebuild_leaves_rubric_unset(no_config):
    """A rubric the agent cannot build must not linger as half-applied state."""
    s = FakeSession()

    def boom():
        return False

    s.try_rebuild = boom
    C._cmd_rubric("some criteria", s)
    assert s.rubric is None


def test_dispatch_reaches_the_rubric_handler(no_config, monkeypatch):
    s = FakeSession()
    seen = {}
    monkeypatch.setattr(C, "_cmd_rubric", lambda rest, session: seen.update(rest=rest, s=session))

    C._run_command("/rubric keep it under 10 lines", s)
    assert seen["rest"] == "keep it under 10 lines"
    assert seen["s"] is s


def test_trace_no_longer_clears_the_rubric(no_config, monkeypatch):
    """Regression: /trace used to reset session.rubric to None."""
    import inspect

    src = inspect.getsource(C._cmd_trace)
    assert "rubric" not in src, "tracing must not touch the rubric"


def test_rubric_reaches_build_agent(no_config, monkeypatch):
    """The whole point: the value flows through rebuild() into build_agent()."""
    captured = {}
    # cli.py imports build_agent into its own namespace, so patch it there.
    monkeypatch.setattr(C, "build_agent", lambda root, **kw: captured.update(kw) or "AGENT")

    s = FakeSession()
    s.checkpointer = object()
    s.model = object()
    s.model_alias = None
    s.rubric = "keep diffs minimal"

    C.Session.rebuild(s)
    assert captured.get("rubric") == "keep diffs minimal"
