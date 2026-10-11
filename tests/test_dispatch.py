"""Slash-command dispatch must never leak a command into the model."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from minicode.usage import Usage


class FakeSession:
    def __init__(self, root, session_id, store, *, approve, sandbox="", sandbox_setup=""):
        self.root = root
        self.id = session_id
        self.plan_mode = False
        self.model_spec = None
        self.mcp_tools = []
        self.mcp_dead = []
        self.store = store
        self.usage = Usage()
        self.max_model_calls = None
        self.sandbox = sandbox
        self.redact_pii = False
        self.rubric = None
        self.last_prompt = None
        self.agent = types.SimpleNamespace(
            get_state=lambda cfg: types.SimpleNamespace(values={}),
            update_state=lambda cfg, s: None,
        )

    def config(self):
        return {"configurable": {"thread_id": self.id}}

    def rebuild(self):
        pass

    def try_rebuild(self):
        return True

    def close(self):
        pass


class FakeStore:
    path = Path("unused")

    def create(self, *a, **k):
        pass

    def touch(self, *a, **k):
        pass

    def latest(self, cwd):
        return None

    def list(self, cwd, limit=20):
        return []

    def rename(self, *a, **k):
        return True

    def export(self, *a, **k):
        return None


def _run_repl(inputs, monkeypatch, tmp_path):
    """Run the real run() loop with scripted input; record every model payload."""
    import minicode.cli as C

    sent: list[str] = []

    def fake_drive(agent, config, payload, usage=None):
        if isinstance(payload, dict) and "messages" in payload:
            sent.append(payload["messages"][0]["content"])

    replies = iter(inputs)

    def fake_prompt(root):
        try:
            return next(replies)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(C, "Session", FakeSession)
    monkeypatch.setattr(C, "Store", lambda *a, **k: FakeStore())
    monkeypatch.setattr(C, "_load_mcp", lambda: ([], []))
    monkeypatch.setattr(C, "_read_prompt", fake_prompt)
    monkeypatch.setattr(C, "_drive", fake_drive)
    monkeypatch.setattr(C, "_header", lambda s: None)
    monkeypatch.setattr(C, "new_session_id", lambda: "s1")

    C.run(None, cwd=tmp_path, approve=False, resume=False)
    return sent


@pytest.fixture
def no_config(tmp_path, monkeypatch):
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    return tmp_path


def test_slash_command_never_reaches_the_model(no_config, monkeypatch, tmp_path):
    """/mcp, /model, /todos and friends are CLI-side only."""
    import minicode.cli as C

    monkeypatch.setattr(C, "_cmd_mcp", lambda *a: None, raising=False)
    monkeypatch.setattr(C, "_cmd_model", lambda *a: None, raising=False)
    monkeypatch.setattr(C, "_cmd_skills", lambda *a: None, raising=False)

    for cmd in ("/mcp", "/model", "/skills list", "/todos", "/memory", "/sessions", "/help"):
        sent = _run_repl([cmd, "/exit"], monkeypatch, tmp_path)
        assert sent == [], f"{cmd} leaked to the model: {sent}"


def test_unknown_command_does_not_reach_the_model(no_config, monkeypatch, tmp_path):
    sent = _run_repl(["/nope", "/exit"], monkeypatch, tmp_path)
    assert sent == [], sent


def test_real_prompt_after_a_command_still_runs(no_config, monkeypatch, tmp_path):
    """A command, then a real question: only the question reaches the model."""
    sent = _run_repl(["/help", "fix the bug", "/exit"], monkeypatch, tmp_path)
    assert sent == ["fix the bug"], sent


def test_skill_pick_becomes_a_prompt(no_config, monkeypatch, tmp_path):
    """/skills -> picker -> the choice turns into the turn prompt."""
    import minicode.cli as C

    monkeypatch.setattr(C, "_pick_skill", lambda root: "release-check")
    monkeypatch.setattr(C, "_cmd_skills", lambda *a: None)
    monkeypatch.setattr(type(C.console), "is_terminal", property(lambda self: True))

    sent = _run_repl(["/skills", "/exit"], monkeypatch, tmp_path)
    assert sent == ["Use the release-check skill for this task."], sent


def test_cancelled_picker_sends_nothing(no_config, monkeypatch, tmp_path):
    import minicode.cli as C

    monkeypatch.setattr(C, "_pick_skill", lambda root: None)
    monkeypatch.setattr(type(C.console), "is_terminal", property(lambda self: True))

    sent = _run_repl(["/skills", "/exit"], monkeypatch, tmp_path)
    assert sent == [], sent


def test_skills_list_never_opens_the_picker(no_config, monkeypatch, tmp_path):
    """/skills list must stay non-interactive, so pipes still work."""
    import minicode.cli as C

    monkeypatch.setattr(C, "_pick_skill", lambda root: "boom")
    monkeypatch.setattr(C, "_cmd_skills", lambda *a: None)
    monkeypatch.setattr(type(C.console), "is_terminal", property(lambda self: True))

    sent = _run_repl(["/skills list", "/exit"], monkeypatch, tmp_path)
    assert sent == [], sent


def test_bad_model_does_not_kill_the_session(tmp_path):
    """A rebuild failure must be reported, not crash the whole TUI."""
    from minicode.cli import Session

    s = Session.__new__(Session)
    s.root = tmp_path
    s.id = "s1"
    s.approve = False
    s.plan_mode = False
    s.model = object()
    s.model_alias = None
    s.mcp_tools = []
    s.max_model_calls = None
    s.redact_pii = False
    s.rubric = None
    s.agent = object()

    def boom():
        raise ValueError("model 'nope' is not configured")

    s.rebuild = boom
    assert s.try_rebuild() is False, "should report failure"
    assert isinstance(s.agent, object), "previous agent must survive"


def test_try_rebuild_succeeds_when_ok(tmp_path):
    from minicode.cli import Session

    s = Session.__new__(Session)
    s.rebuild = lambda: None
    assert s.try_rebuild() is True


def test_spec_nit_warnings_are_dropped():
    """The cosmetic skill-spec warnings must not repeat every turn."""
    import logging

    from minicode.cli import _DropSpecNits

    f = _DropSpecNits()

    def rec(msg):
        return logging.LogRecord("deepagents.middleware.skills", logging.WARNING, "", 0, msg, (), None)

    assert not f.filter(rec("Skill 'x' does not follow Agent Skills specification: blah"))
    assert f.filter(rec("Skipping foo: no valid YAML frontmatter found")), "real warnings must survive"


def test_spec_filter_is_attached_to_the_emitting_logger(caplog):
    """A filter on a parent logger is not applied to child records.

    This is the bug: the filter existed and was unit-tested, but never actually
    filtered anything, because it hung off `deepagents` instead of the logger
    that emits.
    """
    import logging

    import minicode.cli

    logger = logging.getLogger("deepagents.middleware.skills")
    assert any(isinstance(f, type(minicode.cli._DropSpecNits())) for f in logger.filters), (
        f"filter not attached to {logger.name}: {logger.filters}"
    )

    with caplog.at_level(logging.WARNING, logger="deepagents.middleware.skills"):
        logger.warning("Skill 'x' does not follow Agent Skills specification: blah")
        logger.warning("Skipping foo: no valid YAML frontmatter found")

    texts = [r.getMessage() for r in caplog.records]
    assert not any("does not follow Agent Skills" in t for t in texts), texts
    assert any("no valid YAML" in t for t in texts), texts
