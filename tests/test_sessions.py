"""Session persistence, plan mode, and slash commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from minicode.sessions import Store, new_session_id

CFG = {"configurable": {"thread_id": "t"}}


class _Fake(GenericFakeChatModel):
    """Never touches the network -- the repo has a real .env, so build_agent
    would otherwise dial the user's live endpoint during tests."""

    script: list[AIMessage] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _next(self) -> AIMessage:
        return self.script.pop(0) if self.script else AIMessage(content="done")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        m = self._next()
        yield ChatGenerationChunk(message=AIMessageChunk(content=m.content, tool_calls=m.tool_calls, id=m.id))


def echo_model(*replies: str) -> _Fake:
    return _Fake(messages=iter(()), script=[AIMessage(content=r) for r in replies])


def tool_model(name: str, args: dict) -> _Fake:
    return _Fake(
        messages=iter(()),
        script=[
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "1"}]),
            AIMessage(content="done"),
        ],
    )


def _mem():
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


def test_derive_title_is_concrete_not_untitled():
    """Regression: typing at the prompt left the title empty forever, so
    /sessions showed (untitled) for every interactive session."""
    from minicode.sessions import derive_title

    cases = {
        "check C:\\Users\\me\\Desktop\\thirtyapi": "check thirtyapi",
        "read C:\\Users\\me\\AppData\\Local\\Temp\\opencode\\minicode": "read minicode",
        "  refactor   the\n  session title  code ": "refactor the session title code",
        "fix the failing test": "fix the failing test",
    }
    for raw, want in cases.items():
        got = derive_title(raw)
        assert got and got != "(untitled)", raw
        assert "\n" not in got, raw
        assert got == want, f"{raw!r} -> {got!r}"


def test_derive_title_truncates_without_crashing():
    """rsplit(' ') found no space when the title is one long unbroken word."""
    from minicode.sessions import derive_title

    long = "a" * 200
    assert derive_title(long).endswith("…")
    # one huge token, no spaces at all anywhere in the cut window
    assert derive_title("x" * 90).endswith("…")
    assert derive_title("")


def test_derive_title_handles_empty_and_quotes():
    from minicode.sessions import derive_title

    assert derive_title("") == "(untitled)"
    assert derive_title("   ") == "(untitled)"
    assert derive_title('"quoted ask"') == "quoted ask"


def test_derive_title_truncates_long_prompts():
    from minicode.sessions import derive_title

    got = derive_title("word " * 60, limit=60)
    assert len(got) <= 60, len(got)
    assert got.endswith("…")
    assert got != "(untitled)"


def test_first_prompt_names_the_session(tmp_path: Path):
    from minicode.sessions import derive_title

    store = Store(tmp_path / "s.db")
    store.create("abc-123", tmp_path)
    store.touch("abc-123", tmp_path, title=derive_title("check the thirtyapi project"))
    assert store.list(tmp_path)[0]["title"] == "check the thirtyapi project"


def test_rename_survives_later_touches(tmp_path: Path):
    """COALESCE means a None title must not wipe an explicit /rename."""
    store = Store(tmp_path / "s.db")
    store.create("abc-123", tmp_path, title="original title")
    store.rename("abc-123", "my chosen name")
    store.touch("abc-123", tmp_path, title=None)
    assert store.list(tmp_path)[0]["title"] == "my chosen name"


class _FakeSession:
    def __init__(self, store, root, sid):
        self.store, self.root, self.id = store, root, sid


def test_title_for_returns_none_once_titled(tmp_path: Path):
    """A resumed session must keep its name, not take the latest stray message."""
    from minicode.cli import _title_for

    store = Store(tmp_path / "s.db")
    store.create("s1", tmp_path, title="first real ask")
    assert _title_for(_FakeSession(store, tmp_path, "s1"), "some later message") is None


def test_title_for_fills_untitled_session(tmp_path: Path):
    from minicode.cli import _title_for

    store = Store(tmp_path / "s.db")
    store.create("s1", tmp_path)
    got = _title_for(_FakeSession(store, tmp_path, "s1"), "check this project")
    assert got == "check this project"


def test_new_session_command_leaves_title_open_for_first_prompt(tmp_path: Path):
    """Regression: /new used to create with an empty title and nothing ever
    filled it in, so every /new session stayed (untitled)."""
    store = Store(tmp_path / "s.db")
    store.create("new-1", tmp_path)
    store.touch("new-1", tmp_path, title="first prompt after /new")
    assert store.list(tmp_path)[0]["title"] == "first prompt after /new"


def test_store_create_and_latest(tmp_path: Path):
    store = Store(tmp_path / "s.db")
    a, b = new_session_id(), new_session_id()
    store.create(a, tmp_path, "first task")
    store.create(b, tmp_path, "second task")

    assert {r["id"] for r in store.list(tmp_path)} == {a, b}
    assert store.latest(tmp_path) == b, "latest must be the newest"


def test_store_scopes_by_directory(tmp_path: Path):
    store = Store(tmp_path / "s.db")
    here, there = tmp_path / "p1", tmp_path / "p2"
    store.create("a", here)
    store.create("b", there)

    assert [r["id"] for r in store.list(here)] == ["a"], "cwd filter leaked"
    assert store.latest(here) == "a"
    assert store.latest(there) == "b"


def test_store_rename_touch_export_delete(tmp_path: Path):
    store = Store(tmp_path / "s.db")
    store.create("s1", tmp_path)

    assert store.rename("s1", "checkout timeout") is True
    assert store.rename("nope", "x") is False

    store.touch("s1", tmp_path)
    assert store.list(tmp_path)[0]["title"] == "checkout timeout", "touch clobbered the title"

    out = store.export("s1", tmp_path / "out.json")
    assert out.exists() and "checkout timeout" in out.read_text()

    assert store.delete("s1") is True
    assert store.list(tmp_path) == []


def test_session_roundtrips_through_sqlite(tmp_path: Path):
    """A rebuilt agent must see the conversation a previous one left behind."""
    from minicode.agent import build_agent
    from minicode.cli import Session

    store = Store(tmp_path / "s.db")
    root = tmp_path / "proj"
    root.mkdir()
    store.create("sess1", root)

    s = Session(root, "sess1", store, approve=False, model=echo_model("hi"))
    try:
        s.agent = build_agent(root, checkpointer=s.checkpointer, model=echo_model("hi"))
        s.agent.invoke({"messages": [HumanMessage("remember: the port is 8080")]}, s.config())

        # brand new Session object, same thread -> history must survive
        s2 = Session(root, "sess1", store, approve=False, model=echo_model("ok"))
        try:
            s2.rebuild()
            state = s2.agent.get_state(s2.config()).values
            texts = [m.content for m in state["messages"]]
            assert any("8080" in str(t) for t in texts), f"history lost: {texts}"
        finally:
            s2.close()
    finally:
        s.close()


def test_plan_mode_denies_writes(tmp_path: Path):
    """Plan mode must refuse edits at the permission layer, not just by prompt."""
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from minicode.agent import build_agent

    root = tmp_path / "proj"
    root.mkdir()

    agent = build_agent(
        root,
        checkpointer=InMemorySaver(),
        plan_mode=True,
        model=tool_model("write_file", {"file_path": "/plan.txt", "content": "nope"}),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "write it"}]}, CFG)

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs, "tool never ran"
    assert tool_msgs[0].status == "error", f"plan mode allowed the write: {tool_msgs[0].content}"
    assert not (root / "plan.txt").exists(), "file was written in plan mode"


def test_plan_mode_hides_the_shell(tmp_path: Path):
    """Plan mode must not even expose `execute`."""
    from minicode.agent import build_agent

    root = tmp_path / "proj"
    root.mkdir()

    plan = build_agent(root, checkpointer=_mem(), plan_mode=True, model=echo_model())
    live = build_agent(root, checkpointer=_mem(), plan_mode=False, model=echo_model())

    def names(a):
        node = a.nodes["tools"]
        return set(node.bound.tools_by_name) if hasattr(node.bound, "tools_by_name") else set()

    assert {"read_file", "ls", "glob", "grep"} <= names(plan), f"plan mode lost read tools: {names(plan)}"
    assert not ({"write_file", "edit_file", "delete", "execute"} & names(plan)), (
        f"plan mode still exposes mutating tools: {names(plan)}"
    )
    assert "execute" in names(live), "normal mode lost the shell"
    assert "write_file" in names(live), "normal mode lost write_file"


def test_normal_mode_allows_writes(tmp_path: Path):
    """Guard against the permission rule leaking into the default path."""
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.memory import InMemorySaver

    from minicode.agent import build_agent

    root = tmp_path / "proj"
    root.mkdir()

    agent = build_agent(
        root,
        checkpointer=InMemorySaver(),
        plan_mode=False,
        model=tool_model("write_file", {"file_path": "/ok.txt", "content": "yes"}),
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "write it"}]}, CFG)

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].status == "success", tool_msgs[0].content if tool_msgs else "none"
    assert (root / "ok.txt").read_text() == "yes"


def test_find_memory_picks_up_context_files(tmp_path: Path):
    from minicode.agent import find_memory

    assert find_memory(tmp_path) == []
    (tmp_path / "AGENTS.md").write_text("rules", encoding="utf-8")
    assert len(find_memory(tmp_path)) == 1
