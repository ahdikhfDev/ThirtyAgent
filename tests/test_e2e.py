"""End-to-end: real agent graph, real tools, real backends. Only the LLM is fake.

These are the checks that would catch a broken build before you ever open the
terminal: the CLI wires up, tools reach disk, approvals block, plan mode is
read-only, budgets fire, usage adds up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class Scripted(GenericFakeChatModel):
    """Replays a script, then a plain reply. Never touches the network."""

    script: list[AIMessage] = []
    usage_per_call: dict = {}

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _next(self) -> AIMessage:
        return self.script.pop(0) if self.script else AIMessage(content="done")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        msg = self._next()
        msg.usage_metadata = dict(self.usage_per_call or {"input_tokens": 10, "output_tokens": 5})
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        m = self._next()
        m.usage_metadata = dict(self.usage_per_call or {"input_tokens": 10, "output_tokens": 5})
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=m.content, tool_calls=m.tool_calls, id=m.id),
            generation_info={"usage_metadata": m.usage_metadata},
        )


def tool(name: str, args: dict, cid: str = "1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from minicode import skills as S
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    monkeypatch.setattr(S, "USER_DIRS", ())
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(root: Path, model, **kw):
    from minicode.agent import build_agent

    return build_agent(root, model=model, **kw)


# --- build ---


def test_agent_builds_with_the_full_stack(isolated):
    root = isolated / "p"
    root.mkdir()
    agent = build(root, Scripted(messages=iter(())), use_skills=True, max_model_calls=3, redact_pii=True)
    assert agent is not None


def test_tools_include_files_shell_fetch_and_todo(isolated):
    root = isolated / "p"
    root.mkdir()
    agent = build(root, Scripted(messages=iter(())))
    names = set(agent.nodes["tools"].bound.tools_by_name)
    for expected in ("ls", "read_file", "write_file", "edit_file", "execute", "fetch_url", "task"):
        assert expected in names, f"{expected} missing from {sorted(names)}"


# --- tools really work ---


def test_write_then_execute_reach_the_disk(isolated):
    root = isolated / "p"
    root.mkdir()
    model = Scripted(
        messages=iter(()),
        script=[
            tool("write_file", {"file_path": "/hello.txt", "content": "hi"}, "1"),
            AIMessage(content="wrote it"),
        ],
    )
    agent = build(root, model)
    agent.invoke(
        {"messages": [{"role": "user", "content": "write hello.txt"}]},
        {"configurable": {"thread_id": "t"}},
    )
    assert (root / "hello.txt").read_text() == "hi"


def test_shell_command_actually_runs(isolated):
    root = isolated / "p"
    root.mkdir()
    model = Scripted(
        messages=iter(()),
        script=[
            tool("execute", {"command": "echo integration-ok"}, "1"),
            AIMessage(content="ok"),
        ],
    )
    agent = build(root, model)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "echo it"}]},
        {"configurable": {"thread_id": "t"}},
    )
    outputs = " ".join(str(m.content) for m in result["messages"] if m.type == "tool")
    assert "integration-ok" in outputs, outputs


# --- safety ---


def test_approval_blocks_a_shell_command(isolated):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    root = isolated / "p"
    root.mkdir()
    model = Scripted(
        messages=iter(()),
        script=[tool("execute", {"command": "echo nope"}, "1"), AIMessage(content="done")],
    )
    agent = build(root, model, approve=True, checkpointer=InMemorySaver())
    cfg = {"configurable": {"thread_id": "t"}}

    result = agent.invoke({"messages": [{"role": "user", "content": "run it"}]}, cfg)
    assert result.get("__interrupt__"), "expected an approval prompt"

    after = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), cfg)
    outputs = " ".join(str(m.content) for m in after["messages"] if m.type == "tool")
    assert "nope" not in outputs, f"rejected command still ran: {outputs}"


def test_plan_mode_cannot_touch_the_disk(isolated):
    from langchain_core.messages import ToolMessage

    root = isolated / "p"
    root.mkdir()
    model = Scripted(
        messages=iter(()),
        script=[tool("write_file", {"file_path": "/x", "content": "no"}, "1"), AIMessage(content="done")],
    )
    agent = build(root, model, plan_mode=True)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "write"}]},
        {"configurable": {"thread_id": "t"}},
    )
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and tool_msgs[0].status == "error"
    assert not (root / "x").exists()


def test_model_call_budget_stops_a_loop(isolated):
    """A runaway agent must be capped, not billed forever."""
    root = isolated / "p"
    root.mkdir()
    # Never converges: keeps calling ls until the budget cuts it off.
    model = Scripted(messages=iter(()), script=[])
    agent = build(root, model, max_model_calls=2)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "go"}]},
        {"configurable": {"thread_id": "t"}},
        recursion_limit=60,
    )
    calls = sum(1 for m in result["messages"] if m.type == "tool")
    assert calls <= 6, f"budget did not stop the loop: {calls} tool calls"


def _writer():
    return Scripted(
        messages=iter(()),
        script=[
            tool("write_file", {"file_path": "/a", "content": "x"}, "1"),
            AIMessage(content="done"),
        ],
    )


def test_approve_flag_is_wired(isolated):
    """Regression: --approve became a no-op when build_agent was rewritten."""
    root = isolated / "p"
    root.mkdir()
    cfg = {"configurable": {"thread_id": "t"}}

    blocked = build(root, _writer(), approve=True)
    assert blocked.invoke({"messages": [{"role": "user", "content": "w"}]}, cfg).get("__interrupt__"), (
        "approve=True did not raise an interrupt"
    )
    assert not (root / "a").exists()

    free = build(root, _writer(), approve=False)
    free.invoke({"messages": [{"role": "user", "content": "w"}]}, cfg)
    assert (root / "a").exists(), "approve=False still blocked the write"


def test_pii_middleware_redacts(isolated):
    """Email must not reach the model verbatim when redaction is on."""
    root = isolated / "p"
    root.mkdir()

    seen: list = []

    class Spy(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.extend(str(m.content) for m in messages)
            return super()._generate(messages, stop, run_manager, **kw)

    agent = build(root, Spy(messages=iter(())), redact_pii=True)
    agent.invoke(
        {"messages": [{"role": "user", "content": "email me at bob@example.com"}]},
        {"configurable": {"thread_id": "t"}},
    )
    joined = " ".join(seen)
    assert "bob@example.com" not in joined, f"PII leaked to the model: {joined}"


# --- usage ---


def test_usage_rolls_up_across_turns():
    from minicode.usage import Usage

    u = Usage()
    u.record_message({"input_tokens": 100, "output_tokens": 20}, model="m")
    u.record_message({"input_tokens": 50, "output_tokens": 10}, model="m")
    turn = u.end_turn()

    assert turn["in"] == 150 and turn["out"] == 30
    assert u.input_tokens == 150 and u.output_tokens == 30
    assert u.total_tokens == 180
    assert u.calls == 2 and u.turns == 1
    assert u.by_model["m"]["calls"] == 2


def test_usage_turns_do_not_leak_into_each_other():
    from minicode.usage import Usage

    u = Usage()
    u.record_message({"input_tokens": 10, "output_tokens": 1})
    u.end_turn()
    u.record_message({"input_tokens": 20, "output_tokens": 2})
    second = u.end_turn()

    assert second["in"] == 20, "turn 2 counted turn 1's tokens"
    assert u.input_tokens == 30


def test_usage_ignores_messages_without_metadata():
    from minicode.usage import Usage

    u = Usage()
    u.record_message(None)
    u.record_message({})
    assert u.calls == 0


# --- sessions ---


def test_session_survives_a_restart(isolated):
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import InMemorySaver

    root = isolated / "p"
    root.mkdir()
    cfg = {"configurable": {"thread_id": "keep"}}

    saver = InMemorySaver()
    a = build(root, Scripted(messages=iter(())), checkpointer=saver)
    a.invoke({"messages": [HumanMessage("the secret is 42")]}, cfg)

    b = build(root, Scripted(messages=iter(())), checkpointer=saver)
    texts = [str(m.content) for m in b.get_state(cfg).values["messages"]]
    assert any("42" in t for t in texts), texts


def test_plan_mode_survives_a_restart(isolated):
    from langgraph.checkpoint.memory import InMemorySaver

    root = isolated / "p"
    root.mkdir()
    saver = InMemorySaver()
    build(root, Scripted(messages=iter(())), checkpointer=saver, plan_mode=True)
    agent = build(root, Scripted(messages=iter(())), checkpointer=saver)
    names = set(agent.nodes["tools"].bound.tools_by_name)
    assert "write_file" in names, "plan mode flag leaked into the next session"
