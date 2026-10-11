"""Model config, skill discovery, MCP wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import minicode.config as cfgmod
from minicode.config import load, save


def test_discover_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICODE_BASE_URL", "https://api.example/v1")
    monkeypatch.setenv("MINICODE_API_KEY", "sk-test")
    monkeypatch.setenv("MINICODE_MODEL", "big-model")

    models = cfgmod.discover_models()
    assert "default" in models, models
    assert models["default"]["model"] == "big-model"
    assert cfgmod.resolve()["alias"] == "default"


def test_explicit_config_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MINICODE_BASE_URL", "https://env.example/v1")
    monkeypatch.setenv("MINICODE_MODEL", "env-model")
    cfg = load()
    cfg["models"]["fast"] = {
        "base_url": "https://fast.example/v1",
        "api_key": "k",
        "model": "small",
    }
    save(cfg)

    models = cfgmod.discover_models()
    assert models["fast"]["model"] == "small"
    assert cfgmod.resolve("fast")["base_url"] == "https://fast.example/v1"


def test_active_alias_is_persisted(tmp_path):
    cfg = load()
    cfg["models"] = {
        "a": {"base_url": "u", "api_key": "k", "model": "m1"},
        "b": {"base_url": "u", "api_key": "k", "model": "m2"},
    }
    cfg["active"] = "b"
    save(cfg)
    assert cfgmod.active_alias() == "b"
    assert cfgmod.resolve()["model"] == "m2"


def test_resolve_returns_none_for_unknown_alias():
    assert cfgmod.resolve("nope") is None


def test_config_roundtrips(tmp_path):
    cfg = load()
    cfg["mcp"]["github"] = {"command": "npx", "args": ["-y", "mcp"]}
    save(cfg)
    assert json.loads((tmp_path / "cfg" / "config.json").read_text())["mcp"]["github"]["command"] == "npx"


# --- skills ---


def _skill(root: Path, base: Path, name: str, desc: str, front: bool = True) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    fm = f"---\nname: {name}\ndescription: {desc}\n---\n" if front else ""
    (d / "SKILL.md").write_text(f"{fm}\n# {name}\n", encoding="utf-8")
    return d


@pytest.fixture
def no_user_skills(monkeypatch):
    """The developer really has ~/.agents/skills installed; don't assert against those."""
    from minicode import skills as S

    monkeypatch.setattr(S, "USER_DIRS", ())
    return S


def test_discover_reads_frontmatter(tmp_path, no_user_skills):
    root = tmp_path / "proj"
    root.mkdir()
    _skill(root, root / ".thirtyagent" / "skills", "release-check", "Use when releasing.")

    found = no_user_skills.discover(root)
    assert [s.name for s in found] == ["release-check"]
    assert found[0].description == "Use when releasing."
    assert found[0].scope == "project"


def test_discover_without_frontmatter_uses_dir_name(tmp_path, no_user_skills):
    root = tmp_path / "proj"
    root.mkdir()
    _skill(root, root / ".thirtyagent" / "skills", "bare-skill", "", front=False)
    assert [s.name for s in no_user_skills.discover(root)] == ["bare-skill"]


def test_project_scope_wins_over_user(tmp_path, monkeypatch):
    """omp precedence: a project skill shadows a user skill of the same name."""
    root = tmp_path / "proj"
    root.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    _skill(root, user, "dup", "user copy")
    _skill(root, root / ".thirtyagent" / "skills", "dup", "project copy")

    from minicode import skills as S

    monkeypatch.setattr(S, "USER_DIRS", (user,))
    found = S.discover(root)
    assert len(found) == 1, "duplicate names should collapse"
    assert found[0].description == "project copy", "project scope did not win"


def test_include_and_ignore_filters(tmp_path, monkeypatch):
    root = tmp_path / "proj"
    root.mkdir()
    for n in ("git-flow", "postgres", "pg-experimental"):
        _skill(root, root / ".thirtyagent" / "skills", n, "x")

    from minicode import skills as S

    monkeypatch.setattr(S, "USER_DIRS", ())
    cfg = load()
    cfg["skills"] = {"enabled": True, "include": ["git-*", "postgres"], "ignore": ["*-experimental"]}
    save(cfg)

    assert [s.name for s in S.discover(root)] == ["git-flow", "postgres"]


# --- mcp ---


def test_mcp_connection_shapes(tmp_path):
    import minicode.mcp as M

    M.add("local", {"command": "npx", "args": ["-y", "server"]})
    M.add("remote", {"url": "https://x/mcp", "transport": "streamable_http"})
    assert M.connections()["local"]["command"] == "npx"
    assert M.remove("local") is True
    assert M.remove("local") is False
    assert "remote" in M.connections()


def test_mcp_dead_server_does_not_raise(tmp_path):
    """One broken server must not take down the whole agent build."""
    import minicode.mcp as M

    M.add("broken", {"command": "definitely-not-a-real-binary-xyz", "args": []})
    tools, dead = M.load_tools()
    assert isinstance(tools, list)
    assert any(n == "broken" for n, _ in dead), dead


def test_mcp_no_servers_returns_nothing(tmp_path):
    import minicode.mcp as M

    assert M.client() is None
    assert M.load_tools() == ([], [])


def test_stdio_spec_includes_transport(tmp_path):
    """langchain-mcp-adapters rejects a stdio spec with no transport key."""
    import minicode.mcp as M

    seen = {}
    monkey = pytest.MonkeyPatch()
    try:
        import langchain_mcp_adapters.client as C

        monkey.setattr(C, "MultiServerMCPClient", lambda specs: seen.setdefault("specs", specs))
        M.client({"s": {"command": "npx", "args": ["-y", "srv"]}})
        assert seen["specs"]["s"]["transport"] == "stdio"
        assert seen["specs"]["s"]["command"] == "npx"
    finally:
        monkey.undo()


def test_http_spec_carries_transport(tmp_path):
    import minicode.mcp as M

    seen = {}
    monkey = pytest.MonkeyPatch()
    try:
        import langchain_mcp_adapters.client as C

        monkey.setattr(C, "MultiServerMCPClient", lambda specs: seen.setdefault("specs", specs))
        M.client({"s": {"url": "https://x/mcp"}})
        assert seen["specs"]["s"]["transport"] == "streamable_http"
        assert seen["specs"]["s"]["url"] == "https://x/mcp"
    finally:
        monkey.undo()


def test_config_load_does_not_share_default_dict():
    """Shallow-copying DEFAULT made add() mutate the module default process-wide."""
    import minicode.config as C

    first = C.load()
    first["mcp"]["leak"] = {"command": "x"}
    assert "leak" not in C.load()["mcp"], "load() leaked state into DEFAULT"
    assert "leak" not in C.DEFAULT["mcp"]
