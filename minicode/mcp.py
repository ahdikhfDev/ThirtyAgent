from __future__ import annotations

from typing import Any


def connections() -> dict[str, dict]:
    from .config import load

    return dict(load().get("mcp") or {})


def client(overrides: dict | None = None):
    """MultiServerMCPClient over the configured servers, or None if there are none."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    servers = {**connections(), **(overrides or {})}
    if not servers:
        return None

    specs: dict[str, dict] = {}
    for name, spec in servers.items():
        if "url" in spec:
            specs[name] = {
                "url": spec["url"],
                "transport": spec.get("transport", "streamable_http"),
                **({"headers": spec["headers"]} if spec.get("headers") else {}),
            }
        else:
            specs[name] = {
                "transport": "stdio",
                "command": spec["command"],
                "args": spec.get("args", []),
                **({"env": spec["env"]} if spec.get("env") else {}),
            }
    return MultiServerMCPClient(specs)


def load_tools(servers: dict | None = None) -> tuple[list[Any], list[tuple[str, str]]]:
    """Connect and pull every tool. A dead server must not kill the agent.

    Returns (tools, [(server_name, error)]). Never raises.
    """
    all_servers = {**connections(), **(servers or {})}
    if not all_servers:
        return [], []

    tools: list[Any] = []
    dead: list[tuple[str, str]] = []
    try:
        c = client(all_servers)
        for name in all_servers:
            try:
                got = c.get_tools(server_name=name)
                if hasattr(got, "__await__"):
                    got = _run(got)
                tools.extend(got)
            except Exception as e:  # noqa: BLE001 - one bad server must not kill the session
                dead.append((name, str(e).splitlines()[0][:70]))
    except Exception as e:  # noqa: BLE001
        return [], [("all", str(e).splitlines()[0][:70])]
    return tools, dead


def _run(coro):
    """get_tools is async in some builds and sync in others; handle both."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # already inside a loop -- the CLI is sync, so this should not happen
    raise RuntimeError("MCP get_tools called from a running event loop")


def probe(name: str, spec: dict) -> tuple[bool, str]:
    """Try one server and report what happened, for `/mcp test`."""
    tools, dead = load_tools({name: spec})
    for bad, err in dead:
        if bad == name:
            return False, err
    return True, f"{len(tools)} tool(s)"


def add(name: str, spec: dict) -> None:
    from .config import load, save

    cfg = load()
    cfg.setdefault("mcp", {})[name] = spec
    save(cfg)


def remove(name: str) -> bool:
    from .config import load, save

    cfg = load()
    if name not in (cfg.get("mcp") or {}):
        return False
    del cfg["mcp"][name]
    save(cfg)
    return True
