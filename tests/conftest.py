"""Stop the repo's real .env from leaking into tests.

`minicode.agent` calls load_dotenv() at import, and build_agent() falls back to a
live ChatOpenAI when no model is injected. Tests must never reach the user's
paid endpoint.
"""

import os
from pathlib import Path

import pytest

_MINICODE_VARS = (
    "MINICODE_BASE_URL",
    "MINICODE_API_KEY",
    "MINICODE_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Never read the developer's real config, history, or the repo .env."""
    from minicode import theme

    monkeypatch.setattr(theme, "config_dir", lambda: tmp_path / "cfg")
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _no_live_endpoint(monkeypatch):
    """Fail loudly if a test tries to build a model that could hit the network."""
    for var in _MINICODE_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Socket guard. The fetch tool's own tests use 127.0.0.1, which stays allowed."""
    import socket

    real = socket.socket.connect

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError(f"test tried to reach the network: {host}")
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def test_env_is_actually_isolated():
    """If this fails, the repo .env is still leaking into the test process."""
    for var in _MINICODE_VARS:
        assert var not in os.environ, f"{var} leaked into tests"


def test_dotenv_file_is_not_committed():
    """The .env holds real credentials and must stay untracked."""
    root = Path(__file__).resolve().parent.parent
    assert (root / ".env.example").exists(), "no example to copy from"
    gitignore = root / ".gitignore"
    if gitignore.exists():
        assert ".env" in gitignore.read_text(encoding="utf-8"), ".env is not gitignored"
