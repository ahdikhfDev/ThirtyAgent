"""Adapter turning a provider SDK sandbox into a deepagents backend.

Provider SDKs expose their sandbox as a live object with a `commands.run(...)`
style call. Deep Agents wants a backend implementing four members: ``id``,
``execute``, ``upload_files`` and ``download_files``; it builds ls/read/write/
grep/glob on top of those. One adapter covers every provider because they all
agree on the shape, and a provider that differs only needs a hook here.
"""

from __future__ import annotations

import logging

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.sandbox import BaseSandbox

MAX_UPLOAD_BYTES = 2_000_000


def _run(sandbox, command: str, timeout: int | None):
    """Call whichever execute API this SDK exposes, normalising the return."""
    runner = getattr(sandbox, "commands", None) or sandbox
    for name in ("run", "exec", "execute", "run_command"):
        fn = getattr(runner, name, None)
        if fn is None:
            continue
        result = fn(command, timeout=timeout or 120)
        return _result_fields(result)
    raise TypeError(f"{type(sandbox).__name__} exposes no way to run a command")


def _result_fields(result):
    """Pull output/exit code out of whatever shape the SDK returned."""
    if isinstance(result, str):
        return result, 0
    if isinstance(result, tuple):
        out = result[0] if result else ""
        code = result[1] if len(result) > 1 else 0
        return str(out), int(code or 0)
    if isinstance(result, dict):
        return str(result.get("stdout", result.get("output", ""))), int(result.get("exit_code", 0) or 0)
    out = getattr(result, "stdout", None) or getattr(result, "output", None)
    code = getattr(result, "exit_code", None)
    if out is None:
        out = getattr(result, "result", None)
    return str(out or ""), int(code or 0)


def _read_file(sandbox, path: str) -> str:
    files = getattr(sandbox, "files", None)
    if files is not None and hasattr(files, "read"):
        return str(files.read(path))
    out, _ = _run(sandbox, f"cat {path!r}")
    return out


def _write_file(sandbox, path: str, content: str) -> None:
    files = getattr(sandbox, "files", None)
    if files is not None and hasattr(files, "write"):
        files.write(path, content)
        return
    _run(sandbox, f"cat > {path!r} <<'EOF'\n{content}\nEOF")


class SdkSandboxBackend(BaseSandbox):
    """Wraps a live provider sandbox object for deepagents."""

    def __init__(self, sandbox: object, name: str = "sandbox"):
        self._sb = sandbox
        self._name = name

    @property
    def id(self) -> str:
        for attr in ("id", "name", "sandbox_id"):
            val = getattr(self._sb, attr, None)
            if isinstance(val, str) and val:
                return val
        return self._name

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        out, code = _run(self._sb, command, timeout)
        return ExecuteResponse(output=out, exit_code=code, truncated=False)

    def upload_files(self, files: list[tuple[str, str]]) -> None:
        """files is [(remote_path, local_path)] in deepagents; we only support
        remote content, so callers should use write() instead."""
        raise NotImplementedError("upload_files needs local paths; use write()")

    def download_files(self, paths: list[str], destination: str | None = None) -> None:
        raise NotImplementedError("download_files is not wired up yet")

    # deepagents uses these two for write/edit; override with plain file IO so
    # content does not have to round-trip through the shell.
    def write(self, path: str, content: str) -> None:
        _write_file(self._sb, path, content)

    def read(self, path: str, offset: int = 0, limit: int | None = None) -> object:
        return _read_file(self._sb, path)

    def kill(self) -> None:
        """Best-effort teardown: a sandbox that outlives the session costs money."""
        for name in ("kill", "close", "stop", "shutdown"):
            fn = getattr(self._sb, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - teardown never raises upward
                    logging.getLogger(__name__).debug("sandbox %s teardown failed", name)
                return
