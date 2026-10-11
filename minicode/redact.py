"""Secret protection.

Two layers, because file-tool guards alone are not enough: the agent also has a
shell, so `cat .env` bypasses any read_file denylist. Both layers are cheap.

  1. secret_paths()  - files the file tools must refuse to touch.
  2. SecretRedactor  - masks the *actual* secret values from anything printed,
                       so a value that slips through the shell still never
                       reaches the screen, the transcript, or LangSmith.
"""

from __future__ import annotations

import re
from pathlib import Path

# Files that hold credentials. Matched on the file name, not the full path, so
# a nested .env is caught too. The agent never needs to read these to work.
SECRET_NAMES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "credentials",
        "id_rsa",
        "id_ed25519",
    }
)
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".keystore", ".jks")


def is_secret_file(path: str) -> bool:
    name = Path(str(path).replace("\\", "/")).name
    if name in SECRET_NAMES:
        return True
    if any(name.endswith(s) for s in SECRET_SUFFIXES):
        return True
    # .env.<anything> (not .env.example, which is meant to be public)
    return name.startswith(".env.") and not name.endswith((".example", ".sample", ".template"))


def secret_paths(root: Path) -> list[Path]:
    return [p for p in (root / n for n in SECRET_NAMES) if p.exists()]


_MIN_SECRET_LEN = 8


class SecretRedactor:
    """Replace known secret values with a mask before anything is printed."""

    def __init__(self, values=None):
        vals = values if values is not None else self._from_env_file()
        # longest first so an overlapping short value cannot partially mask
        self._values = sorted({v for v in vals if v and len(v) >= _MIN_SECRET_LEN}, key=len, reverse=True)

    @staticmethod
    def _from_env_file() -> list[str]:
        vals: list[str] = []
        for env_file in (Path(".env"), Path(".env.local")):
            if not env_file.is_file():
                continue
            try:
                for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    vals.append(line.split("=", 1)[1].strip().strip("\"'"))
            except OSError:
                continue
        return vals

    def text(self, s: str) -> str:
        if not s or not self._values:
            return s
        for v in self._values:
            if v in s:
                s = s.replace(v, "«redacted»")
        return s

    def rich_text(self, t):
        """Mask a rich Text in place, keeping its styles."""
        if t is not None and getattr(t, "plain", None):
            masked = self.text(t.plain)
            if masked != t.plain:
                t.plain = masked
        return t


# tool name -> the arg key that holds a path
_PATH_ARGS = ("file_path", "path", "notebook_path")


def _blocked_reason(tool: str, args: dict) -> str | None:
    if not isinstance(args, dict):
        return None
    for key in _PATH_ARGS:
        val = args.get(key)
        if val and is_secret_file(str(val)):
            return f"refused: {tool} cannot open the credential file {val}"
    if tool == "execute":
        cmd = " ".join(str(v) for v in args.values())
        if is_secret_file(cmd.strip().strip("'\"")) or any(
            is_secret_file(tok) for tok in re.split(r"[|;&<>]+|\s+", cmd) if tok
        ):
            return "refused: the shell command touches a credential file"
    return None


def secret_guard_middleware():
    """Block tool calls that would hand a credential file to the model.

    The display redactor is the second line of defence; this is the first, so
    the secret never reaches the provider at all. Shell commands are checked
    token-wise because `cat .env` slips past any file-path check.
    """
    from langchain.agents.middleware import AgentMiddleware

    class SecretGuard(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            reason = _blocked_reason(request.tool_call["name"], request.tool_call.get("args") or {})
            if reason:
                from langchain_core.messages import ToolMessage

                return ToolMessage(content=reason, tool_call_id=request.tool_call["id"], status="error")
            return handler(request)

    return SecretGuard()
