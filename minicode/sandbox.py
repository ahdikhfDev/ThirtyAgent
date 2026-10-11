"""Remote sandbox backends (deepagents Sandboxes).

LocalShellBackend runs every command on the host with no isolation -- fine for a
throwaway box, not for a machine holding your keys. A sandbox backend runs the
agent's shell inside someone else's VM/container, reached over the provider's
API. Deep Agents builds the whole filesystem toolset (read/write/ls/grep/glob)
on top of a single ``execute()``, so a provider only has to implement that.

Providers are imported on demand: None are required unless you actually pick
one, so the base install stays lean. See https://docs.langchain.com/oss/python/
deepagents/sandboxes
"""

from __future__ import annotations

from collections.abc import Callable

# name -> (module to import, class/attr on it, env var it needs)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "daytona": ("daytona", "Daytona", "DAYTONA_API_KEY"),
    "modal": ("modal", "Sandbox", "MODAL_TOKEN_ID"),
    "runloop": ("runloop", "Runloop", "RUNLOOP_API_KEY"),
    "e2b": ("e2b", "Sandbox", "E2B_API_KEY"),
    "vercel": ("vercel", "Sandbox", "VERCEL_TOKEN"),
    "langsmith": ("langsmith.sandbox", "Sandbox", "LANGSMITH_API_KEY"),
}

# Extra names accepted for people typing the family name.
ALIASES = {"modal": "modal", "daytona": "daytona", "runloop": "runloop", "e2b": "e2b"}


class SandboxUnavailable(RuntimeError):
    """Raised with an actionable message when a provider is not installed."""


def available() -> list[str]:
    """Providers whose package is importable right now."""
    out = []
    for name, (mod, _cls, _env) in PROVIDERS.items():
        try:
            __import__(mod)
            out.append(name)
        except ImportError:
            continue
    return out


def _auto_install(mod_name: str) -> bool:
    """Install a provider's SDK into the running venv, so the user does not have
    to stop and run `uv sync --extra ...` themselves. Best effort; returns
    whether it worked.
    """
    import subprocess
    import sys

    for cmd in (
        [sys.executable, "-m", "pip", "install", "-q", mod_name],
        ["uv", "pip", "install", "-q", mod_name],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _build_one(name: str) -> object:
    mod_name, cls_name, env_var = PROVIDERS[name]
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
    except ImportError:
        if _auto_install(mod_name):
            try:
                mod = __import__(mod_name, fromlist=[cls_name])
            except ImportError:
                mod = None
        else:
            mod = None
        if mod is None:
            raise SandboxUnavailable(
                f"sandbox {name!r} needs the '{mod_name}' package.\n"
                f"  install:  uv sync --extra sandbox-{name}   (or: uv pip install {mod_name})\n"
                f"  then set {env_var} in .env"
            ) from None
    if not hasattr(mod, cls_name):
        raise SandboxUnavailable(
            f"{mod_name} is installed but has no {cls_name!r}; the {name} API may have changed."
        )
    return getattr(mod, cls_name)


def build(name: str, setup_script: str | None = None) -> object:
    """Create a sandbox backend for `name` (one of PROVIDERS).

    Raises SandboxUnavailable with an install hint if the provider's package is
    missing, so the CLI can print something the user can act on.
    """
    name = ALIASES.get(name, name)
    if name not in PROVIDERS:
        raise SandboxUnavailable(f"unknown sandbox {name!r}; choose one of: {', '.join(sorted(PROVIDERS))}")
    factory: Callable[[], object] = _build_one(name)
    try:
        backend = factory()
    except SandboxUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - providers raise their own types
        # Missing/bad API key, quota, wrong region... all arrive as the
        # provider's own exception, which means nothing to the user.
        raise SandboxUnavailable(
            f"could not start the {name} sandbox: {type(e).__name__}: {e}\n"
            f"  check {PROVIDERS[name][2]} in .env"
        ) from None
    if setup_script:
        try:
            backend.execute(setup_script)
        except Exception as e:
            raise SandboxUnavailable(f"sandbox setup script failed for {name}: {e}") from e
    return backend
