"""Remote sandbox backends (deepagents Sandboxes).

LocalShellBackend runs every command on the host with no isolation -- fine for a
throwaway box, not for a machine holding your keys. A sandbox backend runs the
agent's shell inside someone else's VM, reached over the provider's API. Deep
Agents builds the whole filesystem toolset (read/write/ls/grep/glob) on top of a
single ``execute()``.

Every provider wants a different construction dance (a client object, then a
create call, and sometimes wrapping the result), so each entry gets an explicit
``_create_*`` function instead of a guessed constructor -- instantiating
``e2b.Sandbox()`` directly fails, it needs a live sandbox id.

Providers are imported on demand and auto-installed, so the base install stays
lean. See https://docs.langchain.com/oss/python/deepagents/sandboxes
"""

from __future__ import annotations

import subprocess
import sys

# name -> (pip package, import module, class/attr, env var it needs)
PROVIDERS: dict[str, tuple[str, str, str, str]] = {
    "e2b": ("e2b-code-interpreter", "e2b", "Sandbox", "E2B_API_KEY"),
    "daytona": ("daytona", "daytona", "Daytona", "DAYTONA_API_KEY"),
    "modal": ("modal", "modal", "Sandbox", "MODAL_TOKEN_ID"),
    "runloop": ("runloop", "runloop", "Runloop", "RUNLOOP_API_KEY"),
    "vercel": ("vercel", "vercel", "Sandbox", "VERCEL_TOKEN"),
    "langsmith": ("langsmith", "langsmith.sandbox", "Sandbox", "LANGSMITH_API_KEY"),
}


class SandboxUnavailable(RuntimeError):
    """Raised with an actionable message when a sandbox cannot be started."""


def _auto_install(pkg: str) -> bool:
    """Install a provider's SDK into the running venv so the user does not have
    to stop and run `uv sync --extra ...` themselves. Best effort."""
    for cmd in (
        [sys.executable, "-m", "pip", "install", "-q", pkg],
        ["uv", "pip", "install", "-q", pkg],
    ):
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600, check=False)
            if proc.returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


def _import_provider(name: str):
    """Import the provider module, installing its package first if needed."""
    pkg, mod_name, cls_name, env_var = PROVIDERS[name]
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
    except ImportError:
        if not _auto_install(pkg):
            raise SandboxUnavailable(
                f"sandbox {name!r} needs the '{pkg}' package.\n"
                f"  install:  uv sync --extra sandbox-{name}\n"
                f"  then set {env_var} in .env"
            ) from None
        mod = __import__(mod_name, fromlist=[cls_name])
    if not hasattr(mod, cls_name):
        raise SandboxUnavailable(f"{pkg} is installed but exposes no {cls_name!r}; the {name} API changed.")
    return getattr(mod, cls_name)


def available() -> list[str]:
    """Providers whose package is importable right now."""
    out = []
    for name, (pkg, mod_name, _cls, _env) in PROVIDERS.items():
        try:
            __import__(mod_name)
            out.append(name)
        except ImportError:
            continue
    return out


def _create_e2b(Sandbox):
    # e2b.Sandbox is a handle to a *running* sandbox; create() provisions one.
    return Sandbox.create()


def _create_daytona(Daytona):
    client = Daytona()
    box = client.create()
    # deepagents needs the sandbox object, not the manager client.
    return client.get(box.id) if hasattr(box, "id") else box


def _create_modal(Sandbox):
    return Sandbox.create()


def _create_runloop(Runloop):
    client = Runloop()
    for maker in ("create", "start"):
        if hasattr(client, maker):
            return getattr(client, maker)()
    return client


def _create_vercel(Sandbox):
    for maker in ("create",):
        if hasattr(Sandbox, maker):
            return getattr(Sandbox, maker)()
    raise SandboxUnavailable("vercel sandbox SDK exposes no create(); needs a newer vercel")


def _create_langsmith(Sandbox):
    for maker in ("create",):
        if hasattr(Sandbox, maker):
            return getattr(Sandbox, maker)()
    raise SandboxUnavailable("langsmith SDK has no Sandbox.create(); upgrade: uv pip install -U langsmith")


_FACTORIES = {
    "e2b": _create_e2b,
    "daytona": _create_daytona,
    "modal": _create_modal,
    "runloop": _create_runloop,
    "vercel": _create_vercel,
    "langsmith": _create_langsmith,
}


def build(name: str, setup_script: str | None = None):
    """Create a sandbox backend for `name`.

    Raises SandboxUnavailable with something the user can act on -- an install
    hint, a missing env var, or the provider's own error -- never a traceback.
    """
    if name not in PROVIDERS:
        raise SandboxUnavailable(f"unknown sandbox {name!r}; choose one of: {', '.join(sorted(PROVIDERS))}")
    env_var = PROVIDERS[name][3]
    factory = _FACTORIES[name]
    try:
        backend = factory(_import_provider(name))
    except SandboxUnavailable:
        raise
    except Exception as e:  # noqa: BLE001 - providers raise their own types
        raise SandboxUnavailable(
            f"could not start the {name} sandbox: {type(e).__name__}: {e}\n  check {env_var} in .env"
        ) from None

    if setup_script:
        try:
            backend.execute(setup_script)
        except Exception as e:  # noqa: BLE001
            raise SandboxUnavailable(f"sandbox setup script failed for {name}: {e}") from None
    return backend
