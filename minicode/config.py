from __future__ import annotations

import copy
import json
import os
from pathlib import Path


def config_path() -> Path:
    # Imported lazily so tests can patch minicode.theme.config_dir.
    from .theme import config_dir

    return config_dir() / "config.json"


DEFAULT = {
    "models": {},  # alias -> {"base_url", "api_key_env", "model"}
    "active": None,  # alias of the model in use
    "mcp": {},  # name -> {"command","args","env"} | {"url","transport","headers"}
    "skills": {"enabled": True, "include": [], "ignore": []},
}

SECRET = "api_key"


def load() -> dict:
    # deepcopy: a shallow copy shares DEFAULT["mcp"], so add() would mutate the
    # module default and leak servers into every later load() in the process.
    cfg = copy.deepcopy(DEFAULT)
    p = config_path()
    if p.exists():
        cfg.update(json.loads(p.read_text(encoding="utf-8")))
    return cfg


def save(cfg: dict) -> Path:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    try:
        p.chmod(0o600)
    except OSError:
        pass  # Windows: ACLs, not POSIX bits
    return p


def _pick(*names: str) -> str:
    """First non-empty of THIRTYAGENT_* then the legacy MINICODE_* names."""
    dotenv = _env_file_values()
    for name in names:
        for source in (dotenv, os.environ):
            value = source.get(name)
            if value:
                return value
    return ""


def _env_file_values() -> dict:
    """Values from .env in the project, so /model can show what is actually set."""
    out = {}
    env = Path.cwd() / ".env"
    if not env.exists():
        return out
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def discover_models() -> dict:
    """Alias -> connection info. Explicit config wins over .env, which wins over env vars."""
    found = {}

    for alias, spec in load()["models"].items():
        found[alias] = dict(spec)

    base = _pick("THIRTYAGENT_BASE_URL", "MINICODE_BASE_URL", "OPENAI_BASE_URL")
    key = _pick("THIRTYAGENT_API_KEY", "MINICODE_API_KEY", "OPENAI_API_KEY")
    model = _pick("THIRTYAGENT_MODEL", "MINICODE_MODEL")

    if base and model:
        found.setdefault("default", {"base_url": base, SECRET: key or "", "model": model})

    return found


def active_alias() -> str | None:
    return load().get("active") or ("default" if "default" in discover_models() else None)


def resolve(alias: str | None = None) -> dict | None:
    models = discover_models()
    name = alias or active_alias()
    if not name or name not in models:
        return None
    return {"alias": name, **models[name]}
