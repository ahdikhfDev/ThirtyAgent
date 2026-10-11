from __future__ import annotations

import fnmatch
from pathlib import Path

import yaml

from .theme import config_dir

# omp-compatible discovery: user scope + project scope, one skill per directory.
USER_DIRS = (
    Path.home() / ".agents" / "skills",
    Path.home() / ".agent" / "skills",
    Path.home() / ".claude" / "skills",
    Path.home() / ".codex" / "skills",
    Path.home() / ".config" / "opencode" / "skills",
    config_dir() / "skills",
)
PROJECT_DIRS = (
    Path(".agents") / "skills",
    Path(".agent") / "skills",
    Path(".claude") / "skills",
    Path(".codex") / "skills",
    Path(".opencode") / "skills",
    Path(".thirtyagent") / "skills",
)


class Skill:
    def __init__(self, name: str, path: Path, description: str, scope: str):
        self.name = name
        self.path = path
        self.description = description
        self.scope = scope

    def __repr__(self) -> str:
        return f"Skill({self.name!r}, {self.scope})"


def _read_meta(md: Path) -> tuple[str, str]:
    """Frontmatter name/description; the directory name is the fallback."""
    name = md.parent.name
    desc = ""
    try:
        head = md.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return name, desc
    if not head.startswith("---"):
        return name, desc
    _, _, rest = head.partition("---")
    fm, _, _ = rest.partition("---")
    try:
        data = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        return name, desc
    if isinstance(data, dict):
        name = data.get("name") or name
        desc = (data.get("description") or "").strip()
    return str(name), desc


def _settings() -> dict:
    # A broken config.json must not stop the agent from starting; no skills is
    # the safe fallback.
    from .config import load

    try:
        return load().get("skills") or {}
    except (OSError, ValueError):
        return {}


def discover(root: Path) -> list[Skill]:
    """Later scopes win on name collision, matching omp's precedence rule."""
    settings = _settings()

    found: dict[str, Skill] = {}
    roots = [(d, "user") for d in USER_DIRS] + [(root / d, "project") for d in PROJECT_DIRS]

    for base, scope in roots:
        if not base.is_dir():
            continue
        for md in sorted(base.glob("*/SKILL.md")):
            name, desc = _read_meta(md)
            found[name] = Skill(name, md, desc, scope)

    names = list(found)
    if settings.get("include"):
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in settings["include"])]
    if settings.get("ignore"):
        names = [n for n in names if not any(fnmatch.fnmatch(n, p) for p in settings["ignore"])]
    return sorted((found[n] for n in names), key=lambda s: s.name)


def skill_source_paths(root: Path) -> list[str]:
    """Backend-readable paths for each skills dir.

    Project dirs sit under root and are read directly. User dirs live outside
    root, so the agent backend must route to them (see build_agent's
    CompositeBackend) -- the virtual prefix is what the model sees.
    """
    project = sorted(str(root / d) for d in PROJECT_DIRS if (root / d).is_dir())
    return project


def user_skill_roots() -> list[Path]:
    return [d for d in USER_DIRS if d.is_dir()]
