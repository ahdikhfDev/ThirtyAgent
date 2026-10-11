from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .redact import secret_guard_middleware

load_dotenv()

MEMORY_FILES = ("AGENTS.md", "CLAUDE.md", ".thirtyagent/AGENTS.md")

USER_SKILL_PREFIX = "/skills-user/"

WRITE_TOOLS = ("write_file", "edit_file", "delete", "execute")


def build_model(alias: str | None = None, temperature: float = 0.0) -> ChatOpenAI:
    from .config import SECRET, resolve

    spec = resolve(alias)
    if spec is None:
        raise SystemExit(
            "No model configured.\n"
            "Set MINICODE_BASE_URL / MINICODE_API_KEY / MINICODE_MODEL in .env,\n"
            "or run: thirtyagent /model add <alias>"
        )

    missing = [k for k in ("base_url", "model", SECRET) if not spec.get(k)]
    if missing:
        raise SystemExit(
            f"Model {spec['alias']!r} is missing: {', '.join(missing)}\n"
            "Fix it with: thirtyagent /model add " + spec["alias"]
        )

    return ChatOpenAI(
        model=spec["model"],
        base_url=spec["base_url"],
        api_key=spec[SECRET],
        temperature=temperature,
        max_retries=3,
    )


SYSTEM_PROMPT = """\
You are a coding agent working directly in the user's terminal.

- Files are addressed by VIRTUAL paths rooted at `/`, not OS paths. `/src/a.py` \
means `<the project directory>/src/a.py`. Never pass `C:\\...`, `D:/...`, or any \
drive-letter path to a file tool; those are rejected. To reach the project, call \
`ls` with `/` first. Shell commands are the exception: they run on the real \
machine and do take OS paths.
- When the user gives you a URL or asks about something online, call \
`fetch_url` instead of guessing. Never invent page contents.
- Read before you write. Use `ls` and `grep` to understand existing code and \
conventions instead of guessing.
- Make the smallest change that solves the task. No unrequested refactors, no \
new dependencies, no boilerplate.
- Trace the real flow before patching. Fix the shared function, not just the \
symptom that was reported.
- After changing code, run the project's own lint/typecheck/test commands \
before claiming it works.
- Match the surrounding style: naming, comment density, error handling.

Be concise. Report what you changed and what you verified.
"""

PLAN_PROMPT = """\
You are in PLAN MODE. Do not modify anything.

File tools take virtual paths rooted at `/`, never OS paths or drive letters.

Investigate and produce a concrete implementation plan. Read as much as you
need to be confident, but make no edits and run no mutating commands.

End your reply with a numbered plan: the files to touch, what changes in each,
and the command that proves it works.
"""


def find_memory(root: Path) -> list[str]:
    """Project context files, if present (omp's /memory)."""
    return [str(root / name) for name in MEMORY_FILES if (root / name).is_file()]


def build_agent(
    root_dir: Path,
    *,
    approve: bool = False,
    checkpointer=None,
    plan_mode: bool = False,
    model=None,
    model_alias: str | None = None,
    use_skills: bool = True,
    mcp_tools: list | None = None,
    max_model_calls: int | None = None,
    redact_pii: bool = False,
    rubric: str | None = None,
    sandbox: str | None = None,
    sandbox_setup: str | None = None,
):
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend, LocalShellBackend
    from deepagents.middleware import FilesystemMiddleware
    from langchain.agents.middleware import TodoListMiddleware

    from .fetch import fetch_url
    from .skills import skill_source_paths, user_skill_roots
    from .usage import budget_middleware, pii_middleware

    kwargs = {}
    if approve:
        # Pause before anything that mutates the machine.
        kwargs["interrupt_on"] = {
            "write_file": True,
            "edit_file": True,
            "delete": True,
            "execute": True,
        }

    middleware = [TodoListMiddleware()]
    # fetch_url is the only tool we author; the rest come from deepagents.
    extra_tools = [fetch_url] + list(mcp_tools or [])
    skill_paths = skill_source_paths(root_dir) if use_skills else []
    user_roots = user_skill_roots() if use_skills else []

    if max_model_calls:
        limit = budget_middleware(max_model_calls)
        if limit is not None:
            middleware.append(limit)
    if redact_pii:
        middleware.extend(pii_middleware())

    middleware.append(secret_guard_middleware())

    if plan_mode:
        # No shell tool at all, and only read tools: plan mode is read-only.
        # (FilesystemMiddleware rejects permissions on executables backends, so
        #  the read-only allowlist is what actually drops write_file/execute.)
        backend = FilesystemBackend(root_dir=str(root_dir), virtual_mode=True)
        middleware.append(
            FilesystemMiddleware(
                backend=backend,
                tools=["read_file", "ls", "glob", "grep"],
            )
        )
    else:
        if sandbox:
            # Remote sandbox: the agent's shell and filesystem live off-box, so
            # a stray `rm -rf` or a prompt injection cannot touch the host.
            from .sandbox import build as build_sandbox

            backend = build_sandbox(sandbox, sandbox_setup)
        else:
            backend = LocalShellBackend(root_dir=str(root_dir), inherit_env=True)

    if user_roots:
        # User-scope skills live outside --cwd; route them so the agent can read
        # them without the backend refusing a path outside its root.
        from deepagents.backends import CompositeBackend

        routes = {
            f"{USER_SKILL_PREFIX}{i}/": FilesystemBackend(root_dir=str(p), virtual_mode=True)
            for i, p in enumerate(user_roots)
        }
        backend = CompositeBackend(default=backend, routes=routes)
        skill_paths += [f"{USER_SKILL_PREFIX}{i}" for i in range(len(user_roots))]

    if rubric:
        # Self-evaluation loop: the agent is re-run until the rubric passes or
        # max_iterations is hit. Opt-in only -- it costs extra model calls.
        from deepagents.middleware import RubricMiddleware

        middleware.append(RubricMiddleware(model=model or build_model(model_alias), max_iterations=3))

    memory = find_memory(root_dir)

    return create_deep_agent(
        model=model or build_model(model_alias),
        system_prompt=PLAN_PROMPT if plan_mode else SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        memory=memory or None,
        skills=skill_paths or None,
        middleware=middleware,
        tools=extra_tools,
        **kwargs,
    )
