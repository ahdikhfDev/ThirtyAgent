# ThirtyAgent

A polished, OpenCode-Zen-style coding-agent TUI for your terminal.

## Quickstart

```powershell
git clone https://github.com/ahdikhfDev/ThirtyAgent.git
cd ThirtyAgent
uv sync
uv run thirtyagent --cwd .
```

## Configuration

Copy `.env.example` to `.env` and fill in your endpoint:

```
MINICODE_BASE_URL=https://your-host/v1
MINICODE_API_KEY=sk-...
MINICODE_MODEL=your-model-id
```

`.env` is git-ignored; never commit real credentials.

## Remote sandbox (recommended)

By default the agent runs shell commands on your host with no isolation. To run
them somewhere else instead:

```powershell
uv sync --extra sandbox-modal      # daytona | modal | runloop | e2b | vercel | langsmith
$env:MODAL_TOKEN_ID = "..."
uv run thirtyagent --sandbox modal --sandbox-setup ./setup.sh
```

The agent's `execute`, file tools and git all run inside the remote sandbox; your
machine only holds the API key. See the
[LangChain sandbox docs](https://docs.langchain.com/oss/python/deepagents/sandboxes).

## Keys

| Key | Action |
| --- | --- |
| `Ctrl+K` | command palette |
| `Ctrl+O` | expand / fold tool output |