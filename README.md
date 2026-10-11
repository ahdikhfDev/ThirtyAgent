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

## Keys

| Key | Action |
| --- | --- |
| `Ctrl+K` | command palette |
| `Ctrl+O` | expand / fold tool output |