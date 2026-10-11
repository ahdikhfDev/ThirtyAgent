from __future__ import annotations

import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id          TEXT PRIMARY KEY,
  cwd         TEXT NOT NULL,
  title       TEXT,
  created_at  REAL NOT NULL,
  updated_at  REAL NOT NULL
);
"""


def derive_title(prompt: str, limit: int = 60) -> str:
    """A concrete, scannable title from the first user message.

    ponytail: no LLM call, no extra round trip, no network. The first prompt is
    already a good title 95% of the time -- it just needs cleaning up.
    """
    text = " ".join((prompt or "").split())
    if not text:
        return "(untitled)"
    text = text.strip("\"'`“”‘’ ")
    # A bare path is a terrible title ("check C:\\Users\\me\\AppData\\...\\minicode").
    # Keep the command words and swap the path for its last segment.
    text = shorten_paths(text)
    if len(text) > limit:
        head = text[: limit - 1]
        cut = head.rfind(" ")
        text = (head[:cut] if cut > 0 else head).rstrip(",.;:") + "…"
    return text or "(untitled)"


_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")


def leaf_name(value: str) -> str:
    """Last path segment, for a value that is a whole path.

    Unlike the regex above this copes with spaces in the path, which Windows
    user profiles routinely have ("C:\\Users\\Ahdi Khalida Fathir\\...").
    """
    raw = str(value or "").strip().rstrip("\\/")
    parts = [p for p in re.split(r"[\\/]", raw) if p]
    return parts[-1] if parts else raw


def shorten_paths(text: str) -> str:
    """C:\\a\\b\\c.txt -> c.txt, keeping the surrounding words intact."""

    def sub(m: re.Match) -> str:
        raw = m.group(0).rstrip("\\/")
        tail = re.split(r"[\\/]", raw)[-1]
        # a file has an extension, a folder does not; either is a better title
        # than the whole absolute path, so keep the leaf as-is
        return tail or "the project"

    return _PATH_RE.sub(sub, text)


class Store:
    """Session index. Conversation state lives in LangGraph's own sqlite checkpointer.

    ponytail: titles come from the first user message truncated, no LLM call.
    Upgrade path if summaries are wanted: summarize on /rename.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(SCHEMA)

    @contextmanager
    def _conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def create(self, session_id: str, cwd: Path, title: str = "") -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO sessions (id, cwd, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (session_id, str(cwd), title, now, now),
            )

    def touch(self, session_id: str, cwd: Path, title: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET updated_at = ?, cwd = ?, title = COALESCE(?, title) WHERE id = ?",
                (time.time(), str(cwd), title, session_id),
            )

    def rename(self, session_id: str, title: str) -> bool:
        with self._conn() as c:
            cur = c.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
            return cur.rowcount > 0

    def latest(self, cwd: Path | None = None) -> str | None:
        q = "SELECT id FROM sessions"
        args: tuple = ()
        if cwd is not None:
            q += " WHERE cwd = ?"
            args = (str(cwd),)
        q += " ORDER BY updated_at DESC LIMIT 1"
        with self._conn() as c:
            row = c.execute(q, args).fetchone()
        return row["id"] if row else None

    def list(self, cwd: Path | None = None, limit: int = 20) -> list[dict]:
        q = "SELECT * FROM sessions"
        args: tuple = ()
        if cwd is not None:
            q += " WHERE cwd = ?"
            args = (str(cwd),)
        q += " ORDER BY updated_at DESC LIMIT ?"
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, (*args, limit)).fetchall()]

    def delete(self, session_id: str) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def export(self, session_id: str, out: Path) -> Path:
        row = next((r for r in self.list(limit=10_000) if r["id"] == session_id), None)
        if row is None:
            with self._conn() as c:
                row = c.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
                row = dict(row) if row else None
        if row is None:
            raise FileNotFoundError(session_id)
        out.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return out


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
