from __future__ import annotations

import html
import re
import urllib.error
import urllib.request

from langchain.tools import tool

MAX_BYTES = 400_000
UA = "Mozilla/5.0 (compatible; ThirtyAgent/0.1)"

_BLOCKED = re.compile(r"<(script|style|noscript|template|svg)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_HEADING = re.compile(r"<h[1-3][^>]*>(.*?)</h[1-3]>", re.IGNORECASE | re.DOTALL)
_META_DESC = re.compile(
    r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', re.IGNORECASE | re.DOTALL
)
_BLOCK = re.compile(r"</?(p|div|section|article|br|li|tr|h[1-6]|pre|blockquote)\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(page: str) -> str:
    page = _COMMENT.sub(" ", page)
    page = _BLOCKED.sub(" ", page)

    title = _TITLE.search(page)
    desc = _META_DESC.search(page)
    headings = [_clean(h) for h in _HEADING.findall(page)]

    body = _BLOCK.sub("\n", page)
    body = _TAG.sub(" ", body)
    body = html.unescape(body)

    head = []
    if title:
        head.append(f"# {_clean(title.group(1))}")
    if desc:
        head.append(_clean(desc.group(1)))
    if headings:
        head.append("## Headings\n" + "\n".join(f"- {h}" for h in headings[:40]))

    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n\s*\n+", "\n\n", body).strip()
    return "\n".join(head) + "\n\n---\n\n" + body if head else body


def _clean(s: str) -> str:
    return " ".join(html.unescape(_TAG.sub("", s)).split())


def truncate(text: str, limit: int) -> str:
    """Cut at the cap and say so, so the model never reports a partial page as whole."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n... [truncated at {limit} chars of {len(text)}]"


@tool
def fetch_url(url: str, limit: int = 20_000) -> str:
    """Fetch a web page and return its readable text content.

    Use this whenever the user gives you a URL or asks about something on the
    internet. Returns the page title, meta description, headings, and body text.
    """
    if not url.startswith(("http://", "https://")):
        return f"Error: only http and https are supported, got {url!r}"

    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read(MAX_BYTES)
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} {e.reason} for {url}"
    except Exception as e:  # noqa: BLE001 - the model should see why, not a stack trace
        return f"Error: could not fetch {url}: {e}"

    charset = "utf-8"
    if "charset=" in ctype:
        charset = ctype.split("charset=")[-1].split(";")[0].strip() or "utf-8"
    text = raw.decode(charset, errors="replace")

    if "html" in ctype.lower() or text.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        text = html_to_text(text)
    elif len(text) > MAX_BYTES:
        text = text[:MAX_BYTES]

    if len(text) > limit:
        return truncate(text, limit)

    return text or "(empty response)"
