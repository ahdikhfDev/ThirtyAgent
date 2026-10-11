"""fetch_url behaviour. Uses a local HTTP server so tests never hit the network."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from minicode.fetch import fetch_url, html_to_text

PAGE = """<!doctype html><html><head>
<title>  Thirty Store </title>
<meta name="description" content="We sell thirty things.">
<script>var tracking = "ignore me";</script>
<style>body{color:red}</style>
</head><body>
<h1>Thirty Store</h1>
<h2>Catalog</h2>
<p>First &amp; foremost, we sell <b>thirty</b> things.</p>
<!-- a comment that should vanish -->
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_error(404, "Not Found")
            return
        if self.path == "/boom":
            self.send_error(500, "Server Error")
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_extracts_title_description_and_text(base_url):
    out = fetch_url.invoke({"url": f"{base_url}/"})
    assert "Thirty Store" in out
    assert "We sell thirty things." in out
    assert "first & foremost" in out.lower() or "First & foremost" in out
    assert "- Catalog" in out, "headings were dropped"


def test_strips_scripts_styles_and_comments(base_url):
    out = fetch_url.invoke({"url": f"{base_url}/"})
    assert "tracking" not in out, "script body leaked"
    assert "color:red" not in out, "style body leaked"
    assert "should vanish" not in out, "comment leaked"


def test_reports_http_errors_instead_of_raising(base_url):
    assert "404" in fetch_url.invoke({"url": f"{base_url}/missing"})
    assert "500" in fetch_url.invoke({"url": f"{base_url}/boom"})


def test_refuses_non_http_schemes():
    for bad in ("file:///etc/passwd", "ftp://x/y", "javascript:alert(1)"):
        out = fetch_url.invoke({"url": bad})
        assert out.startswith("Error"), f"{bad} was not refused: {out!r}"


def test_unreachable_host_returns_a_message_not_a_crash():
    out = fetch_url.invoke({"url": "http://127.0.0.1:1/nope"})
    assert out.startswith("Error"), out


def test_truncation_marks_the_cut():
    """The cap must be visible, so the model never reports a partial page as complete."""
    from minicode.fetch import truncate

    assert "truncated at 100 chars of 500" in truncate("x" * 500, 100)
    assert "truncated" not in truncate("short", 100)
    assert truncate("exact", 5) == "exact"


def test_html_to_text_handles_empty_and_plain():
    assert html_to_text("<html><body></body></html>").strip() == ""
    assert "hello" in html_to_text("<p>hello</p>")
