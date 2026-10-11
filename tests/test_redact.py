from pathlib import Path

from minicode.redact import SecretRedactor, is_secret_file


def test_secret_file_detection():
    for secret in ("/.env", "/proj/.env", "C:/x/.env.local", "/a/id_rsa", "/a/cert.pem", "/a/k.jks"):
        assert is_secret_file(secret), secret
    for ok in ("/.env.example", "/.env.sample", "/src/app.py", "/README.md", "/.envrc"):
        assert not is_secret_file(ok), ok


def test_redactor_masks_real_env_values(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "API_KEY=supersecret12345\nEMPTY=\nSHORT=abc\n# note=skipme\n", encoding="utf-8"
    )
    r = SecretRedactor()
    out = r.text("key is supersecret12345 here")
    assert "supersecret12345" not in out
    assert "redacted" in out
    # too-short and comment values are not treated as secrets
    assert "abc" in r.text("abc")
    assert "skipme" in r.text("skipme")


def test_redactor_does_not_mask_ordinary_text(tmp_path: Path):
    (tmp_path / ".env").write_text("TOKEN=realtoken987\n", encoding="utf-8")
    r = SecretRedactor()
    assert r.text("just a normal line of code") == "just a normal line of code"


def test_redactor_with_no_env_is_a_noop():
    r = SecretRedactor(values=[])
    assert r.text("anything at all") == "anything at all"


def test_redactor_rich_text_preserves_style(tmp_path: Path):
    from rich.text import Text

    (tmp_path / ".env").write_text("SECRET=abcd1234efgh\n", encoding="utf-8")
    r = SecretRedactor()
    t = Text("value=abcd1234efgh", style="muted")
    out = r.rich_text(t)
    assert "abcd1234efgh" not in out.plain
    assert out.style == "muted"


def test_tool_output_is_redacted_end_to_end(tmp_path: Path, monkeypatch):
    """The redaction must be wired into the TUI, not just exist as a helper."""

    import minicode.cli as C

    monkeypatch.setattr(C, "_redactor", SecretRedactor(values=["leakedtoken1234"]))

    class Msg:
        name = "execute"
        content = "output leakedtoken1234 more"
        status = "success"
        artifact = None

    lines = C._tool_result_lines(Msg())
    joined = "".join(getattr(x, "plain", str(x)) for x in lines)
    assert "leakedtoken1234" not in joined
    assert "redacted" in joined


def test_no_mojibake_in_source_files():
    """Regression: the palette searchbar shipped with U+00E2 U+2013 U+008F -- the
    UTF-8 bytes of U+258F stored as latin-1. It rendered as 'a-<8f>' garbage."""
    import pathlib

    bad = chr(0xE2) + chr(0x2013) + chr(0x8F)
    for p in list(pathlib.Path("minicode").glob("*.py")) + list(pathlib.Path("tests").glob("*.py")):
        s = p.read_text(encoding="utf-8")
        assert bad not in s, f"{p} contains mojibake"
        assert chr(0xE2) not in s, f"{p} contains a stray latin-1 e-acute"


def test_use_utf8_console_is_callable_and_safe():
    """It runs before any output; it must never raise, even with no console."""
    from minicode.cli import use_utf8_console

    use_utf8_console()  # must not raise


def test_card_title_is_redacted(tmp_path: Path, monkeypatch):
    import minicode.cli as C

    monkeypatch.setattr(C, "_redactor", SecretRedactor(values=["supersecret9999"]))
    title = C._card_title("execute", {"command": "curl -H 'k: supersecret9999'"})
    assert "supersecret9999" not in title.plain


def test_blocked_reason_covers_file_tools_and_shell():
    from minicode.redact import _blocked_reason

    for tool, args in (
        ("read_file", {"file_path": "/.env"}),
        ("write_file", {"file_path": "/.env.local", "content": "x"}),
        ("delete", {"file_path": "/proj/id_rsa"}),
        ("execute", {"command": "cat .env"}),
        ("execute", {"command": "type .env | more"}),
    ):
        assert _blocked_reason(tool, args), f"{tool} {args} should be blocked"


def test_blocked_reason_allows_normal_work():
    from minicode.redact import _blocked_reason

    for tool, args in (
        ("read_file", {"file_path": "/src/app.py"}),
        ("execute", {"command": "pytest -q"}),
        ("execute", {"command": "git log > out.txt"}),
        ("read_file", {"file_path": "/.env.example"}),
    ):
        assert _blocked_reason(tool, args) is None, f"{tool} {args} should be allowed"


def test_secret_guard_middleware_blocks_and_passes():
    from minicode.redact import secret_guard_middleware

    mw = secret_guard_middleware()
    called = []

    class Req:
        def __init__(self, name, args, tid="t1"):
            self.tool_call = {"name": name, "args": args, "id": tid}

    # blocked: handler never runs
    out = mw.wrap_tool_call(Req("read_file", {"file_path": "/.env"}), lambda r: called.append(1))
    assert "refused" in out.content.lower()
    assert out.status == "error"
    assert not called, "handler must not run for a blocked call"

    # allowed: handler runs
    marker = object()
    seen = []
    out = mw.wrap_tool_call(Req("execute", {"command": "ls"}), lambda r: seen.append(r) or marker)
    assert seen and out is marker, "allowed calls must reach the handler"


def test_agent_wires_the_secret_guard():
    """The guard is pointless unless build_agent actually installs it."""
    import inspect

    import minicode.agent as A

    src = inspect.getsource(A.build_agent)
    assert "secret_guard_middleware()" in src
