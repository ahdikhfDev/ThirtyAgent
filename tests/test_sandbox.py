import pytest

from minicode.sandbox import PROVIDERS, SandboxUnavailable, available, build


def test_registry_lists_the_documented_providers():
    """Names come straight from the LangChain sandbox docs."""
    for name in ("daytona", "modal", "runloop", "e2b", "vercel", "langsmith"):
        assert name in PROVIDERS, name


def test_unknown_sandbox_lists_the_options():
    with pytest.raises(SandboxUnavailable) as e:
        build("nope")
    msg = str(e.value)
    for name in ("daytona", "modal"):
        assert name in msg, msg


def test_missing_package_gives_an_install_hint():
    """The error has to be actionable, not just ImportError."""
    if "daytona" in available():
        pytest.skip("daytona is installed in this environment")
    with pytest.raises(SandboxUnavailable) as e:
        build("daytona")
    msg = str(e.value)
    assert "daytona" in msg
    assert "install" in msg.lower()


def test_available_never_raises():
    assert isinstance(available(), list)


def test_setup_script_failure_is_reported(monkeypatch):
    """A broken setup script must not silently produce a half-configured sandbox."""
    import minicode.sandbox as sb

    monkeypatch.setitem(sb.PROVIDERS, "fake", ("json", "dumps", "FAKE_KEY"))

    class BadBackend:
        def execute(self, cmd):
            raise RuntimeError("boom")

    monkeypatch.setattr(sb, "_build_one", lambda name: BadBackend)
    with pytest.raises(SandboxUnavailable) as e:
        sb.build("fake", setup_script="echo hi")
    assert "setup script failed" in str(e.value)


def test_setup_script_runs_on_success(monkeypatch):
    import minicode.sandbox as sb

    ran = []

    class GoodBackend:
        def execute(self, cmd):
            ran.append(cmd)
            return "ok"

    monkeypatch.setitem(sb.PROVIDERS, "fake2", ("json", "dumps", "FAKE_KEY"))
    monkeypatch.setattr(sb, "_build_one", lambda name: GoodBackend)
    sb.build("fake2", setup_script="git clone x")
    assert ran == ["git clone x"]
