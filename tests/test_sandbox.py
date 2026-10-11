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


def test_providers_are_created_not_constructed():
    """Regression: `e2b.Sandbox()` raised 'missing 5 required positional
    arguments' because it is a handle to a *running* sandbox. Each provider needs
    its own creation call, so a bare constructor is never used."""
    import minicode.sandbox as sb

    calls = []

    class NeedsCreate:
        def __init__(self, *a, **kw):
            raise TypeError("SandboxBase.__init__() missing 5 required arguments")

        @classmethod
        def create(cls, **kw):
            calls.append("create")
            return "backend-from-create"

    monkey = sb._create_e2b
    assert monkey(NeedsCreate) == "backend-from-create"
    assert calls == ["create"]


def test_every_provider_has_a_factory():
    import minicode.sandbox as sb

    for name in sb.PROVIDERS:
        assert name in sb._FACTORIES, f"{name} has no factory"


def test_available_never_raises():
    assert isinstance(available(), list)


def test_setup_script_failure_is_reported(monkeypatch):
    """A broken setup script must not silently produce a half-configured sandbox."""
    import minicode.sandbox as sb

    monkeypatch.setitem(sb.PROVIDERS, "fake", ("json", "json", "dumps", "FAKE_KEY"))
    monkeypatch.setitem(sb._FACTORIES, "fake", lambda cls: _BadBackend())
    with pytest.raises(SandboxUnavailable) as e:
        sb.build("fake", setup_script="echo hi")
    assert "setup script failed" in str(e.value)


def test_setup_script_runs_on_success(monkeypatch):
    import minicode.sandbox as sb

    monkeypatch.setitem(sb.PROVIDERS, "fake2", ("json", "json", "dumps", "FAKE_KEY"))
    monkeypatch.setitem(sb._FACTORIES, "fake2", lambda cls: _GoodBackend())
    sb.build("fake2", setup_script="git clone x")
    assert _RAN == ["git clone x"]


class _BadBackend:
    def execute(self, cmd):
        raise RuntimeError("boom")


class _GoodBackend:
    def execute(self, cmd):
        _RAN.append(cmd)
        return "ok"


_RAN: list[str] = []
