"""Unit tests for test_integration.py's skip-vs-fail guard.

Deliberately *not* integration-marked and Julia-free: the thing under test is
the guard that decides whether a missing backend is a skip or a failure, and a
bug in it can only show up as the whole integration module quietly skipping —
which is exactly how a broken backend passed CI on 2026-07-25. See
`_unavailable`'s docstring in test_integration.py.
"""
import pytest

# Sibling test module; pytest's default (prepend) import mode puts
# python/tests/ on sys.path. Importing it runs no Julia — the backend is only
# touched inside the `real_amalthea` fixture.
from test_integration import _unavailable


# pytest's outcome exceptions derive from BaseException, not Exception, so
# these must be caught by their concrete classes.
def test_unavailable_skips_when_not_required(monkeypatch):
    monkeypatch.delenv("AMALTHEA_REQUIRE_INTEGRATION", raising=False)
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _unavailable("no Julia on PATH")
    assert "no Julia on PATH" in str(excinfo.value)


def test_unavailable_fails_when_required(monkeypatch):
    monkeypatch.setenv("AMALTHEA_REQUIRE_INTEGRATION", "1")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        _unavailable("undefined symbol: native_compute_extra_stages")
    assert "undefined symbol" in str(excinfo.value)
    assert "AMALTHEA_REQUIRE_INTEGRATION=1" in str(excinfo.value)


@pytest.mark.parametrize("value", ["", "0", "true", "yes"])
def test_only_exactly_one_enables_required_mode(monkeypatch, value):
    # Strict "1" so a stray/legacy value can't silently arm a mode that turns
    # every environment without Julia into a red build.
    monkeypatch.setenv("AMALTHEA_REQUIRE_INTEGRATION", value)
    with pytest.raises(pytest.skip.Exception):
        _unavailable("no Julia on PATH")
