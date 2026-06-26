import sys
import unittest.mock as mock
import pytest

# Mock julia to avoid segfaults and dependency on juliacall
mock_julia = mock.MagicMock()
mock_jl = mock.MagicMock()
mock_jl.Symbol = lambda x: f"Symbol({x})" # Simple mock for julia symbols
mock_julia.get_julia.return_value = (mock_jl, None)

sys.modules['luna_rust._julia'] = mock_julia

from luna_rust._kwargs import translate_kwargs

def test_translate_kwargs_basic():
    kwargs = {"energy": 1.0, "duration": 10e-15}
    out = translate_kwargs(kwargs)
    assert out["energy"] == 1.0
    assert out["τfwhm"] == 10e-15
    assert len(out) == 2

def test_translate_kwargs_symbol():
    kwargs = {"gas": "He", "model": "GNLSE"}
    out = translate_kwargs(kwargs)
    assert out["gas"] == "Symbol(He)"
    assert out["model"] == "Symbol(GNLSE)"

def test_translate_kwargs_modes():
    kwargs = {"modes": ["HE11", "HE12", 3]}
    out = translate_kwargs(kwargs)
    assert out["modes"] == ("Symbol(HE11)", "Symbol(HE12)", 3)

def test_translate_kwargs_duplicate():
    kwargs = {"duration": 10e-15, "tau_fwhm": 15e-15}
    with pytest.raises(ValueError, match="Duplicate kwarg"):
        translate_kwargs(kwargs)
