import pytest
from luna_rust._kwargs import translate_kwargs

def test_translate_kwargs_duplicate_lambda(monkeypatch):
    """
    Test that translate_kwargs correctly raises a ValueError when a duplicate kwarg
    (both ASCII and Unicode variants of the same parameter) is supplied.
    """
    # Mock _to_julia_type to avoid dependency on juliapkg/Julia backend for this pure unit test
    monkeypatch.setattr("luna_rust._kwargs._to_julia_type", lambda k, v: v)

    with pytest.raises(ValueError, match="Duplicate kwarg: 'λ0' and 'λ0' both supplied."):
        translate_kwargs({"lambda0": 800e-9, "λ0": 800e-9})

def test_translate_kwargs_basic(monkeypatch):
    import luna_rust._kwargs
    mock_jl = type('MockJL', (), {'Symbol': lambda x: f"Symbol({x})"})
    monkeypatch.setattr(luna_rust._kwargs, "_to_julia_type", lambda k, v: mock_jl.Symbol(v) if k in ("gas", "model", "pulseshape", "polarisation", "plasma", "ramanmodel", "modes") else v)
    kwargs = {"energy": 1.0, "duration": 10e-15}
    out = luna_rust._kwargs.translate_kwargs(kwargs)
    assert out["energy"] == 1.0
    assert out["τfwhm"] == 10e-15
    assert len(out) == 2

def test_translate_kwargs_symbol(monkeypatch):
    import luna_rust._kwargs
    mock_jl = type('MockJL', (), {'Symbol': lambda x: f"Symbol({x})"})
    monkeypatch.setattr(luna_rust._kwargs, "_to_julia_type", lambda k, v: mock_jl.Symbol(v) if k in ("gas", "model", "pulseshape", "polarisation", "plasma", "ramanmodel", "modes") else v)
    kwargs = {"gas": "He", "model": "GNLSE"}
    out = luna_rust._kwargs.translate_kwargs(kwargs)
    assert out["gas"] == "Symbol(He)"
    assert out["model"] == "Symbol(GNLSE)"

def test_translate_kwargs_modes(monkeypatch):
    import luna_rust._kwargs
    mock_jl = type('MockJL', (), {'Symbol': lambda x: f"Symbol({x})"})
    monkeypatch.setattr(luna_rust._kwargs, "_to_julia_type", lambda k, v: tuple(mock_jl.Symbol(x) if isinstance(x, str) else x for x in v) if k == "modes" else v)
    kwargs = {"modes": ["HE11", "HE12", 3]}
    out = luna_rust._kwargs.translate_kwargs(kwargs)
    assert out["modes"] == ("Symbol(HE11)", "Symbol(HE12)", 3)

def test_translate_kwargs_duplicate_duration(monkeypatch):
    import luna_rust._kwargs
    mock_jl = type('MockJL', (), {'Symbol': lambda x: f"Symbol({x})"})
    monkeypatch.setattr(luna_rust._kwargs, "_to_julia_type", lambda k, v: mock_jl.Symbol(v) if k in ("gas", "model", "pulseshape", "polarisation", "plasma", "ramanmodel", "modes") else v)
    kwargs = {"duration": 10e-15, "tau_fwhm": 15e-15}
    with pytest.raises(ValueError, match="Duplicate kwarg"):
        luna_rust._kwargs.translate_kwargs(kwargs)
