import pytest
from luna_rust._kwargs import translate_kwargs

def test_translate_kwargs(monkeypatch):
    monkeypatch.setattr("luna_rust._kwargs._to_julia_type", lambda k, v: v)
    kwargs = {"lambda0": 800e-9, "tau_fwhm": 10e-15, "energy": 1e-12, "shotnoise": False}
    translated = translate_kwargs(kwargs)
    assert translated == {"λ0": 800e-9, "τfwhm": 10e-15, "energy": 1e-12, "shotnoise": False}

def test_duplicate_kwargs(monkeypatch):
    monkeypatch.setattr("luna_rust._kwargs._to_julia_type", lambda k, v: v)
    kwargs = {"lambda0": 800e-9, "λ0": 800e-9, "tau_fwhm": 10e-15}
    with pytest.raises(ValueError, match="Duplicate kwarg: 'λ0' and 'λ0' both supplied."):
        translate_kwargs(kwargs)
