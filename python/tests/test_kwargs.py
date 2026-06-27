import pytest
from luna_rust._kwargs import translate_kwargs

def test_translate_kwargs_duplicate(monkeypatch):
    """
    Test that translate_kwargs correctly raises a ValueError when a duplicate kwarg
    (both ASCII and Unicode variants of the same parameter) is supplied.
    """
    # Mock _to_julia_type to avoid dependency on juliapkg/Julia backend for this pure unit test
    monkeypatch.setattr("luna_rust._kwargs._to_julia_type", lambda k, v: v)

    with pytest.raises(ValueError, match="Duplicate kwarg: 'λ0' and 'λ0' both supplied."):
        translate_kwargs({"lambda0": 800e-9, "λ0": 800e-9})
