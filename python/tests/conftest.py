import pytest
import sys
import unittest.mock
import numpy as np

# We mock julia, Luna, and also the PyCall/juliacall boundaries required to not trigger Julia
mock_julia = unittest.mock.MagicMock()
mock_Luna = unittest.mock.MagicMock()

# Make isa return True only for Arrays
def mock_isa(val, t):
    if t == "AbstractArray":
        return True
    return False

mock_julia.isa.side_effect = mock_isa
mock_julia.haskey.return_value = True
mock_julia.Dict = "Dict"
mock_julia.AbstractArray = "AbstractArray"

mock_module = unittest.mock.MagicMock()
mock_module.get_julia.return_value = (mock_julia, mock_Luna)

class MockJuliaDict:
    def __init__(self, d):
        self.d = d
    def __getitem__(self, k):
        return self.d[k]
    def keys(self):
        return self.d.keys()

@pytest.fixture(autouse=True)
def mock_julia_backend(monkeypatch):
    import luna_rust
    import luna_rust.output
    # Mock at the dict level so get_julia returns our mock
    monkeypatch.setitem(sys.modules, 'luna_rust._julia', mock_module)

    # Also patch luna_rust.__init__ uses of get_julia if they already bound it
    monkeypatch.setattr(luna_rust, "get_julia", mock_module.get_julia)
    monkeypatch.setattr(luna_rust.output, "get_julia", mock_module.get_julia, raising=False)

    # Mock Luna functions returning a mock dict
    def mock_prop_capillary(*args, **kwargs):
        return MockJuliaDict({"Eω": np.zeros((10, 10))})

    def mock_prop_gnlse(*args, **kwargs):
        return MockJuliaDict({"Eω": np.zeros((10, 10))})

    mock_Luna.prop_capillary.side_effect = mock_prop_capillary
    mock_Luna.prop_gnlse.side_effect = mock_prop_gnlse

    # Needs to pretend it's a dict for __contains__
    monkeypatch.setattr(luna_rust.output.LunaOutput, "__contains__", lambda self, k: True)

    import luna_rust._kwargs
    monkeypatch.setattr(luna_rust._kwargs, "_to_julia_type", lambda k, v: v)
