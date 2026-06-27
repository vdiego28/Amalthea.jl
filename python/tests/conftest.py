import sys
from unittest.mock import MagicMock

class MockLunaOutput:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data[key]
    def __contains__(self, key):
        return key in self.data

class MockLuna:
    def prop_capillary(self, *args, **kwargs):
        import numpy as np
        return {"Eω": np.zeros((10, 10))}

    def prop_gnlse(self, *args, **kwargs):
        import numpy as np
        return {"Eω": np.zeros((10, 10))}

mock_jl = MagicMock()
mock_jl.Symbol = lambda x: x
mock_luna = MockLuna()

mock_get_julia = MagicMock(return_value=(mock_jl, mock_luna))

import pytest
@pytest.fixture(autouse=True)
def mock_julia_backend(monkeypatch):
    import luna_rust
    monkeypatch.setattr(luna_rust._julia, "get_julia", mock_get_julia)
    monkeypatch.setattr(luna_rust, "get_julia", mock_get_julia)
    monkeypatch.setattr(luna_rust._kwargs, "get_julia", mock_get_julia)
    monkeypatch.setattr(luna_rust, "LunaOutput", MockLunaOutput)

    # In some module setups, patching `sys.modules` is the only way to intercept nested intra-package imports
    monkeypatch.setitem(sys.modules, "luna_rust._julia", MagicMock(get_julia=mock_get_julia))
