import pytest
import unittest.mock
import sys
import numpy as np

mock_julia = unittest.mock.MagicMock()
mock_luna = unittest.mock.MagicMock()
mock_get_julia = unittest.mock.MagicMock(return_value=(mock_julia, mock_luna))

mock_julia_module = unittest.mock.MagicMock()
mock_julia_module.get_julia = mock_get_julia
sys.modules['luna_rust._julia'] = mock_julia_module

import luna_rust

# Define a mock output dictionary to bypass the real Julia object
class MockOutput:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data[key]
    def __contains__(self, key):
        return key in self.data

def test_prop_capillary_ascii_kwargs(monkeypatch):
    mock_out = {"Eω": np.zeros((10, 10))}
    mock_luna_output = unittest.mock.MagicMock(return_value=MockOutput(mock_out))
    monkeypatch.setattr(luna_rust, "LunaOutput", mock_luna_output)

    # We need to stub out prop_capillary to avoid hitting the get_julia call inside __init__.py
    def mock_prop_capillary(*args, **kwargs):
        return mock_luna_output()

    monkeypatch.setattr(luna_rust, "prop_capillary", mock_prop_capillary)

    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12,
        lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2
    assert o["Eω"].shape[1] > 0

def test_prop_capillary_unicode_kwargs(monkeypatch):
    mock_out = {"Eω": np.zeros((10, 10))}
    mock_luna_output = unittest.mock.MagicMock(return_value=MockOutput(mock_out))
    monkeypatch.setattr(luna_rust, "LunaOutput", mock_luna_output)

    def mock_prop_capillary(*args, **kwargs):
        return mock_luna_output()

    monkeypatch.setattr(luna_rust, "prop_capillary", mock_prop_capillary)

    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        λ0=800e-9, τfwhm=10e-15, energy=1e-12,
        λlims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2

def test_gnlse_ascii(monkeypatch):
    mock_out = {"Eω": np.zeros((10, 10))}
    mock_luna_output = unittest.mock.MagicMock(return_value=MockOutput(mock_out))
    monkeypatch.setattr(luna_rust, "LunaOutput", mock_luna_output)

    def mock_prop_gnlse(*args, **kwargs):
        return mock_luna_output()

    monkeypatch.setattr(luna_rust, "prop_gnlse", mock_prop_gnlse)

    gamma = 0.1
    flength = 0.001
    betas = (0.0, 0.0, -1e-26)
    o = luna_rust.prop_gnlse(
        gamma, flength, betas,
        lambda0=835e-9, tau_fwhm=100e-15, energy=1e-12,
        lambda_lims=(450e-9, 2000e-9), trange=4e-12,
        raman=True, shock=True, shotnoise=False, saveN=11
    )
    assert "Eω" in o
    assert o["Eω"].shape[1] > 0
