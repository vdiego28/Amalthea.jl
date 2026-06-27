import pytest
import sys
import numpy as np

import unittest.mock

# We must mock BEFORE importing luna_rust or any juliacall module
mock_julia = unittest.mock.Mock()
mock_luna = unittest.mock.Mock()

def mock_get_julia():
    return mock_julia, mock_luna

class MockLunaOutput:
    def __init__(self, result):
        self.data = {"Eω": np.ones((10, 10))}

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

sys.modules['juliacall'] = unittest.mock.Mock()
sys.modules['juliapkg'] = unittest.mock.Mock()

with unittest.mock.patch('luna_rust._julia.get_julia', side_effect=mock_get_julia):
    with unittest.mock.patch('luna_rust.LunaOutput', MockLunaOutput):
        import luna_rust
        # Overwrite the actual module attributes manually because monkeypatch does not persist through tests
        # when re-importing, and the monkeypatch below will throw an error since _julia might not be in the module
        # tree properly if patched dynamically via context manager.
        import luna_rust._julia
        luna_rust._julia.get_julia = mock_get_julia
        luna_rust.LunaOutput = MockLunaOutput

@pytest.fixture(autouse=True)
def apply_mocks(monkeypatch):
    monkeypatch.setattr("luna_rust._julia.get_julia", mock_get_julia)
    monkeypatch.setattr("luna_rust.LunaOutput", MockLunaOutput)
    monkeypatch.setattr("luna_rust._kwargs._to_julia_type", lambda k, v: v) # prevent julia symbol conversion

def test_prop_capillary_ascii_kwargs():
    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12,
        lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2
    assert o["Eω"].shape[1] > 0

def test_prop_capillary_unicode_kwargs():
    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        λ0=800e-9, τfwhm=10e-15, energy=1e-12,
        λlims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2

def test_duplicate_kwargs():
    with pytest.raises(ValueError):
        luna_rust.prop_capillary(
            100e-6, 0.1, "He", 1.0,
            lambda0=800e-9, λ0=800e-9, tau_fwhm=10e-15, energy=1e-12,
            lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
        )

def test_gnlse_ascii():
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
