import pytest
import numpy as np

# We must mock `get_julia` before `luna_rust` initializes.
# However, because `__init__.py` imports `get_julia` locally, mocking it via patch can be tricky.
# We'll just patch the internal module.
from unittest import mock
import sys

# Creating a mock module
mock_julia = mock.MagicMock()
mock_jl = mock.MagicMock()
mock_luna = mock.MagicMock()
mock_julia.get_julia.return_value = (mock_jl, mock_luna)

sys.modules['luna_rust._julia'] = mock_julia

import luna_rust

@mock.patch("luna_rust.LunaOutput")
def test_prop_capillary_ascii_kwargs(mock_luna_output):
    mock_luna_output.return_value = {"Eω": np.array([[1.0, 2.0], [3.0, 4.0]])}

    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12,
        lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2
    assert o["Eω"].shape[1] > 0

@mock.patch("luna_rust.LunaOutput")
def test_prop_capillary_unicode_kwargs(mock_luna_output):
    mock_luna_output.return_value = {"Eω": np.array([[1.0, 2.0], [3.0, 4.0]])}

    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        λ0=800e-9, τfwhm=10e-15, energy=1e-12,
        λlims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2

@mock.patch("luna_rust.LunaOutput")
def test_duplicate_kwargs(mock_luna_output):
    with pytest.raises(ValueError):
        luna_rust.prop_capillary(
            100e-6, 0.1, "He", 1.0,
            lambda0=800e-9, λ0=800e-9, tau_fwhm=10e-15, energy=1e-12,
            lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
        )

@mock.patch("luna_rust.LunaOutput")
def test_gnlse_ascii(mock_luna_output):
    mock_luna_output.return_value = {"Eω": np.array([[1.0, 2.0], [3.0, 4.0]])}

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
