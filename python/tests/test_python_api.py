import pytest
import numpy as np

import amalthea

def test_prop_capillary_ascii_kwargs():

    o = amalthea.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12,
        lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2
    assert o["Eω"].shape[1] > 0

def test_prop_capillary_unicode_kwargs():

    o = amalthea.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        λ0=800e-9, τfwhm=10e-15, energy=1e-12,
        λlims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2

def test_duplicate_kwargs():
    with pytest.raises(ValueError):
        amalthea.prop_capillary(
            100e-6, 0.1, "He", 1.0,
            lambda0=800e-9, λ0=800e-9, tau_fwhm=10e-15, energy=1e-12,
            lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
        )

def test_gnlse_ascii():

    gamma = 0.1
    flength = 0.001
    betas = (0.0, 0.0, -1e-26)
    o = amalthea.prop_gnlse(
        gamma, flength, betas,
        lambda0=835e-9, tau_fwhm=100e-15, energy=1e-12,
        lambda_lims=(450e-9, 2000e-9), trange=4e-12,
        raman=True, shock=True, shotnoise=False, saveN=11
    )
    assert "Eω" in o
    assert o["Eω"].shape[1] > 0

def test_gnlse_unicode():
    gamma = 0.1
    flength = 0.001
    betas = (0.0, 0.0, -1e-26)
    o = amalthea.prop_gnlse(
        gamma, flength, betas,
        λ0=835e-9, τfwhm=100e-15, energy=1e-12,
        λlims=(450e-9, 2000e-9), trange=4e-12,
        raman=True, shock=True, shotnoise=False, saveN=11
    )
    assert "Eω" in o
    assert o["Eω"].shape[1] > 0

def test_gnlse_duplicate_kwargs():
    gamma = 0.1
    flength = 0.001
    betas = (0.0, 0.0, -1e-26)
    with pytest.raises(ValueError):
        amalthea.prop_gnlse(
            gamma, flength, betas,
            lambda0=835e-9, λ0=835e-9, tau_fwhm=100e-15, energy=1e-12,
            lambda_lims=(450e-9, 2000e-9), trange=4e-12,
            raman=True, shock=True, shotnoise=False, saveN=11
        )
