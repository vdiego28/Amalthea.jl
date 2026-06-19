import os
os.environ["PYTHON_JULIAPKG_JULIA"] = "julia"
import pytest
import numpy as np
import luna_rust

def test_prop_capillary_ascii_kwargs():
    o = luna_rust.prop_capillary(
        100e-6, 0.1, "He", 1.0,
        lambda0=800e-9, tau_fwhm=10e-15, energy=1e-12,
        lambda_lims=(200e-9, 4e-6), trange=400e-15, shotnoise=False
    )
    assert "Eω" in o
    assert o["Eω"].ndim == 2
    assert o["Eω"].shape[1] > 0

if __name__ == "__main__":
    test_prop_capillary_ascii_kwargs()
    print("Success!")
