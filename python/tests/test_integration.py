"""Real (non-mocked) integration tests for the amalthea Python bindings.

These tests boot real Julia via juliacall, load the actual Amalthea
package, and call into the real Rust-accelerated backend -- unlike the
rest of python/tests/, which mock get_julia() entirely (see conftest.py).

They are skip-guarded, not hard failures, when the prerequisites aren't
available: `cargo build --release` in amalthea/ has not been run, or
Julia/Amalthea can't be loaded (e.g. no Julia on PATH). Run with:

    pytest python/tests/                          # everything, incl. these
    pytest -m "not integration" python/tests/      # fast loop, skips these
    pytest -m integration python/tests/            # only these
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _rust_lib_missing():
    if sys.platform == "darwin":
        name = "libamalthea.dylib"
    elif os.name == "nt":
        name = "amalthea.dll"
    else:
        name = "libamalthea.so"
    lib = os.path.join(_REPO_ROOT, "amalthea", "target", "release", name)
    return not os.path.isfile(lib)


@pytest.fixture(scope="module")
def real_amalthea():
    if _rust_lib_missing():
        pytest.skip(
            "Rust shared library not built; run `cargo build --release` "
            "in amalthea/ first."
        )
    try:
        import amalthea
        amalthea.get_julia()
    except Exception as e:
        pytest.skip(
            f"Could not boot Julia/Amalthea: {e}. See python/README.md's "
            "Install section."
        )
    import amalthea
    return amalthea


def test_prop_capillary_real(real_amalthea):
    out = real_amalthea.prop_capillary(
        75e-6, 0.001, "Ne", 1.5,
        trange=1e-12, lambda_lims=(150e-9, 4e-6),
        lambda0=800e-9, tfwhm=6e-15, energy=100e-6,
        saveN=5,
    )

    expected_keys = {
        "simulation_type", "meta", "Eω", "prop_capillary_args", "grid",
        "stats", "z",
    }
    assert expected_keys <= set(out.keys())

    Eω = out["Eω"]
    assert Eω.shape == (4097, 5)
    assert Eω.dtype == np.complex128

    z = out["z"]
    np.testing.assert_allclose(z, np.linspace(0, 0.001, 5), atol=1e-12)

    meta = out["meta"]
    assert "git_commit" in meta.keys()
    assert "Eω" in out


def test_prop_gnlse_real(real_amalthea):
    gamma = 0.1
    flength = 0.001
    betas = (0.0, 0.0, -1e-26)
    out = real_amalthea.prop_gnlse(
        gamma, flength, betas,
        lambda0=835e-9, tau_fwhm=100e-15, energy=1e-12,
        lambda_lims=(450e-9, 2000e-9), trange=4e-12,
        raman=True, shock=True, shotnoise=False, saveN=5,
    )
    assert "Eω" in out
    Eω = out["Eω"]
    assert Eω.ndim == 2
    assert Eω.shape[1] == 5


def test_prop_capillary_invalid_gas_raises(real_amalthea):
    from juliacall import JuliaError
    with pytest.raises(JuliaError):
        real_amalthea.prop_capillary(
            75e-6, 0.001, "NotARealGas", 1.5,
            lambda0=800e-9, tfwhm=6e-15, energy=100e-6,
        )


def test_prop_capillary_hdf5_output(real_amalthea, tmp_path):
    fpath = str(tmp_path / "out.h5")
    out = real_amalthea.prop_capillary(
        75e-6, 0.001, "Ne", 1.5,
        trange=1e-12, lambda_lims=(150e-9, 4e-6),
        lambda0=800e-9, tfwhm=6e-15, energy=100e-6,
        saveN=5, filepath=fpath,
    )
    assert os.path.isfile(fpath)
    assert "Eω" in out
    assert out["Eω"].shape[1] == 5
    # Exercises LunaOutput.keys()'s HDF5Output branch, which re-opens the
    # file read-only -- otherwise uncovered anywhere in the test suite.
    assert "Eω" in out.keys()
