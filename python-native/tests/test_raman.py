"""Silica response normalization, causality, and model-selection regressions."""
import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest
from scipy.integrate import quad

from amalthea_native import prop_gnlse
from amalthea_native.gnlse import _Gnlse
from amalthea_native.raman import (
    _sio2_normalization, _sio2_raw, _sio2_response,
    _OMEGA, _AMPLITUDE, _GAUSSIAN, _LORENTZIAN,
)

SETTINGS = dict(lambda0=800e-9, lambda_lims=(550e-9, 1700e-9), trange=400e-15,
                tau_fwhm=20e-15, power=2000., saveN=7, ramanmodel='SiO2')
BETAS = [0., 0., -20e-27, 50e-42]


def relative(actual, reference):
    return np.linalg.norm(actual-reference) / np.linalg.norm(reference)


def test_continuous_normalization_refinement():
    # Time in fs keeps the integral away from quad's tiny-absolute-integral trap.
    # Explicit near-zero intervals ensure the ns endpoint cannot hide the pulse.
    boundaries = np.array([0., 50, 100, 250, 500, 1000, 2000, 4000, 8000, 1e6])
    values = []
    for epsrel, points in [(1e-10, boundaries), (2e-12, boundaries), (2e-13, np.sort(np.r_[boundaries, (boundaries[:-1]+boundaries[1:])/2]))]:
        integral, error = quad(lambda fs: float(_sio2_raw(fs*1e-15)), 0, 1e6,
                               points=points[1:-1], epsabs=0, epsrel=epsrel, limit=1000)
        assert error/abs(integral) <= epsrel
        values.append(integral*1e-15)
        print(f'SiO2 quad reltol={epsrel}: integral={values[-1]:.16e}, estimate={error/abs(integral):.3e}')
    norm = _sio2_normalization()
    assert abs(values[-1]/values[-2]-1) < 1e-13
    assert abs(norm/values[-1]-1) < 1e-13
    # Independent bound on the remainder between Julia's 1 ns and infinity.
    assert np.min((_GAUSSIAN*1e-9/2)**2) > 1e6
    assert norm > 0 and np.isfinite(norm)
    print(f'SiO2 closed form={norm:.16e}; refined quadrature difference={abs(norm/values[-1]-1):.3e}')


def test_julia_normalization_and_untapered_response():
    root = os.environ.get('AMALTHEA_GNLSE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_GNLSE_ORACLE for independent Julia acceptance')
    for name in ('sio2', 'sio2-fine', 'sio2-long'):
        directory = Path(root)/name
        oracle = tomllib.loads((directory/'normalization.toml').read_text())
        for name_, coefficients in [('omega',_OMEGA), ('amplitude',_AMPLITUDE),
                                     ('gaussian',_GAUSSIAN), ('lorentzian',_LORENTZIAN)]:
            np.testing.assert_allclose(coefficients, oracle[name_], rtol=1e-15, atol=0)
        assert oracle['relative_error_estimate'] < 1e-13
        norm = _sio2_normalization()
        assert abs(norm/oracle['refined']-1) < 1e-13
        params = tomllib.loads((directory/'parameters.toml').read_text())
        positional = [params.pop(k) for k in ('gamma','flength','betas')]
        model = _Gnlse(*positional, **params)
        original = np.loadtxt(directory/'raman-default.txt')
        refined = np.loadtxt(directory/'raman.txt')
        np.testing.assert_allclose(refined, original*(oracle['default']/oracle['refined']),
                                   rtol=1e-13, atol=np.max(np.abs(refined))*1e-15)
        assert relative(model.h, refined) < 1e-13
        # Explain, rather than hide, the original oracle quadrature discrepancy.
        measured = relative(model.h, original)
        expected = abs(oracle['default']/norm-1)
        assert abs(measured-expected) < 1e-13
        print(f'{name}: default Julia response discrepancy={measured:.3e}; normalization predicts={expected:.3e}')
        assert model.h[-1] != 0  # SDO's taper would incorrectly force this to zero


def test_silica_oracle_physics_sensitivity():
    root = os.environ.get('AMALTHEA_GNLSE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_GNLSE_ORACLE for independent Julia acceptance')
    root = Path(root)
    silica = np.loadtxt(root/'sio2/adaptive.txt')
    for name in ('no-raman', 'base'):
        effect = relative(silica, np.loadtxt(root/name/'adaptive.txt'))
        print(f'Julia SiO2 vs {name}: {effect:.3e}')
        assert effect > 1e-4


@pytest.mark.parametrize('delta_t', [1., .5e-15])
def test_causal_silica_convolution_and_ownership(delta_t):
    model = _Gnlse(.01, .03, BETAS, **SETTINGS, delta_t=delta_t)
    drive = np.zeros(model.no)
    start = model.no//3
    drive[start:start+11] = np.linspace(.2, 1., 11)
    drive[-1] = 2.
    dt = model.grid.to[1]-model.grid.to[0]
    expected = np.convolve(model.h, drive)[:model.no] * dt
    actual = np.fft.irfft(model.hfft * np.fft.rfft(np.pad(drive, (0, model.no))), n=2*model.no)[:model.no]
    assert relative(actual, expected) < 1e-13
    assert np.max(np.abs(actual[:start])) < np.max(np.abs(expected))*1e-13
    np.testing.assert_array_equal(_sio2_response(np.array([-1., -1e-15, 0.]), 1.), 0.)
    reference = model.h.copy()
    for _ in range(5):
        other = _Gnlse(.01, .03, BETAS, **SETTINGS)
        other.h[:] = 0
    np.testing.assert_array_equal(model.h, reference)


def test_silica_unused_sdo_widths_scaling_and_metadata(tmp_path):
    normal = _Gnlse(.01, .03, BETAS, **SETTINGS)
    ignored = _Gnlse(.01, .03, BETAS, **SETTINGS, tau1=None, tau2=None)
    np.testing.assert_array_equal(ignored.h, normal.h)
    zero = _Gnlse(.01, .03, BETAS, **SETTINGS, fr=0)
    off = _Gnlse(.01, .03, BETAS, **(SETTINGS | dict(raman=False, fr=0)))
    np.testing.assert_array_equal(zero.h, 0.)
    np.testing.assert_array_equal(zero.rhs(0, zero.initial), off.rhs(0, off.initial))
    linear = _Gnlse(0., .03, BETAS, **SETTINGS)
    np.testing.assert_array_equal(linear.rhs(0, linear.initial), 0.)
    result = prop_gnlse(.01, .03, BETAS, **SETTINGS, backend='python')
    result.save_npz(tmp_path/'silica.npz')
    with np.load(tmp_path/'silica.npz', allow_pickle=False) as saved:
        assert json.loads(str(saved['parameters']))['ramanmodel'] == 'SiO2'
        assert json.loads(str(saved['metadata']))['backend'] == 'python'
