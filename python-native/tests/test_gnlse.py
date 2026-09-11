import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from amalthea_native import EnvGrid, prop_gnlse
from amalthea_native.gnlse import EPS0, _Gnlse, energy_t

BASE = dict(lambda0=800e-9, lambda_lims=(550e-9, 1700e-9), trange=400e-15,
            tau_fwhm=20e-15, power=2000., saveN=7)
BETAS = [0., 0., -20e-27, 50e-42]


def relative(actual, expected):
    return np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-300)


@pytest.mark.parametrize('name', ['base', 'fine', 'sech', 'energy', 'no-kerr',
                                  'no-raman', 'no-shock', 'no-loss', 'frame',
                                  'sio2', 'sio2-fine', 'sio2-long'])
@pytest.mark.parametrize('backend', ['python', 'native'])
def test_julia_gnlse(name, backend):
    root = os.environ.get('AMALTHEA_GNLSE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_GNLSE_ORACLE for independent Julia acceptance')
    directory = Path(root) / name
    params = tomllib.loads((directory / 'parameters.toml').read_text())
    gamma, length, betas = [params.pop(k) for k in ('gamma', 'flength', 'betas')]
    params['backend'] = backend
    model = _Gnlse(gamma, length, betas, **params)
    setup = np.loadtxt(directory / 'setup.txt')
    for label, actual, reference in (
        ('input', model.initial, setup[:, 0] + 1j * setup[:, 1]),
        ('linop', model.linop, setup[:, 2] + 1j * setup[:, 3]),
        ('rhs', model.rhs(0, model.initial) if backend == 'python' else np.asarray(
            __import__('amalthea_native')._native.envelope_rhs(model.linop.tolist(), model.initial.tolist(), model.native_config())),
         setup[:, 4] + 1j * setup[:, 5]),
    ):
        error = relative(actual, reference)
        print(f'{name} {label}: {error:.3e}')
        assert error < 1e-13
    if name == 'base' or name.startswith('sio2'):
        assert relative(model.h, np.loadtxt(directory / 'raman.txt')) < 1e-13
    for fixed in (False, True):
        suffix = 'fixed' if fixed else 'adaptive'
        result = prop_gnlse(gamma, length, betas, **params, init_dz=.001,
                            min_dz=.001 if fixed else 1e-15, max_dz=.001,
                            rtol=1e-9, atol=1e-12)
        data = np.loadtxt(directory / f'{suffix}.txt')
        reference = data[:, :7] + 1j * data[:, 7:]
        np.testing.assert_array_equal(result.z, np.loadtxt(directory / f'{suffix}-z.txt'))
        error = relative(result.field, reference)
        print(f'{name} {suffix}: {error:.3e}')
        assert error < (1e-13 if fixed else 1e-6)
    if name == 'base' or name.startswith('sio2'):
        result = prop_gnlse(gamma, length, betas, **params)
        data = np.loadtxt(directory / 'entrypoint.txt')
        error = relative(result.field, data[:, :7] + 1j * data[:, 7:])
        print(f'public Julia entrypoint vs Python entrypoint: {error:.3e}')
        assert error < 1e-6
        np.testing.assert_array_equal(result.z, np.loadtxt(directory / 'entrypoint-z.txt'))
    short = params | dict(saveN=3)
    result = prop_gnlse(gamma, .001, betas, **short, init_dz=.001, min_dz=.001, max_dz=.001)
    data = np.loadtxt(directory / 'interval.txt')
    error = relative(result.field, data[:, :3] + 1j * data[:, 3:])
    print(f'{name} single interval: {error:.3e}')
    assert error < 1e-13


def test_oracle_physics_sensitivity():
    root = os.environ.get('AMALTHEA_GNLSE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_GNLSE_ORACLE for independent Julia acceptance')
    base = np.loadtxt(Path(root) / 'base/adaptive.txt')
    for name in ('no-kerr', 'no-raman', 'no-shock', 'no-loss'):
        effect = relative(np.loadtxt(Path(root) / name / 'adaptive.txt'), base)
        print(f'Julia {name} effect: {effect:.3e}')
        assert effect > 1e-4  # comfortably larger than full-solve tolerance


def test_energy_phase_aliases_arrays_and_owned_output(tmp_path):
    args = BASE | dict(power=None, energy=50e-12, phi=[.2, 2e-15, 1e-29])
    model = _Gnlse(.01, .01, BETAS, **args)
    assert energy_t(model.grid, np.fft.ifft(model.initial)) == pytest.approx(50e-12, rel=1e-14)
    aliases = { 'λ0':args.pop('lambda0'), 'λlims':args.pop('lambda_lims'),
                'τfwhm':args.pop('tau_fwhm'), 'ϕ':args.pop('phi') }
    aliased = _Gnlse(.01, .01, BETAS, **(args | aliases))
    np.testing.assert_array_equal(aliased.initial, model.initial)
    for domain, pulse in [('time', np.fft.ifft(model.initial)), ('frequency', model.initial.copy())]:
        before = pulse.copy()
        kw = BASE | dict(tau_fwhm=None, power=None, pulse=pulse, pulse_domain=domain, backend='python')
        result = prop_gnlse(.01, .01, BETAS, **kw)
        np.testing.assert_array_equal(pulse, before)
        assert relative(result.field[:, 0], model.initial) < 1e-14
        assert result.field.flags.owndata
        assert result.metadata['backend'] == 'python'
        np.testing.assert_allclose(result.temporal_field(), np.fft.ifft(result.field, axis=0))
        path = tmp_path / f'{domain}.npz'
        result.save_npz(path)
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['Eomega'], result.field)
            assert json.loads(str(saved['metadata']))['stepper'] == 'rust-precon'
            assert json.loads(str(saved['parameters']))['pulse'] == 'supplied grid-matched array'


def test_linear_loss_analytic():
    # A single frequency at grid center is unaffected by the spectral window.
    # Make the time window identically one for this independent solver check.
    from amalthea_native import solve_precon
    model = _Gnlse(0, .03, [0, 0], **(BASE | dict(loss=1.5)))
    result = solve_precon(model.rhs, model.linop, model.initial, .03, saveN=7)
    reference = model.initial[:, None] * np.exp(model.linop[:, None] * result.z)
    assert relative(result.field, reference) < 1e-13
    assert relative(result.field[:, -1], result.field[:, 0]) > 1e-4


@pytest.mark.parametrize('bad,exception', [
    ({'shotnoise':True}, NotImplementedError),
    ({'ramanmodel':'unknown'}, ValueError), ({'backend':'unknown'}, ValueError),
    ({'λ0':800e-9}, TypeError), ({'made_up':1}, TypeError),
    ({'fr':float('nan')}, ValueError), ({'loss':float('inf')}, ValueError),
    ({'energy':1}, ValueError), ({'tau_fwhm':-1}, ValueError),
    ({'pulse':[1,2], 'tau_fwhm':None}, ValueError),
    ({'shotnoise':'false'}, TypeError), ({'shock':1}, TypeError),
])
def test_invalid_configuration(bad, exception):
    with pytest.raises(exception):
        prop_gnlse(.01, .01, BETAS, **(BASE | bad))


def test_invalid_array_and_phase():
    grid = EnvGrid(.01, BASE['lambda0'], BASE['lambda_lims'], BASE['trange'])
    for pulse in [np.full(grid.t.shape, np.nan), np.zeros((grid.t.size, 1)), np.zeros(grid.t.size)]:
        with pytest.raises(ValueError):
            prop_gnlse(.01, .01, BETAS, **(BASE | dict(pulse=pulse, tau_fwhm=None)))
    with pytest.raises(ValueError):
        _Gnlse(.01, .01, BETAS, **(BASE | dict(phi=[np.inf])))
