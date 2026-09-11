"""Resident lifecycle and complete-path checks, independent of its implementation."""
import copy

import numpy as np
import pytest

from amalthea_native import _native, prop_gnlse, solve_precon
from amalthea_native.gnlse import _Gnlse
from test_gnlse import BASE, BETAS, relative


def setup(windows=False):
    n = 8
    time = .8 + .2j + .1*np.cos(2*np.pi*np.arange(n)/n)
    initial = np.fft.fft(time)
    linop = -.08 + 1j*np.linspace(.3, .8, n)
    twin = np.linspace(.94, 1., n) if windows else np.ones(n)
    owin = np.linspace(1., .97, n) if windows else np.ones(n)
    config = ([1j]*n, np.ones(n).tolist(), twin.tolist(), owin.tolist(), 4/3, 1., None, .1)
    def rhs(z, field):
        e = np.fft.ifft(field)
        return 1j*np.fft.fft(e*np.abs(e)**2)
    def window(z, field):
        return np.fft.fft(twin*np.fft.ifft(owin*field))
    return initial, linop, config, rhs, window


def native_run(initial, linop, config, positions, *, fifth=True, dt=.01,
               min_dt=1e-14, max_dt=.1, rtol=1e-10, max_attempts=10000, repeat_limit=10):
    samples, accepted, rejected, endpoints = _native.solve_envelope(
        linop.tolist(), initial.tolist(), config, list(positions), dt, rtol, 1e-12,
        .9, min_dt, max_dt, fifth, max_attempts, repeat_limit)
    return np.array(samples).T, accepted, rejected, np.array(endpoints)


@pytest.mark.parametrize('fifth',[False,True])
@pytest.mark.parametrize('dt',[.001,.2])
def test_resident_initial_step_outside_subsequent_bounds(fifth,dt):
    initial,linop,config,rhs,window=setup()
    options=dict(dt=dt,min_dt=.01,max_dt=.01,rtol=1e-9)
    expected=solve_precon(rhs,linop,initial,.25,**options,atol=1e-12,saveN=21,locextrap=fifth)
    actual,accepted,rejected,positions=native_run(initial,linop,config,expected.z,**options,fifth=fifth)
    assert relative(actual,expected.field)<1e-13
    np.testing.assert_allclose(positions,expected.metadata['accepted_positions'],rtol=1e-12,atol=1e-14)
    assert accepted==expected.metadata['accepted_steps'] and rejected==expected.metadata['rejected_steps']


@pytest.mark.parametrize('fifth', [False, True])
@pytest.mark.parametrize('windows', [False, True])
def test_native_lifecycle_vs_callback(fifth, windows):
    initial, linop, config, rhs, window = setup(windows)
    expected = solve_precon(rhs, linop, initial, .3, dt=.01, min_dt=.01, max_dt=.01,
                            saveN=37, locextrap=fifth, step_filter=window)
    actual, accepted, rejected, endpoints = native_run(initial, linop, config, expected.z,
        fifth=fifth, dt=.01, min_dt=.01, max_dt=.01)
    error = relative(actual, expected.field)
    print(f'resident fixed fifth={fifth} windows={windows}: {error:.3e}')
    assert error < 1e-13
    np.testing.assert_array_equal(endpoints, expected.metadata['accepted_positions'])
    assert accepted == expected.metadata['accepted_steps'] and rejected == 0
    if windows:
        plain = native_run(initial, linop, setup(False)[2], expected.z,
                          fifth=fifth, dt=.01, min_dt=.01, max_dt=.01)[0]
        assert relative(actual, plain) > 1e-2


@pytest.mark.parametrize('fifth', [False, True])
def test_native_rejection_retry_restart_and_stopping(fifth):
    initial, linop, config, rhs, window = setup()
    expected = solve_precon(rhs, linop, initial, .3, dt=.3, max_dt=.3, saveN=19,
                            rtol=1e-10, atol=1e-12, locextrap=fifth, step_filter=window)
    actual, accepted, rejected, endpoints = native_run(initial, linop, config, expected.z,
        fifth=fifth, dt=.3, max_dt=.3)
    error = relative(actual, expected.field)
    print(f'resident adaptive fifth={fifth}: {error:.3e}, rejected={rejected}')
    assert error < 1e-6 and rejected > 0 and accepted > 2
    # At least one additional accepted endpoint beyond the last requested output.
    assert endpoints[-1] > expected.z[-1]
    restarted = native_run(actual[:, 9], linop, config, expected.z[9:], fifth=fifth)[0]
    assert relative(restarted, actual[:, 9:]) < 1e-6
    with pytest.raises(RuntimeError, match='maximum step attempts'):
        native_run(initial, linop, config, [0, 1], max_attempts=1)
    with pytest.raises(RuntimeError, match='repetition limit'):
        native_run(initial, linop, config, [0, 1], dt=1., max_dt=1., repeat_limit=0)


@pytest.mark.parametrize('fifth,min_ratio', [(False, 20), (True, 35)])
def test_native_dense_convergence_against_analytic(fifth, min_ratio):
    n = 8; a = -.08; b = .3; amplitude = .8+.2j
    initial = np.zeros(n, complex); initial[0] = n*amplitude
    linop = np.full(n, a+1j*b)
    config = setup()[2]
    errors=[]
    for h in [.4, .2, .1]:
        positions = np.array([0., .37*h, .999999*h, h])
        actual = native_run(initial, linop, config, positions,
                            fifth=fifth, dt=h, min_dt=h, max_dt=h)[0]
        reference = initial[:,None]*np.exp(a*positions+1j*(b*positions+
                          abs(amplitude)**2*np.expm1(2*a*positions)/(2*a)))
        errors.append(relative(actual, reference))
    ratios = np.array(errors[:-1])/errors[1:]
    print(f'resident analytic dense fifth={fifth}: errors={errors}, ratios={ratios}')
    assert np.all(ratios > min_ratio)


def test_native_backend_selection_and_no_python_stages(monkeypatch):
    expected = prop_gnlse(.01, .03, BETAS, **BASE, backend='python')
    def forbidden(*args):
        raise AssertionError('Python stage/filter entered during native solve')
    monkeypatch.setattr(_Gnlse, 'rhs', forbidden)
    monkeypatch.setattr(_Gnlse, 'window', forbidden)
    for backend in ['native', 'auto']:
        for _ in range(3):
            actual = prop_gnlse(.01, .03, BETAS, **BASE, backend=backend)
            assert actual.metadata['backend'] == 'native'
            assert actual.metadata['stepper'] == 'rust-resident'
            assert actual.metadata['rhs'] == 'rust'
            assert actual.metadata['fft'] == 'rustfft/realfft'
            assert relative(actual.field, expected.field) < 1e-6


def test_native_private_config_validation_and_ownership():
    initial, linop, config, _, _ = setup()
    preserved = copy.deepcopy(config)
    for index, value in [(0,[1j]), (1,[1.]), (2,[1.]), (3,[1.]),
                         (4,float('nan')), (5,0.), (6,[1.]*7)]:
        bad = list(config); bad[index] = value
        with pytest.raises(ValueError):
            native_run(initial, linop, tuple(bad), [0, .1])
    for value in [np.full(8, np.nan), np.ones(7)]:
        with pytest.raises(ValueError):
            native_run(value, linop, config, [0, .1])
    for controls in [dict(dt=0),dict(min_dt=.2,max_dt=.1),dict(rtol=float('nan'))]:
        with pytest.raises(ValueError):
            native_run(initial, linop, config, [0, .1], **controls)
    actual = native_run(initial, linop, config, [0, .1])[0]
    assert config == preserved and np.all(np.isfinite(actual))
