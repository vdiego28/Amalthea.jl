import gc
import os
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from amalthea_native import solve_precon

BASE = np.array([[.13+.2j, -.08+.31j], [-.05+.4j, .11-.17j]])
SLOPE = np.array([[.04+.13j, .03-.17j], [-.02+.07j, .06+.11j]])
INITIAL = np.array([[1, .3], [.7j, .4+.1j]])


def relative(a, b):
    return np.linalg.norm(a-b)/np.linalg.norm(b)


@pytest.fixture
def oracle():
    root = os.environ.get('AMALTHEA_VARIABLE_SOLVER_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_VARIABLE_SOLVER_ORACLE for variable solver acceptance')
    return Path(root)


@pytest.mark.parametrize('fifth', [False, True])
@pytest.mark.parametrize('kind', ['interval', 'fixed', 'adaptive', 'filtered',
                                 'filtered-adaptive', 'constant', 'linear'])
def test_variable_solver_oracle(oracle, fifth, kind):
    calls = []
    def linear(z):
        calls.append(z)
        return BASE+(0 if kind == 'constant' else np.sin(3*z))*SLOPE
    def rhs(z, y):
        return y*0 if kind == 'linear' else (1+.3*z)*y**2+.04*np.sum(y)
    def window(z, y):
        return y*np.array([[.9998, .9997], [.9996, .9999]]) if kind.startswith('filtered') else y
    adaptive = 'adaptive' in kind
    dt = .3 if adaptive else .05
    name = f'{"fifth" if fifth else "fourth"}-{kind}'
    if fifth and kind == 'filtered-adaptive':
        assert 'step repetition' in (oracle/name/'expected-failure.txt').read_text()
        with pytest.raises(RuntimeError, match='repetition'):
            solve_precon(rhs, linear, INITIAL, .425, z0=.125, dt=dt,
                          min_dt=1e-15, max_dt=dt, rtol=1e-10, atol=1e-12,
                          locextrap=fifth, saveN=17, step_filter=window)
        return
    result = solve_precon(rhs, linear, INITIAL, .175 if kind == 'interval' else .425,
                          z0=.125, dt=dt, min_dt=1e-15 if adaptive else dt,
                          max_dt=dt, rtol=1e-10, atol=1e-12, locextrap=fifth,
                          saveN=7 if kind == 'interval' else 17, step_filter=window)
    data = np.loadtxt(oracle/name/'field.txt')
    reference = (data[:, 1:5]+1j*data[:, 5:]).T.reshape(2, 2, -1, order='F')
    np.testing.assert_array_equal(result.z, data[:, 0])
    error = relative(result.field, reference)
    print(name, 'relative error', error, 'accepted', result.metadata['accepted_steps'],
          'rejected', result.metadata['rejected_steps'])
    assert error < (1e-6 if adaptive else 1e-13)
    assert result.metadata['linear_operator'] == 'python'
    assert result.field.flags.owndata
    assert all(a != b for a, b in zip(calls, calls[1:]))
    if adaptive:
        assert result.metadata['rejected_steps'] > 0
    else:
        np.testing.assert_array_equal(calls, np.loadtxt(oracle/name/'linear-positions.txt'))
        np.testing.assert_array_equal(result.metadata['accepted_positions'], np.loadtxt(oracle/name/'accepted.txt'))


def test_oracle_variable_nonlinear_and_filter_effects(oracle):
    for fifth in ('fourth', 'fifth'):
        def load(kind):
            data = np.loadtxt(oracle/f'{fifth}-{kind}'/'field.txt')
            return data[:, 1:5]+1j*data[:, 5:]
        for control in ('constant', 'linear', 'filtered'):
            effect = relative(load(control), load('fixed'))
            print(fifth, control, 'oracle effect', effect)
            assert effect > 1e-5


@pytest.mark.parametrize('fifth', [False, True])
def test_constant_callable_and_retained_arrays(fifth):
    held = []
    def linear(z):
        output = BASE.copy()
        held.append((output, output.copy(), threading.get_ident()))
        return output
    options = dict(dt=.05, min_dt=.05, max_dt=.05, locextrap=fifth, saveN=9)
    expected = solve_precon(lambda z, y: y*y, BASE, INITIAL, .2, **options)
    actual = solve_precon(lambda z, y: y*y, linear, INITIAL, .2, **options)
    np.testing.assert_array_equal(actual.field, expected.field)
    for output, saved, thread in held:
        np.testing.assert_array_equal(output, saved)
        assert thread == threading.get_ident()
        output[:] = 500
    np.testing.assert_array_equal(actual.field, expected.field)


@pytest.mark.parametrize('when', ['initial', 'stage', 'dense'])
def test_variable_operator_exception_identity_and_stop(when):
    marker = LookupError('original linear callback exception')
    calls = []
    def linear(z):
        calls.append(z)
        # .037 is an interior output position, distinct from DOPRI stages.
        if when == 'initial' or when == 'stage' and z > 0 or when == 'dense' and z == .037:
            raise marker
        return np.array([.2j+z])
    for _ in range(4):
        calls.clear()
        with pytest.raises(LookupError) as caught:
            solve_precon(lambda z, y: y, linear, [1], .074, dt=.1,
                          min_dt=.1, max_dt=.1, saveN=3)
        assert caught.value is marker
        assert calls[-1] == (0 if when == 'initial' else .1*.2 if when == 'stage' else .037)
    assert np.all(np.isfinite(solve_precon(lambda z, y: y, lambda z: [.2j+z], [1], .1).field))


@pytest.mark.parametrize('bad', [lambda z: 1, lambda z: [1, 2], lambda z: [np.nan],
                                lambda z: [np.inf], lambda z: [[1]]])
def test_invalid_linear_callback_result(bad):
    with pytest.raises(ValueError, match='linear callback'):
        solve_precon(lambda z, y: y, bad, [1], .1)


def test_linear_callback_teardown():
    class Model:
        def __call__(self, z):
            return [.2j+z]
    for _ in range(20):
        model = Model()
        ref = weakref.ref(model)
        solve_precon(lambda z, y: y*0, model, [1], .2, dt=.1)
        del model
        gc.collect()
        assert ref() is None


def test_variable_linear_endpoint_formula_and_refinement():
    # Julia's frozen endpoint exponential has first-order variable-L error,
    # independently of the fifth-order nonlinear DOPRI tableau.
    errors = []
    for n in (4, 8, 16):
        step = 1/n
        result = solve_precon(lambda z, y: y*0, lambda z: [1j*z], [1], 1.,
                              dt=step, min_dt=step, max_dt=step, saveN=n+1)
        sums = step**2*np.arange(n+1)*(np.arange(n+1)+1)/2
        assert relative(result.field[0], np.exp(1j*sums)) < 1e-13
        errors.append(abs(result.field[0, -1]-np.exp(.5j)))
    print('variable linear refinement', errors)
    assert 1.9 < errors[0]/errors[1] < 2.1 and 1.9 < errors[1]/errors[2] < 2.1
