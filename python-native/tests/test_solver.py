import os
from pathlib import Path

import numpy as np
import pytest

from amalthea_native import solve_precon


def exact(z, y0, linear):
    return y0 * np.exp(linear * z) / (1 - y0 * np.expm1(linear * z) / linear)


@pytest.mark.parametrize("fifth", [False, True])
def test_nonlinear_adaptive_rejection_and_dense_output(fifth):
    field = np.array([[1.0, .7j], [.4 + .1j, .3]])
    linear = np.full(field.shape, .13 + .2j)
    output = solve_precon(lambda z, y: y**2, linear, field, .4, dt=.4,
        rtol=1e-10, atol=1e-12, saveN=43, locextrap=fifth)
    reference = exact(output.z, field[..., None], linear[..., None])
    relative = np.linalg.norm(output.field - reference) / np.linalg.norm(reference)
    print(f"analytic full-solve fifth={fifth}: {relative:.3e}")
    assert relative < 1e-6
    linear_only = field[..., None] * np.exp(linear[..., None] * output.z)
    assert np.linalg.norm(reference - linear_only) / np.linalg.norm(reference) > .1
    assert output.metadata["rejected_steps"] > 0
    assert output.metadata["accepted_steps"] > 2
    assert output.field.shape == (2, 2, 43) and output.field.flags.owndata


@pytest.mark.parametrize("fifth,ratio", [(False, 32), (True, 64)])
def test_dense_local_order(fifth, ratio):
    errors = []
    for h in [.8, .4, .2]:
        output = solve_precon(lambda z, y: y, [0], [1], .37*h, dt=h,
            min_dt=h, max_dt=h, rtol=1, atol=1, locextrap=fifth, saveN=2)
        errors.append(abs(output.field[0, -1] - np.exp(.37*h)))
    print(f"dense errors fifth={fifth}: {errors}; ratios {np.array(errors[:-1])/errors[1:]}")
    assert errors[1] / errors[2] > .65 * ratio
    assert errors[1] / errors[2] < 1.5 * ratio


@pytest.mark.parametrize("fifth", [False, True])
def test_restart_from_saved_endpoint(fifth):
    options = dict(dt=.1, min_dt=.1, max_dt=.1, rtol=1, atol=1, locextrap=fifth)
    original = solve_precon(lambda z, y: y**2, [.2j], [1], .4, saveN=5, **options)
    restarted = solve_precon(lambda z, y: y**2, [.2j], original.field[:, 1],
        .4, z0=.1, saveN=4, **options)
    np.testing.assert_allclose(restarted.field, original.field[:, 1:], rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("fifth", [False, True])
def test_filter_cadence_and_field_synchronization(fifth):
    seen = []
    def filt(z, field):
        seen.append(z)
        return field * .9
    out = solve_precon(lambda z, y: y*0, [0], [1], .35, dt=.1,
        min_dt=.1, max_dt=.1, saveN=8, step_filter=filt, locextrap=fifth)
    # Compare at the actual floating-point positions: a sample just below a
    # filter discontinuity belongs to the previous interval, even by one ULP.
    filter_counts = np.searchsorted(np.asarray(seen), out.z, side="right")
    np.testing.assert_allclose(out.field[0], .9 ** filter_counts, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(seen, out.metadata["accepted_positions"])
    assert len(seen) == out.metadata["accepted_steps"] == 4


def test_callback_error_identity_and_lifetime():
    marker = RuntimeError("user's original error")
    calls = []
    def rhs(z, y):
        calls.append((z, y))
        if z > 0:
            raise marker
        return y
    with pytest.raises(RuntimeError) as caught:
        solve_precon(rhs, [0], [1], .1)
    assert caught.value is marker
    assert len(calls) == 2  # no callbacks after the first failing stage
    np.testing.assert_array_equal(calls[0][1], [1])
    for _ in range(20):
        out = solve_precon(lambda z, y: y*0, [1j], [1], .1, saveN=3)
        np.testing.assert_allclose(out.field[0], np.exp(1j*out.z), atol=1e-13)


@pytest.mark.parametrize("bad", [lambda z, y: 1, lambda z, y: [1, 2],
    lambda z, y: [np.nan], lambda z, y: [np.inf]])
def test_invalid_callback_output(bad):
    with pytest.raises(ValueError, match="callback"):
        solve_precon(bad, [0], [1], .1)


def test_limits_and_controls():
    with pytest.raises(RuntimeError, match="maximum step"):
        solve_precon(lambda z, y: y*0, [0], [1], 1, max_attempts=1)
    with pytest.raises(RuntimeError, match="repetition"):
        solve_precon(lambda z, y: y**2, [0], [1], .5, dt=.5,
            rtol=1e-14, repeat_limit=0)
    for options in [dict(rtol=0), dict(dt=-1), dict(min_dt=2), dict(saveN=1),
                    dict(z0=.2), dict(locextrap="false")]:
        with pytest.raises((ValueError, TypeError)):
            solve_precon(lambda z, y: y, [0], [1], .1, **options)


def test_independent_julia_solver_oracle():
    root = os.environ.get("AMALTHEA_SOLVER_ORACLE")
    if root is None:
        pytest.skip("Set AMALTHEA_SOLVER_ORACLE to exported Julia trajectories")
    for fifth in [False, True]:
        for fixed in [False, True]:
            name = f'{"fifth" if fifth else "fourth"}-{"fixed" if fixed else "adaptive"}'
            data = np.loadtxt(Path(root) / f"{name}.txt")
            options = dict(dt=.05, rtol=1e-10, atol=1e-12, max_dt=.05,
                           min_dt=.05 if fixed else 1e-15, saveN=17, locextrap=fifth)
            output = solve_precon(lambda z, y: (1+.3*z)*y**2, [.13+.2j], [1], .4, **options)
            reference = data[:, 1] + 1j*data[:, 2]
            error = np.linalg.norm(output.field[0]-reference)/np.linalg.norm(reference)
            print(f"Julia {name} full-solve relative error: {error:.3e}")
            assert error < (1e-13 if fixed else 1e-6)
            if fixed:
                # z=.025 is inside the first .05 step: identical initial
                # field, one attempt, and one continuous-extension evaluation.
                first_error = abs(output.field[0, 1]-reference[1])/abs(reference[1])
                print(f"Julia {name} single-interval relative error: {first_error:.3e}")
                assert first_error < 1e-13


def test_filter_exception_and_input_ownership():
    held = []
    def rhs(z, field):
        held.append((field, field.copy()))
        return field
    solve_precon(rhs, [0j], [1j], .1, saveN=4)
    for field, snapshot in held:
        np.testing.assert_array_equal(field, snapshot)


@pytest.mark.parametrize('fifth',[False,True])
@pytest.mark.parametrize('dt',[.001,.2])
def test_initial_step_outside_subsequent_bounds(fifth,dt):
    root=os.environ.get('AMALTHEA_SOLVER_ORACLE')
    if root is None:pytest.skip('set AMALTHEA_SOLVER_ORACLE')
    name=f'initial-bound-{"fifth" if fifth else "fourth"}-{dt}'
    data=np.loadtxt(Path(root)/f'{name}.txt')
    # Fix subsequent attempts so position equality tests the bound semantics,
    # independently of rounding in adaptive error estimates.
    actual=solve_precon(lambda z,y:y**2,[.13+.2j],[1],.25,dt=dt,min_dt=.01,max_dt=.01,
                         rtol=1e-9,atol=1e-12,locextrap=fifth,saveN=21)
    expected=data[:,1]+1j*data[:,2]
    error=np.linalg.norm(actual.field[0]-expected)/np.linalg.norm(expected)
    print('initial-bound',fifth,dt,error)
    assert error<1e-13
    np.testing.assert_allclose(actual.metadata['accepted_positions'],
                               np.loadtxt(Path(root)/f'{name}-accepted.txt'),rtol=1e-12,atol=1e-14)
    marker = LookupError("filter failure")
    def filt(z, field):
        raise marker
    with pytest.raises(LookupError) as caught:
        solve_precon(lambda z, y: y, [0], [1], .1, step_filter=filt)
    assert caught.value is marker


def test_julia_filtered_output_at_boundaries():
    root = os.environ.get("AMALTHEA_SOLVER_ORACLE")
    if root is None:
        pytest.skip("Set AMALTHEA_SOLVER_ORACLE for Julia filter-boundary tests")
    for fifth in [False, True]:
        for stop, count in [(.35, 8), (.4, 5)]:
            name = f'filter-{"fifth" if fifth else "fourth"}-{stop}'
            reference = np.loadtxt(Path(root)/f"{name}.txt")
            out = solve_precon(lambda z, y: y*0, [0], [1], stop,
                dt=.1, min_dt=.1, max_dt=.1, saveN=count, locextrap=fifth,
                step_filter=lambda z, y: .9*y)
            np.testing.assert_array_equal(out.z, reference[:, 0])
            np.testing.assert_allclose(out.field[0].real, reference[:, 1], rtol=1e-13, atol=1e-14)
    for fifth in [False, True]:
        name = f'window-{"fifth" if fifth else "fourth"}'
        data = np.loadtxt(Path(root)/f"{name}.txt")
        reference = (data[:, 1:3] + 1j*data[:, 3:5]).T
        options = dict(dt=.05, min_dt=.05, max_dt=.05, saveN=7, locextrap=fifth)
        out = solve_precon(lambda z, y: y**2, [.13+.2j, -.1+.4j], [1, .7j], .15,
            step_filter=lambda z, y: y*np.array([.95, .7]), **options)
        unfiltered = solve_precon(lambda z, y: y**2, [.13+.2j, -.1+.4j], [1, .7j], .15, **options)
        assert np.linalg.norm(reference-unfiltered.field)/np.linalg.norm(reference) > .1
        error = np.linalg.norm(out.field-reference)/np.linalg.norm(reference)
        print(f"Julia nonlinear window fifth={fifth}: {error:.3e}")
        assert error < 1e-13
