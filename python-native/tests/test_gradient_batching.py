"""Frequency batching must preserve math and the public callback contract."""
import numpy as np
import pytest

from amalthea_native import MarcatiliMode
from amalthea_native.capillary import _Capillary
from amalthea_native.differentiation import derivative, _evaluate, _evaluate_batched
from amalthea_native.envelope import solve_envelope_model

LENGTH = 1e-4
OPTIONS = dict(lambda0=800e-9, lambda_lims=(200e-9, 1700e-9), trange=100e-15,
               tau_fwhm=20e-15, energy=1e-5, envelope=True, plasma=False,
               raman=False, init_dz=LENGTH/4, max_dz=LENGTH/4,
               rtol=1e-9, atol=1e-12, saveN=7)


def relative(left, right):
    return np.linalg.norm(np.asarray(left)-right)/max(np.linalg.norm(right), 1e-300)


def test_stencil_batch_preserves_positions_and_scalar_values():
    scalar_positions = []
    batch_positions = []

    def scalar(x):
        scalar_positions.append(x)
        return x*x + 3*x

    def batched(x):
        batch_positions.append(x.copy())
        return x*x + 3*x

    grid = tuple(range(-4, 5))
    reference = _evaluate(scalar, grid, 1.25, 0.00137)
    result = _evaluate_batched(batched, grid, 1.25, 0.00137)
    assert len(batch_positions) == 1
    np.testing.assert_array_equal(batch_positions[0], scalar_positions)
    assert result == reference and all(type(value) is float for value in result)


@pytest.mark.parametrize('output', [1., np.ones((2, 2)), np.full(3, np.nan)])
def test_batched_result_must_match_shape_and_be_finite(output):
    with pytest.raises(ValueError):
        _evaluate_batched(lambda x: output, (-1, 0, 1), 1., .01)


def test_public_derivative_keeps_ordered_scalar_calls():
    seen = []

    def callback(x):
        assert np.ndim(x) == 0
        seen.append(float(x))
        return x*x + 3*x

    result = derivative(callback, 1.25)
    assert abs(result/5.5-1) < 1e-13
    assert len(seen) == 16
    assert seen[:9] == sorted(seen[:9]) and seen[9:] == sorted(seen[9:])


def test_public_custom_mode_never_enables_batching():
    seen = []

    def core(omega, *, z):
        assert np.ndim(omega) == 0
        seen.append(float(omega))
        return 1.0001 + z*1e-5

    mode = MarcatiliMode(125e-6, core_index=core)
    assert mode.dispersion(1, 2.35e15, z=.137) > 0
    assert len(seen) == 16


@pytest.mark.parametrize('kind,expected', [
    ('constant', False), ('rise', True), ('multi', True), ('mixture', True),
    ('radius_callback', False), ('pressure_callback', False),
    ('mixture_callback', False), ('flat_callback', False),
])
def test_only_data_defined_variable_modes_batch(kind, expected):
    radius, gas, pressure = 125e-6, 'Ar', (2., 4.)
    profile_calls = []

    def radius_callback(z):
        profile_calls.append(z)
        return 125e-6*(1+.01*z)

    def pressure_callback(z):
        profile_calls.append(z)
        return 2.+z

    def flat_callback(z):
        profile_calls.append(z)
        return 2.

    if kind == 'constant': pressure = 2.
    elif kind == 'multi': pressure = ([0., LENGTH/3, LENGTH], [2., 4., 3.])
    elif kind == 'mixture': gas, pressure = ('Ar', 'Ne'), ((2., 4.), 1.)
    elif kind == 'radius_callback': radius = radius_callback
    elif kind == 'pressure_callback': pressure = pressure_callback
    elif kind == 'mixture_callback': gas, pressure = ('Ar', 'Ne'), ((2., 4.), pressure_callback)
    elif kind == 'flat_callback': pressure = flat_callback
    model = _Capillary(radius, LENGTH, gas, pressure, **OPTIONS)
    original = model.mode.beta
    frequency_shapes = []

    def record(omega, *, z=0.):
        frequency_shapes.append(np.shape(omega))
        return original(omega, z=z)

    model.mode.beta = record
    profile_calls.clear()
    assert model.mode.dispersion(1, model.omega0, z=.137) > 0
    assert frequency_shapes == ([(9,), (7,)] if expected else [()]*16)
    if 'callback' in kind:
        assert profile_calls == [.137]*16


@pytest.mark.parametrize('order', [0, 2, 7])
def test_other_orders_keep_scalar_calls(order):
    model = _Capillary(125e-6, LENGTH, 'Ar', (2., 4.), **OPTIONS)
    original = model.mode.beta
    shapes = []

    def record(omega, *, z=0.):
        shapes.append(np.shape(omega))
        return original(omega, z=z)

    model.mode.beta = record
    assert np.isfinite(model.mode.dispersion(order, model.omega0, z=LENGTH/3))
    assert shapes and all(shape == () for shape in shapes)


CASES = [
    ('Ar', (2., 4.), dict()),
    ('Ar', (4., 2.), dict(envelope=False)),
    ('Ar', ([0., LENGTH/3, LENGTH], [2., 4., 3.]), dict(model='reduced', loss=False)),
    (('Ar', 'Ne'), ((2., 4.), 1.), dict()),
    ('N2', (2., 4.), dict(raman=True)),
    ('N2', (4., 2.), dict(raman=True, envelope=False)),
]


@pytest.mark.parametrize('gas,pressure,extra', CASES)
def test_complete_gradient_transport_and_same_input_operators(gas, pressure, extra):
    options = OPTIONS | extra
    reference = _Capillary(125e-6, LENGTH, gas, pressure, **options)
    reference.mode._batch_dispersion = False
    candidate = _Capillary(125e-6, LENGTH, gas, pressure, **options)
    for z in (0., .1*LENGTH, .371*LENGTH, LENGTH, 1.1*LENGTH):
        assert relative(candidate.mode.dispersion(1, candidate.omega0, z=z),
                        reference.mode.dispersion(1, reference.omega0, z=z)) < 1e-13
        assert relative(candidate.linear_operator(z), reference.linear_operator(z)) < 1e-13
    old = solve_envelope_model(reference)
    new = solve_envelope_model(candidate)
    assert relative(new.field, old.field) < 1e-13
    np.testing.assert_array_equal(new.z, old.z)
    for key in ('accepted_steps', 'rejected_steps'):
        assert new.metadata[key] == old.metadata[key]


def test_nonlinear_callback_trace_and_exceptions_unchanged():
    traces = [[], []]
    results = []
    for batched, trace in zip((False, True), traces):
        def response(field, context):
            trace.append((context.z, field.shape, tuple(context.densities)))
            return np.zeros_like(field)
        model = _Capillary(125e-6, LENGTH, 'Ar', (2., 4.), responses=response, **OPTIONS)
        model.mode._batch_dispersion = batched
        results.append(solve_envelope_model(model))
    assert traces[0] == traces[1] and len(traces[0]) > 10
    assert relative(results[0].field, results[1].field) < 1e-13

    error = RuntimeError('original response failure')

    def fail(field, context):
        raise error

    model = _Capillary(125e-6, LENGTH, 'Ar', (2., 4.), responses=fail, **OPTIONS)
    with pytest.raises(RuntimeError) as caught:
        solve_envelope_model(model)
    assert caught.value is error
