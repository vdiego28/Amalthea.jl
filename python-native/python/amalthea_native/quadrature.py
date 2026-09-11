"""Serial SciPy integration with the modal Cubature.L2 acceptance criterion."""
from dataclasses import dataclass
import math
import operator

import numpy as np
from scipy.integrate import cubature


def _norm(values, multiplier=1.):
    values = np.asarray(values)
    if np.iscomplexobj(values):
        values = np.stack((values.real, values.imag), axis=-1)
    values = np.abs(values).ravel()
    scale = float(np.max(values, initial=0.))
    return (scale * multiplier) * math.sqrt(float(np.sum((values / scale)**2))) if scale else 0.


@dataclass
class _Integral:
    value: np.ndarray
    error: np.ndarray
    error_norm: float
    tolerance: float
    evaluations: int


def integrate(function, lower, upper, *, rtol=1e-3, atol=0., maxevals=100_000,
              rule='gk21', batch_size=32):
    """Integrate a batched array callback and enforce one global real/imag norm.

    The callback takes (points, dimensions), returning (points, *field_shape).
    Error estimates are returned with a trailing (real, imaginary) axis.
    Nonconvergence raises instead of returning an unchecked integral.
    """
    lower, upper = np.asarray(lower), np.asarray(upper)
    if (np.iscomplexobj(lower) or np.iscomplexobj(upper) or lower.ndim != 1
            or upper.shape != lower.shape or lower.size not in (1, 2)):
        raise ValueError('integration limits must be matching real 1-D or 2-D bounds')
    lower, upper = lower.astype(float), upper.astype(float)
    if not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper):
        raise ValueError('integration limits must be finite and increasing')
    rtol, atol = float(rtol), float(atol)
    if not math.isfinite(rtol) or not math.isfinite(atol) or min(rtol, atol) < 0 or max(rtol, atol) == 0:
        raise ValueError('quadrature tolerances must be nonnegative with one positive')
    maxevals, batch_size = operator.index(maxevals), operator.index(batch_size)
    if maxevals < 1 or batch_size < 1:
        raise ValueError('quadrature evaluation and batch limits must be positive')
    if rule not in ('gk21', 'gk15', 'genz-malik') or rule == 'genz-malik' and lower.size == 1:
        raise ValueError('unsupported quadrature rule for this dimension')
    evaluations, shape = 0, None

    def checked(points):
        nonlocal evaluations, shape
        parts = []
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            if evaluations + len(batch) > maxevals:
                raise RuntimeError(f'modal quadrature exceeded maxevals={maxevals}')
            evaluations += len(batch)
            supplied = np.asarray(function(np.array(batch, copy=True)))
            if supplied.ndim < 1 or supplied.shape[0] != len(batch) or supplied.size == 0:
                raise ValueError('integrand must return (points, *field_shape)')
            if shape is None:
                shape = supplied.shape[1:]
            if supplied.shape[1:] != shape:
                raise ValueError('integrand changed its field shape')
            value = np.array(supplied, dtype=complex, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('integrand returned nonfinite values')
            parts.append(np.stack((value.real, value.imag), axis=-1))
        return np.concatenate(parts, axis=0)

    # The pilot supplies the global scale without chasing tiny components.
    result = cubature(checked, lower, upper, rule=rule, rtol=rtol, atol=atol,
                      max_subdivisions=0, workers=1)
    for attempt in range(8):
        value = result.estimate[..., 0] + 1j * result.estimate[..., 1]
        error = result.error
        if not np.all(np.isfinite(value)) or not np.all(np.isfinite(error)) or np.any(error < 0):
            raise RuntimeError('quadrature returned an invalid estimate or error')
        tolerance = max(atol, _norm(value, rtol))
        error_norm = _norm(error)
        if not math.isfinite(tolerance) or not math.isfinite(error_norm):
            raise RuntimeError('modal quadrature global norm exceeds the finite numeric range')
        if error_norm <= tolerance:
            return _Integral(np.array(value, copy=True), np.array(error, copy=True),
                             error_norm, tolerance, evaluations)
        if attempt == 7 or tolerance == 0:
            break
        component_budget = tolerance / (2 * math.sqrt(error.size))
        if component_budget == 0:
            break
        result = cubature(checked, lower, upper, rule=rule, rtol=0.,
                          atol=component_budget, max_subdivisions=min(4**(attempt+1),maxevals), workers=1)
    raise RuntimeError(f'modal quadrature did not meet global L2 tolerance: {error_norm} > {tolerance}')
