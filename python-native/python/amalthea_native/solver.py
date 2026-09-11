"""Development-level spectral solver: Rust stepping with serial Python RHS."""
from dataclasses import dataclass
from fractions import Fraction
import math
import operator

import numpy as np

from . import _native


@dataclass
class SolveResult:
    field: np.ndarray
    z: np.ndarray
    metadata: dict


def _range_positions(start, stop, count):
    start, stop = float(start), float(stop)
    def rational(value):
        # Find a small continued-fraction convergent that round-trips exactly.
        # Julia's Float64 range uses a 2**24 bound for this detection.
        previous_p, p, previous_q, q = 0, 1, 1, 0
        residual = float(value)
        while math.isfinite(residual) and abs(residual) <= 2**24:
            integer = math.trunc(residual)
            next_p, next_q = integer*p + previous_p, integer*q + previous_q
            if max(abs(next_p), abs(next_q)) > 2**24 or next_q == 0:
                break
            candidate = Fraction(next_p, next_q)
            if float(candidate) == value:
                return candidate
            previous_p, p, previous_q, q = p, next_p, q, next_q
            remainder = residual-integer
            if remainder == 0:
                break
            residual = 1/remainder
        return None
    first, last = rational(start), rational(stop)
    if first is None or last is None:
        first, last = Fraction.from_float(float(start)), Fraction.from_float(float(stop))
    else:
        denominator = math.lcm(first.denominator, last.denominator)
        if denominator >= 2**63 or max(abs(denominator*start), abs(denominator*stop)) > 2**53:
            first, last = Fraction.from_float(float(start)), Fraction.from_float(float(stop))
    return np.array([float(first + (last-first)*Fraction(i, count-1))
                     for i in range(count)], dtype=np.float64)


def solve_precon(rhs, linop, field, zmax, *, z0=0.0, dt=1e-3, rtol=1e-6,
                 atol=1e-10, safety=0.9, min_dt=1e-15, max_dt=1.0,
                 locextrap=True, saveN=201, max_attempts=1_000_000,
                 repeat_limit=10, step_filter=None):
    """Solve ``dE/dz = linop*E + rhs(z, E)`` with a diagonal linear operator.

    Fields and callbacks use complex NumPy arrays of the original shape.
    Results add saved position as the last axis. An optional step_filter(z, E)
    returns the accepted field for the next step. Sampling/filter order matches
    Julia's solve_precon, including a final accepted step beyond zmax.
    linop may be an array or a callable linop(z) returning the field shape.
    Variable operators follow Julia's endpoint exponential convention, which
    does not integrate the linear coefficient across each propagation interval.
    """
    if not callable(rhs):
        raise TypeError("rhs must be callable")
    if step_filter is not None and not callable(step_filter):
        raise TypeError("step_filter must be callable")
    field = np.array(field, dtype=np.complex128, copy=True, order="F")
    if field.ndim == 0 or not field.size or not np.all(np.isfinite(field)):
        raise ValueError("field must be a nonempty finite array")
    linear_callback = None
    if callable(linop):
        callback = linop
        def linear_callback(z):
            output = np.asarray(callback(z), dtype=np.complex128)
            if output.shape != field.shape or not np.all(np.isfinite(output)):
                raise ValueError("linear callback must return a finite array with the field shape")
            return output.ravel(order="F").tolist()
        linop = np.zeros_like(field)
    else:
        linop = np.asarray(linop, dtype=np.complex128)
        if linop.shape != field.shape or not np.all(np.isfinite(linop)):
            raise ValueError("linop must be finite and have exactly the field shape")
    if not isinstance(locextrap, (bool, np.bool_)):
        raise TypeError("locextrap must be a boolean")
    saveN = operator.index(saveN)
    if saveN < 2:
        raise ValueError("saveN must be at least two")
    if not np.isfinite(z0) or not np.isfinite(zmax) or zmax <= z0:
        raise ValueError("zmax must be finite and greater than z0")
    positions = _range_positions(z0, zmax, saveN)

    def adapt(callback):
        def invoke(z, values):
            # Owned arrays: a user may retain or mutate their input safely.
            incoming = np.array(values, dtype=np.complex128).reshape(field.shape, order="F").copy(order="F")
            output = np.asarray(callback(z, incoming), dtype=np.complex128)
            if output.shape != field.shape or not np.all(np.isfinite(output)):
                raise ValueError("callback must return a finite array with the field shape")
            return output.ravel(order="F").tolist()
        return invoke

    samples, accepted, rejected, endpoints = _native.solve(
        adapt(rhs), linop.ravel(order="F").tolist(), field.ravel(order="F").tolist(),
        positions.tolist(), dt, rtol, atol, safety, min_dt, max_dt, bool(locextrap),
        operator.index(max_attempts), operator.index(repeat_limit),
        None if step_filter is None else adapt(step_filter),
        linear_callback,
    )
    result = np.stack([np.asarray(x).reshape(field.shape, order="F") for x in samples], axis=-1)
    return SolveResult(result, positions, {
        "stepper": "rust-precon", "rhs": "python", "locextrap": bool(locextrap),
        "accepted_steps": accepted, "rejected_steps": rejected,
        "accepted_positions": np.asarray(endpoints),
        "linear_operator": "python" if linear_callback is not None else "constant",
    })
