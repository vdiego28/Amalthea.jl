"""Validated mode normalization, physical-field synthesis and projection."""
import math

import numpy as np

from . import materials
from .quadrature import integrate

_POWER_FACTOR = .5 * math.sqrt(materials.EPS0 / (4 * math.pi * 1e-7))


def _position(z):
    value = np.asarray(z)
    if value.ndim or np.iscomplexobj(value) or not np.isfinite(value):
        raise ValueError('z must be a finite real scalar')
    return float(value)


def _domain(mode, z):
    specification = mode.dimlimits(z=z)
    if not isinstance(specification, (tuple, list)) or len(specification) != 3:
        raise ValueError('mode dimlimits must be (coordinates, lower, upper)')
    kind, lower, upper = specification
    if kind not in ('polar', 'cartesian'):
        raise ValueError('mode coordinates must be polar or cartesian')
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != (2,) or upper.shape != (2,) or np.iscomplexobj(lower) or np.iscomplexobj(upper):
        raise ValueError('mode domain requires two real lower/upper coordinates')
    lower, upper = np.array(lower, dtype=float), np.array(upper, dtype=float)
    if not np.all(np.isfinite([lower, upper])) or np.any(lower >= upper):
        raise ValueError('mode domain must be finite and increasing')
    if kind == 'polar' and lower[0] < 0:
        raise ValueError('polar radius cannot be negative')
    return kind, lower, upper


def _field(mode, points, z):
    supplied = np.asarray(mode.field((points[:, 0].copy(), points[:, 1].copy()), z=z))
    if supplied.shape != (2, len(points)):
        raise ValueError('mode field must return x/y components with shape (2, points)')
    if np.iscomplexobj(supplied) and np.any(supplied.imag != 0):
        raise ValueError('modal spatial fields must be real, as in the Julia oracle')
    value = np.array(supplied.real, dtype=float, copy=True)
    if not np.all(np.isfinite(value)):
        raise ValueError('mode field returned nonfinite values')
    return value.T


def _normalization(mode, *, z=0., rtol=1e-11):
    z = _position(z)
    kind, lower, upper = _domain(mode, z)

    def power(points):
        field = _field(mode, points, z)
        value = _POWER_FACTOR * np.sum(field**2, axis=1)
        return value * points[:, 0] if kind == 'polar' else value

    result = integrate(power, lower, upper, rtol=rtol)
    value = float(result.value.real)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('mode normalization must be positive and finite')
    return value


class _ModeSpace:
    """Immutable geometry selection; each at(z) refreshes domains and norms."""
    def __init__(self, modes, *, components='xy', full=True):
        self.modes = tuple(modes)
        if not self.modes:
            raise ValueError('modes must not be empty')
        if components not in ('x', 'y', 'xy'):
            raise ValueError('components must be x, y or xy')
        if not isinstance(full, (bool, np.bool_)):
            raise TypeError('full must be a boolean')
        for mode in self.modes:
            if not all(callable(getattr(mode, name, None)) for name in ('field', 'dimlimits', 'N')):
                raise TypeError('modes must provide field, dimlimits and N methods')
        self.indices = {'x': [0], 'y': [1], 'xy': [0, 1]}[components]
        self.full = bool(full)

    def at(self, z):
        return _SpatialSlice(self, _position(z))


class _SpatialSlice:
    def __init__(self, space, z):
        self.space, self.z = space, z
        self.kind, self.lower, self.upper = _domain(space.modes[0], z)
        scales = []
        for mode in space.modes:
            kind, lower, upper = _domain(mode, z)
            if kind != self.kind or not np.array_equal(lower, self.lower) or not np.array_equal(upper, self.upper):
                raise ValueError('all modes must share the same integration domain')
            supplied = np.asarray(mode.N(z=z))
            if (supplied.ndim or np.iscomplexobj(supplied) or not np.isfinite(supplied)
                    or supplied <= 0):
                raise ValueError('mode normalization must be a positive finite real scalar')
            scales.append(math.sqrt(float(supplied)))
        self.scales = np.asarray(scales)

    def matrix(self, points):
        """Return owned (points, modes, polarizations) physical-field matrices."""
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 2 or np.iscomplexobj(points):
            raise ValueError('spatial points must have shape (points, 2) and be real')
        points = np.array(points, dtype=float, copy=True)
        if not np.all(np.isfinite(points)):
            raise ValueError('spatial points must be finite')
        if self.kind == 'polar':
            if np.any(points[:, 0] < 0):
                raise ValueError('polar radius cannot be negative')
            inside = points[:, 0] < self.upper[0]
        else:
            inside = np.all((points > self.lower) & (points < self.upper), axis=1)
        output = np.zeros((len(points), len(self.space.modes), len(self.space.indices)))
        if np.any(inside):
            for i, mode in enumerate(self.space.modes):
                output[inside, i, :] = _field(mode, points[inside], self.z)[:, self.space.indices] / self.scales[i]
        return output

    def synthesize(self, field, points):
        field = np.array(field, dtype=complex, copy=True)
        if field.ndim != 2 or field.shape[1] != len(self.space.modes) or not np.all(np.isfinite(field)):
            raise ValueError('modal field must be finite with shape (frequency, modes)')
        return field[None, :, :] @ self.matrix(points)

    def project(self, field, response, **quadrature):
        """Integrate response(physical_spectrum, points) over the common domain.

        response receives (points, frequency, polarizations) and must return
        the same shape after temporal physics and modal spectral normalization.
        """
        field = np.array(field, dtype=complex, copy=True)
        if field.ndim != 2 or field.shape[1] != len(self.space.modes) or not np.all(np.isfinite(field)):
            raise ValueError('modal field must be finite with shape (frequency, modes)')

        def integrand(nodes):
            points = nodes if self.space.full else np.column_stack((nodes[:, 0], np.zeros(len(nodes))))
            matrix = self.matrix(points)
            physical = field[None, :, :] @ matrix
            supplied = np.asarray(response(physical, points.copy()))
            if supplied.shape != physical.shape:
                raise ValueError('spatial response must preserve the complete field shape')
            polarized = np.array(supplied, dtype=complex, copy=True)
            if not np.all(np.isfinite(polarized)):
                raise ValueError('spatial response returned nonfinite values')
            projected = polarized @ matrix.transpose(0, 2, 1)
            if self.kind == 'polar':
                projected *= points[:, 0, None, None] * (1. if self.space.full else 2*math.pi)
            # Match pointcalc!: the first-coordinate integration endpoints are zero.
            projected[(points[:, 0] <= self.lower[0]) | (points[:, 0] >= self.upper[0])] = 0
            return projected

        ndim = 2 if self.space.full else 1
        return integrate(integrand, self.lower[:ndim], self.upper[:ndim], **quadrature)
