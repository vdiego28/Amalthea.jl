"""Exact scalar pressure profiles and Julia's structured-gradient convention."""
import math
import numpy as np

from . import materials
from ._spline import _NormalizedCubic
from .grid import C
from .modes import _scalar
from .solver import _range_positions


class _GasProfile:
    def __init__(self, gas, pressure, length, temperature):
        self.gas = materials._gas(gas)
        self.temperature = _scalar(temperature, 'temperature', positive=True)
        self.spline = None
        self.constant = None
        self.callback = pressure if callable(pressure) else None
        if self.callback is not None:
            self.kind = 'callable'
        elif np.isscalar(pressure) or isinstance(pressure, np.ndarray) and pressure.ndim == 0:
            self.kind = 'constant'
            self.constant = _scalar(pressure, 'pressure', nonnegative=True)
            self.constant_density = materials.density(self.gas, self.constant, self.temperature)
        else:
            self.kind = 'gradient'
            if len(pressure) != 2:
                raise ValueError('pressure gradient must be (p0,p1) or (positions,pressures)')
            if all(np.ndim(value) == 0 for value in pressure):
                positions, pressures = [0., length], pressure
            else:
                positions, pressures = pressure
            if np.iscomplexobj(positions) or np.iscomplexobj(pressures):
                raise ValueError('gradient positions and pressures must be real')
            self.positions = np.array(positions, dtype=float, copy=True)
            self.pressures = np.array(pressures, dtype=float, copy=True)
            if (self.positions.ndim != 1 or len(self.positions) < 2
                    or self.pressures.shape != self.positions.shape
                    or not np.all(np.isfinite(self.positions))
                    or not np.all(np.isfinite(self.pressures))
                    or np.any(np.diff(self.positions) <= 0) or np.any(self.pressures < 0)):
                raise ValueError('invalid pressure gradient points')
            low, high = float(self.pressures.min()), float(self.pressures.max())
            if high == 0:
                raise ValueError('all-zero pressure gradient has an undefined Julia density spline')
            if low == high:
                low = 0.
            nodes = _range_positions(low, high, 1024)
            self.spline = _NormalizedCubic(nodes, materials.density(self.gas, nodes, self.temperature))

    def pressure(self, z):
        z = _scalar(z, 'z')
        if self.callback is not None:
            return _scalar(self.callback(z), 'pressure', nonnegative=True)
        if self.kind == 'constant':
            return self.constant
        if z <= self.positions[0]:
            return float(self.pressures[0])
        if z >= self.positions[-1]:
            return float(self.pressures[-1])
        i = np.searchsorted(self.positions, z, side='left')-1
        fraction = (z-self.positions[i])/(self.positions[i+1]-self.positions[i])
        return math.sqrt(self.pressures[i]**2+fraction*(self.pressures[i+1]**2-self.pressures[i]**2))

    def density(self, z):
        if self.kind == 'constant':
            return self.constant_density
        pressure = self.pressure(z)
        value = (materials.density(self.gas, pressure, self.temperature) if self.spline is None
                 else self.spline(pressure))
        return _scalar(value, 'density', nonnegative=True)

    def core_index(self, omega, *, z):
        return materials._owned(np.sqrt(1+np.asarray(
            materials.polarizability(self.gas, 2*math.pi*C/omega)*self.density(z), dtype=complex)))


class _MixtureProfile:
    def __init__(self, gases, pressure, length, temperature):
        if (not isinstance(pressure, (list, tuple, np.ndarray))
                or isinstance(pressure, np.ndarray) and pressure.ndim == 0):
            raise ValueError('mixture pressure must contain one specification per gas')
        if len(pressure) != len(gases):
            raise ValueError('mixture gas and pressure counts must match')
        self.profiles = tuple(_GasProfile(gas, partial, length, temperature)
                              for gas, partial in zip(gases, pressure))
        self.kind = 'constant' if all(p.kind == 'constant' for p in self.profiles) else 'variable'

    def density(self, z):
        return np.array([profile.density(z) for profile in self.profiles])

    def core_index(self, omega, *, z):
        wavelength = 2*math.pi*C/omega
        susceptibility = np.zeros(np.shape(omega), dtype=complex)
        for profile in self.profiles:
            susceptibility += materials.polarizability(profile.gas, wavelength)*profile.density(z)
        return materials._owned(np.sqrt(1+susceptibility))
