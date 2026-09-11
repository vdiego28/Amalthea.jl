"""Fixed silica Raman model from PhysData.jl / Raman.jl (SI units)."""
from functools import lru_cache
import math

import numpy as np
from scipy.special import erfcx

from .grid import C

# Hollenbeck–Cantrell, as tabulated in PhysData.raman_parameters(:SiO2).
_OMEGA = 200 * math.pi * C * np.array([
    56.25, 100., 231.25, 362.50, 463., 497., 611.50,
    691.67, 793.67, 835.50, 930., 1080., 1215.,
])
_AMPLITUDE = np.array([1., 11.40, 36.67, 67.67, 74., 4.50, 6.80,
                       4.60, 4.20, 4.50, 2.70, 3.10, 3.])
_GAUSSIAN = 100 * math.pi * C * np.array([
    52.10, 110.42, 175., 162.50, 135.33, 24.50, 41.50,
    155., 59.50, 64.30, 150., 91., 160.,
])
_LORENTZIAN = 100 * math.pi * C * np.array([
    17.37, 38.81, 58.33, 54.17, 45.11, 8.17, 13.83,
    51.67, 19.83, 21.43, 50., 30.33, 53.33,
])
for _coefficients in (_OMEGA, _AMPLITUDE, _GAUSSIAN, _LORENTZIAN):
    _coefficients.flags.writeable = False


def _sio2_raw(t):
    """Unnormalized causal response, evaluated in the oracle's component order."""
    t = np.asarray(t, dtype=float)
    positive = np.maximum(t, 0.)
    response = np.zeros_like(t)
    for omega, amplitude, gaussian, lorentzian in zip(
        _OMEGA, _AMPLITUDE, _GAUSSIAN, _LORENTZIAN
    ):
        response += (amplitude * np.exp(-lorentzian * positive)
                     * np.exp(-gaussian**2 * positive**2 / 4) * np.sin(omega * positive))
    return response


@lru_cache(maxsize=1)
def _sio2_normalization():
    """Continuous integral; the omitted tail after Julia's 1 ns is < exp(-1e6)."""
    values = _AMPLITUDE * math.sqrt(math.pi) / _GAUSSIAN * erfcx(
        (_LORENTZIAN - 1j * _OMEGA) / _GAUSSIAN
    ).imag
    return float(np.sum(values))


def _sio2_response(t, scale):
    # No tail window or sampled-integral renormalization for this Julia model.
    return (scale / _sio2_normalization()) * _sio2_raw(t)
