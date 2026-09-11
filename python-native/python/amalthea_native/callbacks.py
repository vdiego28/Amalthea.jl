"""Owned complete-array nonlinear response callbacks."""
from dataclasses import dataclass

import numpy as np


@dataclass
class ResponseContext:
    """Per-invocation setup in SI units, safe for the callback to retain or mutate.

    ``t`` is the complete oversampled time axis in seconds; ``densities`` are
    number densities in m^-3 in ``gases`` order (both empty for GNLSE).
    Modal coordinates are the actual polar/cartesian integration point;
    mode-average and GNLSE callbacks have no spatial coordinates.
    """
    z: float
    t: np.ndarray
    is_real: bool
    gases: tuple
    densities: np.ndarray
    coordinate_system: str | None
    coordinates: tuple | None
    components: tuple


class _Responses:
    def __init__(self, responses):
        if responses is None:
            responses=()
        elif callable(responses):
            responses=(responses,)
        elif not isinstance(responses,(list,tuple)):
            raise TypeError('responses must be a callable or sequence of callables')
        self.callbacks=tuple(responses)
        if not all(callable(callback) for callback in self.callbacks):
            raise TypeError('responses must contain only callables')
        self.calls=0

    def __bool__(self):
        return bool(self.callbacks)

    def metadata(self):
        return [dict(callable=getattr(callback,'__qualname__',type(callback).__qualname__))
                for callback in self.callbacks]

    def accumulate(self, polarization, field, *, z, grid, gases=(), densities=(),
                   coordinate_system=None, coordinates=None, components=('y',)):
        scalar=field.ndim==1
        field=field[:,None] if scalar else field
        expected=(len(grid.to),len(components))
        if field.shape!=expected:
            raise ValueError('response field must contain the complete time/component array')
        for callback in self.callbacks:
            incoming=np.array(field,copy=True)
            context=ResponseContext(float(z),np.array(grid.to,copy=True),bool(grid.is_real),
                tuple(gases),np.array(densities,dtype=float,copy=True),coordinate_system,
                None if coordinates is None else tuple(float(value) for value in coordinates),tuple(components))
            self.calls+=1
            supplied=np.asarray(callback(incoming,context))
            if supplied.shape!=expected:
                raise ValueError('nonlinear response must preserve the complete time/component shape')
            if supplied.dtype.kind not in 'iufc':
                raise ValueError('nonlinear response must return a numeric polarization array')
            if grid.is_real and np.iscomplexobj(supplied):
                if np.any(supplied.imag!=0):
                    raise ValueError('carrier nonlinear response must be real')
                supplied=supplied.real
            value=np.array(supplied,dtype=float if grid.is_real else complex,copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('nonlinear response returned nonfinite polarization')
            polarization+=value[:,0] if scalar else value
