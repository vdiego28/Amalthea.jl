"""Shared envelope evaluation and constant mode-averaged solver adapter."""
import math
import operator
import numpy as np
from .grid import C
from .solver import solve_precon, SolveResult, _range_positions
from . import _native

EPS0 = 1 / (4 * math.pi * 1e-7 * C**2)
NLSCALE = math.sqrt(EPS0 * C / 2)


class _EnvelopeModel:
    def native_plasma_config(self):
        return getattr(self,'_native_plasma',None)

    def native_config(self):
        return (self.pre.tolist(), self.grid.towin.tolist(), self.grid.twin.tolist(),
                self.grid.omega_win.tolist(), self.kerr / (1. if self.grid.is_real else .75), self.amplitude_scale,
                None if self.h is None else self.h.tolist(), self.grid.to[1]-self.grid.to[0])

    def solve_native(self, max_dz):
        kw = self.kw
        if not isinstance(kw['locextrap'], (bool, np.bool_)):
            raise TypeError('locextrap must be a boolean')
        count = operator.index(kw['saveN'])
        if count < 2:
            raise ValueError('saveN must be at least two')
        positions = _range_positions(0., self.grid.zmax, count)
        native_solve = _native.solve_real if self.grid.is_real else _native.solve_envelope
        extra={'plasma':self.native_plasma_config()} if self.grid.is_real else {}
        samples, accepted, rejected, endpoints = native_solve(
            self.linop.tolist(), self.initial.tolist(), self.native_config(), positions.tolist(),
            kw['init_dz'], kw['rtol'], kw['atol'], kw['safety'], kw['min_dz'], max_dz,
            bool(kw['locextrap']), operator.index(kw['max_attempts']), operator.index(kw['repeat_limit']),**extra)
        return SolveResult(np.array(samples, dtype=complex).T.copy(), positions, {
            'stepper': 'rust-resident', 'rhs': 'rust', 'locextrap': bool(kw['locextrap']),
            'accepted_steps': accepted, 'rejected_steps': rejected,
            'accepted_positions': np.asarray(endpoints),
        })

    def rhs(self, z, field):
        half = self.n // 2
        spectrum = np.zeros(self.no, dtype=complex)
        spectrum[:half] = field[:half] * (self.no / self.n)
        spectrum[-half:] = field[-half:] * (self.no / self.n)
        time = np.fft.ifft(spectrum) / self.amplitude_scale
        polarization = self.polarization(time,z=z)
        transformed = np.fft.fft(polarization * self.grid.towin)
        return np.concatenate((transformed[:half], transformed[-half:])) * (self.n / self.no) * self.pre

    def polarization(self, time, *, z=0.):
        intensity = np.abs(time)**2
        polarization = self.kerr * time * intensity
        if self.h is not None:
            drive = np.fft.rfft(np.pad(.5 * intensity, (0, self.no)))
            polarization += time * np.fft.irfft(self.hfft * drive, n=2 * self.no)[:self.no]
        if getattr(self,'custom_responses',None):
            self.custom_responses.accumulate(polarization,time,z=z,grid=self.grid)
        return polarization

    def window(self, z, field):
        return np.fft.fft(np.fft.ifft(field * self.grid.omega_win) * self.grid.twin)


def solve_envelope_model(model):
    kw = model.kw
    max_dz = model.grid.zmax / 2 if kw['max_dz'] is None else kw['max_dz']
    if kw['backend'] in ('native', 'auto') and getattr(model,'native_eligible',True):
        result = model.solve_native(max_dz)
        result.metadata.update(backend='native', fft='rustfft/realfft', requested_backend=kw['backend'],
                               backend_reason='resident CPU mode average with portable transforms')
    else:
        result = solve_precon(model.rhs, model.linop, model.initial, model.grid.zmax,
                             dt=kw['init_dz'], min_dt=kw['min_dz'], max_dt=max_dz,
                             rtol=kw['rtol'], atol=kw['atol'], safety=kw['safety'],
                             locextrap=kw['locextrap'], saveN=kw['saveN'],
                             max_attempts=kw['max_attempts'], repeat_limit=kw['repeat_limit'],
                             step_filter=model.window)
        result.metadata.update(backend='python', fft='numpy', requested_backend=kw['backend'],
                               backend_reason=('Python evaluator selected' if kw['backend']=='python' else
                                               getattr(model,'backend_reason','Python evaluator required')))
    custom=getattr(model,'custom_responses',None)
    if custom:
        result.metadata.update(custom_response_calls=custom.calls,
                               custom_responses=custom.metadata(),response_evaluator='python')
    return result
