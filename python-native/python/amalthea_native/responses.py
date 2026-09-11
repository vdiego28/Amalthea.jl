"""Per-species scalar response setup and polarization accumulation."""
from collections.abc import Mapping
import math

import numpy as np

from . import materials
from .envelope import EPS0
from .grid import C
from .ionisation import IonRateADK
from .molecular import MolecularRaman
from .plasma import _PlasmaResponse
from .ppt import IonRatePPT, IonRatePPTAccel
from .pulses import _analytic_signal

_SPECIES_KEYS = frozenset(('kerr', 'raman', 'plasma', 'rotation', 'vibration',
                           'PPT_options', 'preionfrac'))
_RAMAN_GASES = ('N2', 'H2', 'D2', 'N2O', 'CH4', 'SF6')


def species_options(gases, options, overrides, real):
    if overrides is None:
        overrides = [{} for _ in gases]
    if (not isinstance(overrides, (list, tuple)) or len(overrides) != len(gases)
            or not all(isinstance(item, Mapping) for item in overrides)):
        raise ValueError('species_options must contain one mapping per gas')
    result = []
    for gas, supplied in zip(gases, overrides):
        normalized = {}
        for key, value in supplied.items():
            key = 'PPT_options' if key == 'ppt_options' else key
            if key not in _SPECIES_KEYS:
                raise TypeError(f'unsupported species option: {key}')
            if key in normalized:
                raise TypeError(f'duplicate species option alias: {key}')
            normalized[key] = value
        selected = {key: options[key] for key in _SPECIES_KEYS} | normalized
        for key in ('kerr', 'rotation', 'vibration'):
            if not isinstance(selected[key], (bool, np.bool_)):
                raise TypeError(f'{key} must be a boolean')
        if selected['raman'] is None:
            selected['raman'] = gas in _RAMAN_GASES
        elif not isinstance(selected['raman'], (bool, np.bool_)):
            raise TypeError('raman must be a boolean or None')
        plasma = real if selected['plasma'] is None else selected['plasma']
        if not isinstance(plasma, (bool, np.bool_, str, IonRateADK, IonRatePPT, IonRatePPTAccel)):
            raise TypeError('plasma must be a boolean, ADK/PPT name or ionisation model')
        if isinstance(plasma, str) and plasma not in ('ADK', 'PPT'):
            raise ValueError('unknown plasma model')
        if plasma and not real:
            raise NotImplementedError('envelope plasma is not supported by the Julia oracle')
        if isinstance(plasma, (bool, np.bool_)) and plasma:
            plasma = 'ADK' if gas in ('H2', 'D2', 'N2O', 'CH4', 'SF6') else 'PPT'
        selected['plasma'] = plasma
        ppt = {} if selected['PPT_options'] is None else selected['PPT_options']
        if not isinstance(ppt, Mapping):
            raise TypeError('PPT_options must be a mapping')
        ppt = dict(ppt)
        if ppt.get('cachedir') is not None:
            ppt['cachedir'] = str(ppt['cachedir'])
        if ppt and plasma != 'PPT':
            raise ValueError('PPT_options requires PPT table construction')
        selected['PPT_options'] = ppt
        preionfrac = float(selected['preionfrac'])
        if not math.isfinite(preionfrac) or not 0 <= preionfrac <= 1:
            raise ValueError('preionfrac must be in [0,1]')
        selected['preionfrac'] = preionfrac
        result.append(selected)
    return tuple(result)


class _ScalarGasResponse:
    def __init__(self, grid, gas, options, thg):
        self.gas, self.options = gas, options
        self.real, self.thg = grid.is_real, thg
        self.no, self.dt = len(grid.to), grid.to[1] - grid.to[0]
        gamma3 = materials.gamma3(gas) if options['kerr'] else 0.
        if np.imag(gamma3) != 0:
            raise NotImplementedError('complex Kerr coefficients require a Python response callback')
        self.gamma3 = float(np.real(gamma3))
        self.plasma = None
        selection = options['plasma']
        if selection:
            if isinstance(selection, str):
                rate = (IonRateADK(gas) if selection == 'ADK' else
                        IonRatePPTAccel(gas, grid.reference_lambda, **options['PPT_options']))
            else:
                rate = selection
            loss_model = rate.model if isinstance(rate, IonRatePPTAccel) else rate
            ionpot = materials.ionisation_potential(gas) if loss_model is None else loss_model.ionpot
            self.plasma = _PlasmaResponse(self.dt, rate, ionpot, options['preionfrac'])
        self.raman_response = (MolecularRaman(grid.to, gas, rotation=options['rotation'],
                               vibration=options['vibration'], temperature=options['temperature'])
                               if options['raman'] else None)
        self.h = self.hfft = None
        self.thg_phase = (np.exp(2j * (2 * math.pi * C / grid.reference_lambda) * grid.to)
                          if not self.real and thg else None)

    def refresh(self, density):
        self.density = density
        self.kerr = (1. if self.real else .75) * density * EPS0 * self.gamma3
        if self.raman_response is not None:
            self.h = density * self.raman_response(density)
            self.hfft = np.fft.rfft(np.pad(self.h, (0, self.no))) * self.dt

    def accumulate(self, polarization, time):
        if self.real:
            polarization += (self.kerr * time**3 if self.thg else
                             .75 * self.kerr * abs(_analytic_signal(time))**2 * time)
        else:
            polarization += self.kerr * time * np.abs(time)**2
            if self.thg:
                polarization += self.kerr / 3 * self.thg_phase * time**3
        if self.plasma is not None:
            polarization += self.density * self.plasma(time)
        if self.h is not None:
            intensity = (time**2 if self.thg else .5 * abs(_analytic_signal(time))**2
                         ) if self.real else .5 * np.abs(time)**2
            drive = np.fft.rfft(np.pad(intensity, (0, self.no)))
            polarization += time * np.fft.irfft(self.hfft * drive, n=2*self.no)[:self.no]


class _VectorGasResponse(_ScalarGasResponse):
    def __init__(self, grid, gas, options, thg):
        if options['raman']:
            raise NotImplementedError('vector Raman is not supported by the Julia oracle')
        if grid.is_real and not thg and options['kerr']:
            raise NotImplementedError('carrier vector Kerr with thg=False is not supported by the Julia oracle')
        super().__init__(grid, gas, options, thg)

    def accumulate(self, polarization, time):
        x,y=time[:,0],time[:,1]
        if self.real:
            intensity=x*x+y*y
            polarization[:,0]+=self.kerr*intensity*x
            polarization[:,1]+=self.kerr*intensity*y
        elif self.thg:
            polarization+=self.kerr*time*np.abs(time)**2
            polarization+=(self.kerr/3)*self.thg_phase[:,None]*time**3
        else:
            x2,y2=np.abs(x)**2,np.abs(y)**2
            polarization[:,0]+=self.kerr*((x2+(2/3)*y2)*x+(1/3)*np.conj(x)*y*y)
            polarization[:,1]+=self.kerr*((y2+(2/3)*x2)*y+(1/3)*np.conj(y)*x*x)
        if self.plasma is not None:
            polarization+=self.density*self.plasma.vector(time)
