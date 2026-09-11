"""Owned input pulses following Fields.jl's carrier and envelope conventions."""
import copy
import math

import numpy as np
from scipy.interpolate import splrep, splev

from .grid import C, _positive

_ALIASES = {
    'λ0': 'lambda0', 'lam0': 'lambda0', 'wavelength': 'lambda0',
    'λlims': 'lambda_lims', 'lam_lims': 'lambda_lims', 'wavelength_lims': 'lambda_lims',
    'τfwhm': 'tau_fwhm', 'tfwhm': 'tau_fwhm', 'duration': 'tau_fwhm',
    'τw': 'tau_w', 'tw': 'tau_w', 'ϕ': 'phi', 'φ': 'phi', 'phase': 'phi',
    'δt': 'delta_t', 'dt': 'delta_t', 'τ1': 'tau1', 'τ2': 'tau2',
    'polarization': 'polarisation',
}
def _vector(value, name, *, complex_values=False):
    result = np.array(value, dtype=np.complex128 if complex_values else float, copy=True)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be a finite one-dimensional array')
    return result


def _analytic_signal(field):
    field=np.asarray(field)
    if field.ndim!=1 or np.iscomplexobj(field):
        raise ValueError('analytic signal requires a real one-dimensional field')
    spectrum=np.fft.fft(field)
    n=field.size
    spectrum[1:(n+1)//2]*=2
    spectrum[n//2+1:]=0
    return np.fft.ifft(spectrum)


def _intensity(grid,field):
    return np.abs(_analytic_signal(field) if grid.is_real else field)**2


def _simpson(axis,power):
    if power.shape != axis.shape or power.size < 8:
        raise ValueError('energy integration requires at least eight grid-matched samples')
    edges = sum(w * (power[i] + power[-i-1]) for i, w in enumerate((17, 59, 43, 49))) / 48
    return (axis[1] - axis[0]) * (edges + np.sum(power[4:-4]))


def energy_t(grid, field):
    """Pulse energy from analytic-signal intensity and extended Simpson weights."""
    return _simpson(grid.t,_intensity(grid,field))


def _forward(grid,field):
    return np.fft.rfft(field) if grid.is_real else np.fft.fft(field)


def _inverse(grid,spectrum):
    return np.fft.irfft(spectrum,n=grid.t.size) if grid.is_real else np.fft.ifft(spectrum)


def _initial(grid, kw):
    if kw['pulse_domain'] not in ('time', 'frequency'):
        raise ValueError('pulse_domain must be time or frequency')
    if kw['pulse'] is not None:
        if kw['tau_fwhm'] is not None or kw['tau_w'] is not None:
            raise ValueError('pulse arrays cannot also specify a pulse duration')
        frequency=kw['pulse_domain']=='frequency'
        if grid.is_real and not frequency and np.iscomplexobj(kw['pulse']):
            raise ValueError('carrier-resolved time pulse must be real')
        pulse = _vector(kw['pulse'], 'pulse', complex_values=frequency or not grid.is_real)
        if pulse.shape != (grid.omega.shape if frequency else grid.t.shape):
            raise ValueError('pulse must match the constructed grid')
        time = _inverse(grid,pulse) if frequency else pulse
    else:
        if kw['pulse_domain'] != 'time':
            raise ValueError('pulse_domain requires an array pulse')
        if kw['pulseshape'] == 'gauss':
            if kw['tau_w'] is not None:
                raise ValueError('Gaussian pulses require tau_fwhm')
            width = _positive('tau_fwhm', kw['tau_fwhm'])
            time = np.exp(-2 * math.log(2) * (grid.t / width)**2).astype(complex)
        elif kw['pulseshape'] == 'sech':
            if (kw['tau_w'] is None) == (kw['tau_fwhm'] is None):
                raise ValueError('sech requires exactly one of tau_w or tau_fwhm')
            width = (_positive('tau_w', kw['tau_w']) if kw['tau_w'] is not None
                     else _positive('tau_fwhm', kw['tau_fwhm']) / (2 * math.log(1 + math.sqrt(2))))
            decay = np.exp(-np.abs(grid.t / width))
            time = (2 * decay / (1 + decay**2)).astype(complex)
        else:
            raise ValueError('pulseshape must be gauss or sech')
    if kw['pulse'] is None:
        if grid.is_real:
            time=time.real*np.cos((2*math.pi*C/kw['lambda0'])*grid.t)
        else:
            time *= np.exp(1j * (2 * math.pi * C / kw['lambda0'] - grid.omega0) * grid.t)
    phase = _vector(kw['phi'], 'phi')
    if phase.size:
        offset = grid.omega - (2 * math.pi * C / kw['lambda0'])
        phi = sum((offset**i / math.factorial(i)) * v for i, v in enumerate(phase))
        time = _inverse(grid,_forward(grid,time)*np.exp(-1j*phi))
    energy, power = kw['energy'], kw['power']
    if energy is not None and power is not None:
        raise ValueError('specify only one of energy or power')
    if energy is None and power is None and kw['pulse'] is None:
        raise ValueError('specify energy or power')
    if energy is not None:
        target, actual = _positive('energy', energy), energy_t(grid, time)
    elif power is not None:
        target, actual = _positive('power', power), np.max(_intensity(grid,time))
    else:
        return _forward(grid,time)
    if not np.isfinite(actual) or actual <= 0:
        raise ValueError('cannot normalize a zero or nonfinite pulse')
    return _forward(grid,time*(math.sqrt(target)/math.sqrt(actual)))



def _options(kwargs, defaults):
    supplied = {}
    for key, value in kwargs.items():
        key = _ALIASES.get(key, key)
        if key in supplied:
            raise TypeError(f'duplicate pulse keyword alias for {key}')
        if key not in defaults:
            raise TypeError(f'unsupported pulse keyword: {key}')
        supplied[key] = copy.deepcopy(value) if not callable(value) else value
    return defaults | supplied


def _apply_propagator(field, grid, propagator):
    if propagator is None:
        return field
    if not callable(propagator):
        raise TypeError('propagator must be callable')
    incoming = np.array(field, dtype=complex, copy=True)
    result = propagator(incoming, copy.deepcopy(grid))
    output = _vector(incoming if result is None else result, 'propagator output', complex_values=True)
    if output.shape != grid.omega.shape:
        raise ValueError('propagator output must have exactly the field shape')
    return output


def pulse_metadata(value):
    if isinstance(value, (GaussPulse, SechPulse, DataPulse, PropagatedPulse)):
        return value.metadata()
    if callable(value):
        return {'callable': getattr(value, '__qualname__', type(value).__name__)}
    if isinstance(value, (list, tuple)):
        return [pulse_metadata(x) for x in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: pulse_metadata(v) for k, v in value.items()}
    return value


class GaussPulse:
    """Gaussian pulse with independent wavelength, duration, energy/peak power."""
    shape = 'gauss'

    def __init__(self, **kwargs):
        self.options = _options(kwargs, dict(lambda0=None, tau_fwhm=None, tau_w=None,
                                             energy=None, power=None, phi=(), propagator=None,
                                             mode='lowest', polarisation='linear'))
        self.options['lambda0'] = _positive('lambda0', self.options['lambda0'])
        _vector(self.options['phi'], 'phi')
        if self.options['propagator'] is not None and not callable(self.options['propagator']):
            raise TypeError('propagator must be callable')

    def spectrum(self, grid):
        kw = self.options | dict(pulse=None, pulse_domain='time', pulseshape=self.shape)
        return _apply_propagator(_initial(grid, kw), grid, self.options['propagator'])

    def metadata(self):
        return {'type': type(self).__name__} | pulse_metadata(self.options)


class SechPulse(GaussPulse):
    """Sech pulse; supply tau_w or intensity tau_fwhm."""
    shape = 'sech'


def energy_omega(grid, spectrum):
    if grid.is_real:
        return 2*math.pi/grid.omega[-1]**2*_simpson(grid.omega,np.abs(spectrum)**2)
    delta = grid.omega[1] - grid.omega[0]
    return 2 * math.pi * delta / (len(grid.omega)*delta)**2 * np.sum(np.abs(spectrum)**2)


class DataPulse:
    """Spectrum on a supplied angular-frequency axis (rad/s), centered like DataField.

    Use DataPulse(omega, complex_field, energy=...) or
    DataPulse(omega, intensity, phase, energy=...). Unlike pulse_domain='frequency',
    this interpolates and applies Julia's half-window centering phase.
    """
    def __init__(self, omega, values, phase=None, **kwargs):
        self.options = _options(kwargs, dict(energy=None, lambda0=None, phi=(), propagator=None,
                                             mode='lowest', polarisation='linear'))
        self.options['energy'] = _positive('energy', self.options['energy'])
        if self.options['lambda0'] is not None:
            self.options['lambda0'] = _positive('lambda0', self.options['lambda0'])
        self.omega = _vector(omega, 'omega')
        if self.omega.size < 4 or np.any(self.omega <= 0):
            raise ValueError('omega must contain at least four positive frequencies')
        order = np.argsort(self.omega)
        self.omega = self.omega[order]
        if np.any(np.diff(self.omega) <= 0):
            raise ValueError('omega coordinates must be unique')
        values = _vector(values, 'spectrum', complex_values=phase is None)
        if values.shape != self.omega.shape:
            raise ValueError('spectrum must have the omega shape')
        if phase is None:
            self.intensity = (np.abs(values)**2)[order]
            self.phase = np.unwrap(np.angle(values))[order]
        else:
            phases = _vector(phase, 'spectral phase')
            if phases.shape != values.shape:
                raise ValueError('spectral phase must have the omega shape')
            self.intensity, self.phase = values[order], phases[order]
        if not np.all(np.isfinite(self.intensity)) or np.any(self.intensity < 0) or not np.any(self.intensity > 0):
            raise ValueError('spectral intensity must be finite, nonnegative, and nonzero')
        self.options['phi'] = _vector(self.options['phi'], 'phi')
        if self.options['propagator'] is not None and not callable(self.options['propagator']):
            raise TypeError('propagator must be callable')
        self._intensity_spline = splrep(self.omega, self.intensity, k=3, s=0)
        self._phase_spline = splrep(self.omega, self.phase, k=3, s=0)

    def spectrum(self, grid):
        phase = splev(grid.omega, self._phase_spline)
        intensity = np.maximum(splev(grid.omega, self._intensity_spline), 0)
        intensity[(grid.omega <= self.omega[0]) | (grid.omega >= self.omega[-1])] = 0
        intensity *= grid.omega_win
        spectrum = np.sqrt(intensity) * np.exp(1j * phase)
        energy = energy_omega(grid, spectrum)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError('data pulse has no finite energy inside the simulation window')
        spectrum *= math.sqrt(self.options['energy'] / energy)
        tau = len(grid.t)*(grid.t[1]-grid.t[0])/2
        spectrum *= np.exp(-1j * grid.omega * tau)
        if len(self.options['phi']):
            center = (2 * math.pi * C / self.options['lambda0'] if self.options['lambda0'] is not None
                      else np.sum(self.omega*self.intensity) / np.sum(self.intensity))
            # Match DataField's frequency -> wavelength -> frequency roundtrip.
            if self.options['lambda0'] is None:
                center = 2 * math.pi * C / (2 * math.pi * C / center)
            offset = grid.omega - center
            phase = sum(offset**i / math.factorial(i) * v for i, v in enumerate(self.options['phi']))
            spectrum *= np.exp(-1j * phase)
        return _apply_propagator(spectrum, grid, self.options['propagator'])

    def metadata(self):
        return {'type':'DataPulse', 'omega':self.omega.tolist(),
                'intensity':self.intensity.tolist(), 'spectral_phase':self.phase.tolist(),
                **pulse_metadata(self.options)}


class PropagatedPulse:
    """Apply an additional serial input propagator to a pulse spectrum."""
    def __init__(self, pulse, propagator):
        if not isinstance(pulse, (GaussPulse, SechPulse, DataPulse, PropagatedPulse)):
            raise TypeError('pulse must be a supported pulse object')
        if not callable(propagator):
            raise TypeError('propagator must be callable')
        self.pulse, self.propagator = pulse, propagator

    def spectrum(self, grid):
        return _apply_propagator(self.pulse.spectrum(grid), grid, self.propagator)

    def metadata(self):
        return {'type':'PropagatedPulse', 'pulse':pulse_metadata(self.pulse),
                'propagator':pulse_metadata(self.propagator)}


def prepare_inputs(grid, kw):
    pulses = kw['pulses']
    if pulses is None:
        return _apply_propagator(_initial(grid, kw), grid, kw['propagator'])
    if kw['pulse'] is not None:
        raise ValueError('pulse and pulses cannot be supplied together')
    if not isinstance(pulses, (list, tuple)):
        pulses = (pulses,)
    if not pulses:
        raise ValueError('pulses must not be empty')
    field = np.zeros_like(grid.omega, dtype=complex)
    for pulse in pulses:
        if not isinstance(pulse, (GaussPulse, SechPulse, DataPulse, PropagatedPulse)):
            raise TypeError('pulses must contain supported pulse objects')
        from .modal_inputs import pulse_options, mode_spec
        options=pulse_options(pulse)
        if options['polarisation'] not in ('linear','x'):
            raise ValueError('this scalar propagation requires linear or x pulse polarisation')
        selected=options['mode']
        if not (isinstance(selected,str) and selected=='lowest'):
            if 'modes' not in kw or mode_spec(selected)!=mode_spec(kw['modes']):
                raise ValueError('pulse mode was not found in the scalar simulation')
        field += pulse.spectrum(grid)
    return field
