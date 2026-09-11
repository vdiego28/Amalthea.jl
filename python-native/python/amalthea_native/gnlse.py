"""Internal GNLSE evaluator using Julia's field and response conventions."""
from dataclasses import dataclass
import math

import numpy as np

from .grid import C, EnvGrid, RealGrid, _positive, planck_taper
from .envelope import EPS0, NLSCALE, _EnvelopeModel, solve_envelope_model
from .raman import _sio2_response
from .pulses import (_ALIASES, _vector, _initial, energy_t, prepare_inputs, pulse_metadata)
from .callbacks import _Responses

_DEFAULTS = dict(delta_t=1., tau_fwhm=None, tau_w=None, phi=(), power=None,
                 energy=None, pulseshape='gauss', pulse=None, pulse_domain='time', pulses=None, propagator=None,
                 shotnoise=False, shock=True, loss=0., raman=True, fr=.18,
                 ramanmodel='sdo', tau1=12.2e-15, tau2=32e-15, saveN=201,
                 backend='auto', init_dz=1e-4, min_dz=1e-15, max_dz=None,
                 rtol=1e-6, atol=1e-10, safety=.9, locextrap=True,
                 max_attempts=1_000_000, repeat_limit=10, responses=None)


def _keywords(kwargs):
    supplied = {}
    for key, value in kwargs.items():
        canonical = _ALIASES.get(key, key)
        if canonical in supplied:
            raise TypeError(f'duplicate keyword alias for {canonical}')
        if canonical not in _DEFAULTS and canonical not in ('lambda0', 'lambda_lims', 'trange'):
            raise TypeError(f'unsupported GNLSE keyword: {key}')
        supplied[canonical] = value
    for key in ('lambda0', 'lambda_lims', 'trange'):
        if key not in supplied:
            raise TypeError(f'missing required keyword: {key}')
    return _DEFAULTS | supplied




@dataclass
class PropagationResult:
    field: np.ndarray
    z: np.ndarray
    grid: EnvGrid | RealGrid
    parameters: dict
    metadata: dict

    @property
    def Eomega(self):
        return self.field

    @property
    def Eω(self):
        return self.field

    def temporal_field(self):
        if self.grid.is_real:
            return np.fft.irfft(self.field, n=self.grid.t.size, axis=0)
        return np.fft.ifft(self.field, axis=0)

    def save_npz(self, path):
        """Write owned numeric arrays and JSON metadata; loading needs no pickle."""
        from .output import save_npz
        save_npz(self, path)

    def save_hdf5(self, path):
        """Write lossless HDF5 arrays and UTF-8 JSON; requires the hdf5 extra.

        Datasets retain frequency/(mode)/saved-position axes. Existing files
        are overwritten, as with save_npz. Read JSON datasets with asstr()[()].
        """
        from .output import save_hdf5
        save_hdf5(self, path)


class _Gnlse(_EnvelopeModel):
    def __init__(self, gamma, flength, betas, **kwargs):
        kw = _keywords(kwargs)
        self.kw = kw
        self.amplitude_scale = NLSCALE
        gamma = float(gamma)
        if not math.isfinite(gamma):
            raise ValueError('gamma must be finite')
        betas = _vector(betas, 'betas')
        if not betas.size:
            raise ValueError('betas must not be empty')
        for flag in ('shotnoise', 'shock', 'raman'):
            if not isinstance(kw[flag], (bool, np.bool_)):
                raise TypeError(f'{flag} must be a boolean')
        if kw['shotnoise']:
            raise NotImplementedError('quantum noise is disabled in this release')
        if kw['backend'] not in ('auto', 'python', 'native'):
            raise ValueError('backend must be auto, python, or native')
        self.custom_responses=_Responses(kw['responses'])
        self.native_eligible=not self.custom_responses
        self.backend_reason='custom nonlinear responses use Python evaluation'
        if self.custom_responses and kw['backend']=='native':
            raise NotImplementedError(self.backend_reason+'; use backend=auto or python')
        if kw['raman'] and kw['ramanmodel'] not in ('sdo', 'SiO2'):
            raise ValueError('ramanmodel must be sdo or SiO2')
        fraction = float(kw['fr'])
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('fr must be in [0, 1]')
        loss = float(kw['loss'])
        if not math.isfinite(loss):
            raise ValueError('loss must be finite')
        self.grid = grid = EnvGrid(flength, kw['lambda0'], kw['lambda_lims'], kw['trange'],
                                   delta_t=kw['delta_t'])
        offset = grid.omega - grid.omega0
        beta = np.polynomial.polynomial.polyval(offset, [v / math.factorial(i) for i, v in enumerate(betas)])
        beta1 = betas[1] if betas.size > 1 else 0.
        self.linop = -1j * (beta - beta1 * offset - betas[0]) - np.clip(math.log(10) / 10 * loss, 0, 3000) / 2
        self.linop[~grid.sidx] = 0
        self.initial = prepare_inputs(grid, kw)
        n2 = gamma / (2 * math.pi / grid.reference_lambda)
        chi3 = 4 / 3 * (1 - fraction) * n2 * (EPS0 * C)
        self.kerr = 3 / 4 * EPS0 * chi3
        self.n, self.no = grid.t.size, grid.to.size
        shockterm = grid.omega**2 if kw['shock'] else grid.omega * grid.omega0
        self.pre = np.ones(self.n, dtype=complex)
        selected = grid.sidx
        self.pre[selected] = (-1j * shockterm[selected] / (2 * C**1.5 * math.sqrt(2 * EPS0))
                              / (grid.omega[selected] / C)) * grid.omega_win[selected]
        self.h = None
        if kw['raman']:
            dt = grid.to[1] - grid.to[0]
            t = np.arange(self.no) * dt
            chi3r = 2 * fraction * n2 * (EPS0 * C)
            if kw['ramanmodel'] == 'SiO2':
                h = _sio2_response(t, chi3r * EPS0)
            else:
                tau1, tau2 = _positive('tau1', kw['tau1']), _positive('tau2', kw['tau2'])
                h = chi3r * EPS0 * ((1 / tau1)**2 + 1 / tau2**2) / (1 / tau1) * np.sin(t / tau1) * np.exp(-t / tau2)
                h *= planck_taper(t, -t[-1], -.7 * t[-1], .7 * t[-1], t[-1])
            self.h = h
            self.hfft = np.fft.rfft(np.pad(h, (0, self.no))) * dt
        self.parameters = {'gamma': gamma, 'flength': grid.zmax, 'betas': betas.tolist()}
        for key, value in kw.items():
            if key=='responses':
                self.parameters[key]=self.custom_responses.metadata()
            elif key in ('pulses', 'propagator'):
                self.parameters[key] = pulse_metadata(value)
            elif key == 'pulse':
                self.parameters[key] = None if value is None else 'supplied grid-matched array'
            elif isinstance(value, (np.ndarray, tuple, list)):
                self.parameters[key] = np.asarray(value).tolist()
            elif isinstance(value, np.generic):
                self.parameters[key] = value.item()
            else:
                self.parameters[key] = value



def prop_gnlse(gamma, flength, betas, **kwargs):
    """Propagate an envelope (SI units); beta coefficients start at orders 0, 1.

    Required keywords: lambda0, lambda_lims, trange; Gaussian/sech pulses also
    require duration and energy or power. Unicode aliases follow Julia. An
    optional pulse array must already match the grid (time or frequency domain).
    Kerr/shock and SDO/SiO2 Raman support Python evaluation or resident Rust
    evaluation with portable FFTs. Quantum noise is disabled. responses accepts
    callbacks (complete time/component field, ResponseContext) returning physical
    polarization; custom responses select Python evaluation with Rust stepping.
    """
    model = _Gnlse(gamma, flength, betas, **kwargs)
    result = solve_envelope_model(model)
    return PropagationResult(result.field, result.z, model.grid, model.parameters, result.metadata)
