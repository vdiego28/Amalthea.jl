"""Internal capillary propagation; full carrier/modal coverage is in development."""
import math

import numpy as np

from . import materials
from .envelope import NLSCALE, _EnvelopeModel, solve_envelope_model
from .gnlse import PropagationResult
from .grid import C, EnvGrid, RealGrid
from .modes import MarcatiliMode
from .modal_inputs import mode_spec, needs_modal
from .pulses import _ALIASES, prepare_inputs, pulse_metadata
from .profiles import _GasProfile, _MixtureProfile
from .responses import _ScalarGasResponse, species_options
from .callbacks import _Responses

_DEFAULTS = dict(delta_t=1., tau_fwhm=None, tau_w=None, phi=(), power=None,
                 energy=None, pulseshape='gauss', pulse=None, pulse_domain='time',
                 pulses=None, propagator=None, shotnoise=False, envelope=False,
                 thg=None, plasma=None, PPT_options=None, preionfrac=0., raman=None, kerr=True, loss=True,
                 rotation=True, vibration=True, species_options=None,
                 modes='HE11', model='full', temperature=materials.ROOMTEMP,
                 polarisation='linear', saveN=201, backend='auto', init_dz=1e-4,
                 radial_integral_rtol=1e-3, modal_atol=0., modal_maxevals=100_000,
                 modal_rule='gk21', modal_full=None, modal_components=None,
                 min_dz=1e-15, max_dz=None, rtol=1e-6, atol=1e-10, safety=.9,
                 locextrap=True, max_attempts=1_000_000, repeat_limit=10, responses=None)


def _keywords(kwargs):
    supplied={}
    for key,value in kwargs.items():
        key={'polarization':'polarisation','T':'temperature','ppt_options':'PPT_options'}.get(key,_ALIASES.get(key,key))
        if key in supplied:
            raise TypeError(f'duplicate capillary keyword alias: {key}')
        if key not in _DEFAULTS and key not in ('lambda0','lambda_lims','trange'):
            raise TypeError(f'unsupported capillary keyword: {key}')
        supplied[key]=value
    for key in ('lambda0','lambda_lims','trange'):
        if key not in supplied:
            raise TypeError(f'missing required keyword: {key}')
    return _DEFAULTS|supplied


class _Capillary(_EnvelopeModel):
    def __init__(self,radius,flength,gas,pressure,**kwargs):
        self.kw=kw=_keywords(kwargs)
        for flag in ('envelope','shotnoise','kerr','loss','rotation','vibration'):
            if not isinstance(kw[flag],(bool,np.bool_)):
                raise TypeError(f'{flag} must be a boolean')
        for flag in ('raman','thg'):
            if kw[flag] is not None and not isinstance(kw[flag],(bool,np.bool_)):
                raise TypeError(f'{flag} must be a boolean or None')
        if kw['shotnoise']:
            raise NotImplementedError('quantum noise is disabled in this release')
        if kw['backend'] not in ('auto','python','native'):
            raise ValueError('backend must be auto, python, or native')
        self.custom_responses=_Responses(kw['responses'])
        if self.custom_responses and kw['backend']=='native':
            raise NotImplementedError('custom nonlinear responses require Python evaluation; use backend=auto or python')
        real=not kw['envelope']
        self.thg=real if kw['thg'] is None else kw['thg']
        self.is_mixture=not isinstance(gas,str)
        if self.is_mixture:
            if (not isinstance(gas,(list,tuple,np.ndarray))
                    or isinstance(gas,np.ndarray) and gas.ndim!=1 or len(gas)==0):
                raise ValueError('gas must be a name or a nonempty sequence of gas names')
            self.gases=tuple(materials._gas(item) for item in gas)
        else:
            self.gases=(materials._gas(gas),)
        options_by_species=species_options(self.gases,kw,kw['species_options'],real)
        if not self.is_mixture:
            kw['PPT_options']=options_by_species[0]['PPT_options']
        any_kerr=any(item['kerr'] for item in options_by_species)
        any_raman=any(item['raman'] for item in options_by_species)
        any_plasma=any(bool(item['plasma']) for item in options_by_species)
        self.native_eligible=True
        self.backend_reason='resident CPU scalar response'
        if self.custom_responses:
            self.native_eligible=False
            self.backend_reason='custom nonlinear responses use Python evaluation'
        if real and not self.thg and any_kerr:
            self.native_eligible=False
            self.backend_reason='carrier Kerr with thg=False uses Python analytic intensity'
        if not real and self.thg and any_kerr:
            self.native_eligible=False
            self.backend_reason='envelope THG Kerr uses Python evaluation'
        if real and any_raman and not self.thg:
            self.native_eligible=False
            self.backend_reason='carrier Raman with thg=False uses Python analytic intensity'
        if self.is_mixture and (any_raman or any_plasma):
            self.native_eligible=False
            self.backend_reason='gas mixtures with Raman or plasma use Python evaluation'
        if not self.native_eligible and kw['backend']=='native':
            raise NotImplementedError(self.backend_reason+'; use backend=python or auto')
        if kw['pulses'] is None and kw['polarisation']!='linear':
            raise NotImplementedError('polarized capillary propagation is not implemented yet')
        specification=mode_spec(kw['modes'])
        grid_options={} if real else {'thg':self.thg}
        self.grid=grid=(RealGrid if real else EnvGrid)(flength,kw['lambda0'],kw['lambda_lims'],kw['trange'],
                                                      delta_t=kw['delta_t'],**grid_options)
        self.profile=profile=(_MixtureProfile(self.gases,pressure,grid.zmax,kw['temperature']) if self.is_mixture
                              else _GasProfile(self.gases[0],pressure,grid.zmax,kw['temperature']))
        self.variable=callable(radius) or profile.kind!='constant'
        if self.variable:
            self.native_eligible=False
            self.backend_reason='position-dependent capillary profiles use Python evaluation'
            if kw['backend']=='native':
                raise NotImplementedError(self.backend_reason+'; use backend=python or auto')
        if self.is_mixture or profile.kind!='constant':
            mode=MarcatiliMode(radius,core_index=profile.core_index,**specification,
                               model=kw['model'],loss=kw['loss'],temperature=kw['temperature'])
        else:
            mode=MarcatiliMode(radius,self.gases[0],profile.constant,**specification,
                               model=kw['model'],loss=kw['loss'],temperature=kw['temperature'])
        self.mode=mode
        self.responses=tuple(_ScalarGasResponse(grid,name,options|{'temperature':kw['temperature']},self.thg)
                             for name,options in zip(self.gases,options_by_species))
        self._refresh_responses(0.)
        self.n,self.no=grid.omega.size,grid.to.size
        self.initial=prepare_inputs(grid,kw)
        self._native_plasma=None
        if not self.is_mixture and self.plasma is not None:
            self._native_plasma,reason=self.plasma.native_config(self.density)
            if self._native_plasma is None:
                self.native_eligible=False;self.backend_reason=reason
        if not self.native_eligible and kw['backend']=='native':
            raise NotImplementedError(self.backend_reason+'; use backend=python or auto')
        if self.variable:
            self.backend_reason='position-dependent capillary profiles use Python evaluation'
        self.area=mode.Aeff()
        self.amplitude_scale=NLSCALE*math.sqrt(self.area)
        self.beta=np.ones(self.n)
        self.alpha=np.zeros(self.n)
        selected=grid.sidx
        self.beta[selected]=mode.beta(grid.omega[selected])
        self.alpha[selected]=mode.alpha(grid.omega[selected])
        self.omega0=omega0=2*math.pi*C/grid.reference_lambda
        self.beta1=mode.dispersion(1,omega0)
        self.beta0=mode.beta(omega0)
        phase=self.beta-self.beta1*grid.omega if real else self.beta-self.beta1*(grid.omega-omega0)-self.beta0
        self.linop=-1j*phase-np.clip(self.alpha,0,3000)/2
        self.linop[~selected]=0
        self.pre=np.ones(self.n,dtype=complex)
        self.pre[selected]=(-1j*grid.omega[selected]**2/4/NLSCALE/C/self.beta[selected]
                            *math.sqrt(self.area)*grid.omega_win[selected])
        if self.variable:
            self.linop=self.linear_operator
        self.parameters=dict(radius=pulse_metadata(radius),flength=grid.zmax,gas=pulse_metadata(self.gases if self.is_mixture else self.gases[0]),
                             pressure=pulse_metadata(pressure),
                             **{key:pulse_metadata(value) for key,value in kw.items() if key not in ('pulse','responses')})
        self.parameters['responses']=self.custom_responses.metadata()
        if self.is_mixture or kw['species_options'] is not None:
            self.parameters['species_options']=pulse_metadata(options_by_species)
        self.parameters['pulse']=None if kw['pulse'] is None else 'supplied grid-matched array'

    def rhs(self,z,field):
        if self.variable:
            self._refresh(z)
        if not self.grid.is_real:
            return super().rhs(z,field)
        scale=self.no/self.grid.t.size
        spectrum=np.zeros(self.no//2+1,dtype=complex)
        spectrum[:self.n]=field*scale
        time=np.fft.irfft(spectrum,n=self.no)/self.amplitude_scale
        polarization=self.polarization(time,z=z)
        return np.fft.rfft(polarization*self.grid.towin)[:self.n]/scale*self.pre

    def linear_operator(self,z):
        grid=self.grid
        omega=grid.omega[grid.sidx]
        neff=self.mode.neff(omega,z=z)
        nc=np.maximum(np.real(neff),1e-3)-1j*np.clip(np.imag(neff),0,3000*C/omega)
        beta1=self.mode.dispersion(1,self.omega0,z=z)
        frame=beta1*(omega if grid.is_real else omega-self.omega0)
        output=np.zeros(self.n,dtype=complex)
        output[grid.sidx]=-1j*(omega/C*nc-frame)
        if not grid.is_real:
            output[grid.sidx]-=-1j*self.mode.beta(self.omega0,z=z)
        return output

    def _refresh_responses(self,z):
        densities=self.profile.density(z)
        if not self.is_mixture:
            densities=(densities,)
        for response,density in zip(self.responses,densities):
            response.refresh(density)

    @property
    def density(self):
        if not self.is_mixture:
            return self.responses[0].density
        return np.array([response.density for response in self.responses])

    @property
    def kerr(self):
        return sum(response.kerr for response in self.responses)

    @property
    def plasma(self):
        return None if self.is_mixture else self.responses[0].plasma

    @property
    def h(self):
        return None if self.is_mixture else self.responses[0].h

    @property
    def hfft(self):
        return None if self.is_mixture else self.responses[0].hfft

    @property
    def raman_response(self):
        return None if self.is_mixture else self.responses[0].raman_response

    def _refresh(self,z):
        self._refresh_responses(z)
        self.area=self.mode.Aeff(z=z)
        self.amplitude_scale=NLSCALE*math.sqrt(self.area)
        selected=self.grid.sidx
        omega=self.grid.omega[selected]
        self.beta[selected]=self.mode.beta(omega,z=z)
        self.pre[selected]=(-1j*omega**2/4/NLSCALE/C/self.beta[selected]
                            *math.sqrt(self.area)*self.grid.omega_win[selected])

    def polarization(self,time,*,z=0.):
        polarization=np.zeros_like(time)
        for response in self.responses:
            response.accumulate(polarization,time)
        if self.custom_responses:
            self.custom_responses.accumulate(polarization,time,z=z,grid=self.grid,gases=self.gases,
                                             densities=[response.density for response in self.responses])
        return polarization

    def window(self,z,field):
        if not self.grid.is_real:
            return super().window(z,field)
        return np.fft.rfft(np.fft.irfft(field*self.grid.omega_win,n=self.grid.t.size)*self.grid.twin)


def prop_capillary(radius,flength,gas,pressure,**kwargs):
    """Propagate scalar or modal capillary fields with Kerr, Raman, plasma and loss.

    ADK/PPT tables can run natively; envelopes infer plasma off.
    Radius/length are metres, pressure is bar, pulse energies joules. Native
    and Python evaluation share the Rust adaptive solver. Callable radius and
    pressure, (p0,p1) gradients and (positions,pressures) gradients use Python
    evaluation at requested positions. A gas sequence requires matching partial
    pressure specifications. species_options supplies per-species response
    overrides. Constant Kerr mixtures can run natively; Raman/plasma mixtures
    use Python evaluation. Mode counts/collections and polarization pairs use
    serial SciPy quadrature with Python point evaluation and Rust stepping;
    modal fields have (frequency, mode, saved position) axes. Constructed modes
    provide neff, field and dimlimits, with optional normalization/dispersion.
    Vector Raman and envelope plasma are excluded; carrier vector Kerr needs THG.
    responses appends complete-array callbacks (field, ResponseContext) returning
    physical polarization. Custom responses run serially through Python.
    """
    kw=_keywords(kwargs)
    if needs_modal(kw):
        from .modal import _ModalCapillary
        model=_ModalCapillary(radius,flength,gas,pressure,**kw)
    else:model=_Capillary(radius,flength,gas,pressure,**kw)
    result=solve_envelope_model(model)
    if hasattr(model,'execution_metadata'):result.metadata.update(model.execution_metadata())
    return PropagationResult(result.field,result.z,model.grid,model.parameters,result.metadata)
