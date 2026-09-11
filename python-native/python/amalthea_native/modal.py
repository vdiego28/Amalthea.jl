"""Modal capillaries: serial spatial/temporal evaluation and Rust stepping."""
import math

import numpy as np
from scipy.interpolate import splev

from . import materials
from .grid import C, EnvGrid, RealGrid
from .modes import _scalar, _silica_splines
from .modal_inputs import build_modes, both_polarisations, prepare_modal_inputs, mode_metadata
from .profiles import _GasProfile, _MixtureProfile
from .pulses import pulse_metadata
from .responses import _ScalarGasResponse, _VectorGasResponse, species_options
from .spatial import _ModeSpace, _POWER_FACTOR
from .callbacks import _Responses


class _ModalCapillary:
    native_eligible=False
    backend_reason='serial SciPy modal quadrature with Python point evaluation'

    def __init__(self,radius,flength,gas,pressure,**kwargs):
        from .capillary import _keywords
        self.kw=kw=_keywords(kwargs)
        if kw['backend'] not in ('auto','native','python'):raise ValueError('backend must be auto, native or python')
        if kw['backend']=='native':raise NotImplementedError(self.backend_reason+'; use backend=auto or python')
        self.custom_responses=_Responses(kw['responses'])
        for key in ('envelope','shotnoise','kerr','loss','rotation','vibration'):
            if not isinstance(kw[key],(bool,np.bool_)):raise TypeError(f'{key} must be a boolean')
        for key in ('raman','thg','modal_full'):
            if kw[key] is not None and not isinstance(kw[key],(bool,np.bool_)):raise TypeError(f'{key} must be a boolean or None')
        if kw['shotnoise']:raise NotImplementedError('quantum noise is disabled in this release')
        self.is_mixture=not isinstance(gas,str)
        if self.is_mixture:
            if (not isinstance(gas,(list,tuple,np.ndarray)) or isinstance(gas,np.ndarray) and gas.ndim!=1 or len(gas)==0):
                raise ValueError('gas must be a name or a nonempty sequence of names')
            self.gases=tuple(materials._gas(name) for name in gas)
        else:self.gases=(materials._gas(gas),)
        real=not kw['envelope'];self.thg=real if kw['thg'] is None else kw['thg']
        selected=species_options(self.gases,kw,kw['species_options'],real)
        self.grid=grid=(RealGrid if real else EnvGrid)(flength,kw['lambda0'],kw['lambda_lims'],kw['trange'],
                                                      delta_t=kw['delta_t'],**({} if real else dict(thg=self.thg)))
        self.n,self.no=len(grid.omega),len(grid.to);self.omega0=2*math.pi*C/grid.reference_lambda
        self.profile=profile=(_MixtureProfile(self.gases,pressure,grid.zmax,kw['temperature']) if self.is_mixture
                             else _GasProfile(self.gases[0],pressure,grid.zmax,kw['temperature']))
        both=both_polarisations(kw)
        self.modes,constructed=build_modes(kw['modes'],radius,profile,gas,kw,both)
        from .modes import MarcatiliMode
        radial=all(type(mode) is MarcatiliMode and mode.kind=='HE' and mode.n==1 for mode in self.modes)
        components=kw['modal_components'] or ('xy' if both or not radial else 'y')
        full=not radial if kw['modal_full'] is None else kw['modal_full']
        self.space=_ModeSpace(self.modes,components=components,full=full)
        self.npol=len(self.space.indices)
        response_type=_VectorGasResponse if self.npol==2 else _ScalarGasResponse
        self.responses=tuple(response_type(grid,name,options|dict(temperature=kw['temperature']),self.thg)
                             for name,options in zip(self.gases,selected))
        self.variable=constructed or callable(radius) or profile.kind!='constant'
        self.quadrature=dict(rtol=kw['radial_integral_rtol'],atol=kw['modal_atol'],
                             maxevals=kw['modal_maxevals'],rule=kw['modal_rule'])
        # Orthonormality is a physical x/y power integral, regardless of chosen RHS reduction.
        if len(self.modes)>1:
            identity=np.eye(len(self.modes),dtype=complex)
            overlap=_ModeSpace(self.modes,components='xy',full=True).at(0.).project(identity,lambda f,p:f,rtol=1e-10)
            error=np.hypot(overlap.error[...,0],overlap.error[...,1])*_POWER_FACTOR
            if np.any(np.abs(overlap.value*_POWER_FACTOR-identity)>np.maximum(error,32*np.finfo(float).eps)):
                raise ValueError('selected modes do not form a power-orthonormal set')
        self.initial=prepare_modal_inputs(grid,kw,self.modes)
        self.linop=self.linear_operator if self.variable else self.constant_linear_operator()
        self.pre=-1j*grid.omega/4*grid.omega_win
        self.evaluations=0;self.rhs_calls=0;self.max_error_ratio=0.;self.last_integral=None
        self._refresh(0.)
        self.native_points=None
        if (kw['backend']=='auto' and not self.variable and not self.custom_responses
                and self.thg==grid.is_real and not any(response.plasma is not None for response in self.responses)):
            from ._point_native import _NativePoints
            self.native_points=_NativePoints(self)
            self.backend_reason='serial SciPy modal quadrature with Rust point evaluation'
        self.parameters=dict(radius=pulse_metadata(radius),flength=grid.zmax,
                             gas=pulse_metadata(self.gases if self.is_mixture else self.gases[0]),pressure=pulse_metadata(pressure),
                             **{key:pulse_metadata(value) for key,value in kw.items() if key not in ('modes','pulse','responses')})
        self.parameters['responses']=self.custom_responses.metadata()
        self.parameters['modes']=[mode_metadata(mode) for mode in self.modes]
        self.parameters['pulse']=None if kw['pulse'] is None else 'supplied grid-matched modal array'
        self.parameters['species_options']=pulse_metadata(selected)
        self.parameters['modal_components']=components;self.parameters['modal_full']=bool(full)

    def _refresh(self,z):
        densities=self.profile.density(z)
        if not self.is_mixture:densities=(densities,)
        for response,density in zip(self.responses,densities):response.refresh(density)

    @staticmethod
    def _evaluate(method,omega,z,*,real=False):
        supplied=np.asarray(method(np.array(omega,copy=True),z=z))
        if supplied.shape not in ((),np.shape(omega)) or not np.all(np.isfinite(supplied)):
            raise ValueError('mode spectral method must return finite values with the frequency shape')
        if real and np.iscomplexobj(supplied):raise ValueError('beta and alpha must be real')
        return np.array(np.broadcast_to(supplied,np.shape(omega)),copy=True)

    def _frame(self,z):
        reference=self.modes[0]
        beta1=_scalar(reference.dispersion(1,self.omega0,z=z),'mode beta1')
        beta0=_scalar(reference.beta(self.omega0,z=z),'mode beta0')
        return beta1,beta0

    def constant_linear_operator(self):
        grid=self.grid;selected=grid.sidx;beta1,beta0=self._frame(0.)
        output=np.empty((self.n,len(self.modes)),dtype=complex)
        for i,mode in enumerate(self.modes):
            beta=np.ones(self.n);beta[selected]=self._evaluate(mode.beta,grid.omega[selected],0.,real=True)
            alpha=np.zeros(self.n)
            if grid.is_real:alpha[selected]=self._evaluate(mode.alpha,grid.omega[selected],0.,real=True)
            elif mode.loss:
                # Julia evaluates envelope loss at all signed grid frequencies.
                omega=grid.omega
                if np.any(omega==0):raise ValueError('zero-frequency envelope modal loss is undefined')
                real,imag=_silica_splines();wavelength=2*math.pi*C/omega
                cladding=splev(wavelength*1e6,real)+1j*splev(wavelength*1e6,imag)
                core=self.profile.core_index(np.abs(omega),z=0.)
                index=mode._neff_from_indices(omega,core,cladding,mode.radius(0.))
                alpha=2*omega/C*np.imag(index)
            if grid.is_real:output[:,i]=1j*(-beta+grid.omega*beta1)-np.clip(alpha,0,3000)/2
            else:output[:,i]=-1j*(beta-(grid.omega-grid.omega0)*beta1-beta0)-np.clip(alpha,0,3000)/2
        if not np.all(np.isfinite(output)):raise ValueError('nonfinite modal linear operator')
        return output

    def linear_operator(self,z):
        grid=self.grid;omega=grid.omega[grid.sidx];beta1,beta0=self._frame(z)
        output=np.zeros((self.n,len(self.modes)),dtype=complex)
        for i,mode in enumerate(self.modes):
            index=self._evaluate(mode.neff,omega,z)
            nc=np.maximum(index.real,1e-3)-1j*np.clip(index.imag,0,3000*C/omega)
            frame=beta1*(omega if grid.is_real else omega-grid.omega0)
            output[grid.sidx,i]=-1j*(omega/C*nc-frame)
            if not grid.is_real:output[grid.sidx,i]-=-1j*beta0
        return output

    def to_time(self,spectrum):
        count=len(spectrum);grid=self.grid;scale=self.no/len(grid.t)
        if grid.is_real:
            padded=np.zeros((count,self.no//2+1,self.npol),dtype=complex)
            padded[:,:self.n]=spectrum*scale
            return np.fft.irfft(padded,n=self.no,axis=1)
        half=self.n//2;padded=np.zeros((count,self.no,self.npol),dtype=complex)
        padded[:,:half]=spectrum[:,:half]*scale;padded[:,-half:]=spectrum[:,-half:]*scale
        return np.fft.ifft(padded,axis=1)

    def polarization(self,time,*,z=0.,points=None,coordinate_system=None):
        output=np.zeros_like(time)
        for i,field in enumerate(time):
            for response in self.responses:
                if self.npol==1:response.accumulate(output[i,:,0],field[:,0])
                else:response.accumulate(output[i],field)
            if self.custom_responses:
                if points is None or coordinate_system is None:
                    raise ValueError('custom modal responses require spatial coordinates and their system')
                components=tuple('xy'[index] for index in self.space.indices)
                self.custom_responses.accumulate(output[i],field,z=z,grid=self.grid,gases=self.gases,
                    densities=[response.density for response in self.responses],
                    coordinate_system=coordinate_system,coordinates=points[i],components=components)
        return output

    def point_response(self,spectrum,points=None,*,z=0.,coordinate_system=None):
        if self.native_points is not None:
            return self.native_points(spectrum)
        grid=self.grid;time=self.to_time(spectrum)
        polarization=self.polarization(time,z=z,points=points,coordinate_system=coordinate_system)*grid.towin[None,:,None]
        if grid.is_real:out=np.fft.rfft(polarization,axis=1)[:,:self.n]
        else:
            transformed=np.fft.fft(polarization,axis=1);half=self.n//2
            out=np.concatenate((transformed[:,:half],transformed[:,-half:]),axis=1)
        return out*(len(grid.t)/self.no)*self.pre[None,:,None]

    def rhs(self,z,field):
        self._refresh(z)
        geometry=self.space.at(z)
        result=geometry.project(field,lambda spectrum,points:self.point_response(spectrum,points,z=z,
                                               coordinate_system=geometry.kind),**self.quadrature)
        self.last_integral=result;self.evaluations+=result.evaluations;self.rhs_calls+=1
        if result.tolerance:self.max_error_ratio=max(self.max_error_ratio,result.error_norm/result.tolerance)
        return result.value

    def window(self,z,field):
        grid=self.grid;windowed=field*grid.omega_win[:,None]
        if grid.is_real:
            return np.fft.rfft(np.fft.irfft(windowed,n=len(grid.t),axis=0)*grid.twin[:,None],axis=0)
        return np.fft.fft(np.fft.ifft(windowed,axis=0)*grid.twin[:,None],axis=0)

    def execution_metadata(self):
        return dict(quadrature='scipy',quadrature_error_norm='global-l2',
                    point_evaluator='rust' if self.native_points is not None else 'python',
                    point_fft='rustfft/realfft' if self.native_points is not None else 'numpy',
                    quadrature_evaluations=self.evaluations,quadrature_rhs_calls=self.rhs_calls,
                    max_quadrature_error_ratio=self.max_error_ratio)
