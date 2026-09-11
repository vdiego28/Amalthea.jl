"""Mode selection and pulse assignment using Interface.jl's ordering."""
import copy
import math
import operator
import re
from collections.abc import Mapping

import numpy as np

from .modes import Mode, MarcatiliMode
from .pulses import (GaussPulse, SechPulse, DataPulse, PropagatedPulse, _initial,
                     _apply_propagator, _intensity, _simpson, pulse_metadata)
from .grid import _positive


def mode_spec(value):
    if isinstance(value, str):
        match=re.fullmatch(r'(HE|TE|TM)([0-9])([1-9])',value)
        if match is None:
            raise ValueError('mode signifiers require kind and two single-digit indices; use a mapping for larger indices')
        kind,n,m=match.groups()
        return mode_spec(dict(kind=kind,n=int(n),m=int(m)))
    if not isinstance(value,Mapping):
        raise TypeError('mode must be a signifier, mapping or constructed mode')
    out={}
    for key,item in value.items():
        key={'ϕ':'phi','φ':'phi'}.get(key,key)
        if key not in ('kind','n','m','phi'):raise TypeError(f'unsupported mode specification: {key}')
        if key in out:raise TypeError(f'duplicate mode specification: {key}')
        out[key]=item
    if not all(key in out for key in ('kind','n','m')):
        raise ValueError('mode mapping requires kind, n and m')
    out['n'],out['m']=operator.index(out['n']),operator.index(out['m'])
    if out['kind'] not in ('HE','TE','TM') or out['m']<1 or (out['n']<1 if out['kind']=='HE' else out['n']!=0):
        raise ValueError('HE requires n>=1; TE/TM require n=0; radial order m>=1')
    return out


def vector_polarisation(value):
    if isinstance(value,str):
        if value not in ('linear','x','y','circular'):raise ValueError('unknown polarisation')
        return value!='linear'
    supplied=np.asarray(value)
    if supplied.ndim or np.iscomplexobj(supplied) or isinstance(value,(bool,np.bool_)):
        raise ValueError('polarisation must be linear, x, y, circular, or a real ellipticity')
    scalar=float(supplied)
    if not math.isfinite(scalar) or not -1<=scalar<=1:raise ValueError('ellipticity must be in [-1,1]')
    return True


def pulse_options(pulse):
    while isinstance(pulse,PropagatedPulse):pulse=pulse.pulse
    if not isinstance(pulse,(GaussPulse,SechPulse,DataPulse)):
        raise TypeError('pulses must contain supported pulse objects')
    return pulse.options


def both_polarisations(kw):
    pulses=kw['pulses']
    if pulses is None:return vector_polarisation(kw['polarisation'])
    pulses=pulses if isinstance(pulses,(tuple,list)) else [pulses]
    return any(vector_polarisation(pulse_options(p)['polarisation']) for p in pulses)


def needs_modal(kw):
    return (not isinstance(kw['modes'],str) or both_polarisations(kw)
            or kw['modal_full'] is not None or kw['modal_components'] is not None)


class _ModeAdapter(Mode):
    def __init__(self,mode):self.mode=mode
    def neff(self,omega,*,z=0.):return self.mode.neff(omega,z=z)
    def field(self,xy,*,z=0.):return self.mode.field(xy,z=z)
    def dimlimits(self,*,z=0.):return self.mode.dimlimits(z=z)
    def N(self,*,z=0.):
        method=getattr(self.mode,'N',None)
        return method(z=z) if method is not None else super().N(z=z)
    def beta(self,omega,*,z=0.):
        method=getattr(self.mode,'beta',None)
        return method(omega,z=z) if method is not None else super().beta(omega,z=z)
    def alpha(self,omega,*,z=0.):
        method=getattr(self.mode,'alpha',None)
        return method(omega,z=z) if method is not None else super().alpha(omega,z=z)
    def dispersion(self,order,omega,*,z=0.):
        method=getattr(self.mode,'dispersion',None)
        return method(order,omega,z=z) if method is not None else super().dispersion(order,omega,z=z)


def build_modes(selection,radius,profile,gas,kw,both):
    if isinstance(selection,(int,np.integer)) and not isinstance(selection,(bool,np.bool_)):
        if selection<1:raise ValueError('mode count must be positive')
        selected=[dict(kind='HE',n=1,m=m) for m in range(1,int(selection)+1)]
    elif isinstance(selection,(list,tuple,np.ndarray)):
        if isinstance(selection,np.ndarray) and selection.ndim!=1:raise ValueError('modes must be one-dimensional')
        selected=list(selection)
    else:selected=[selection]
    if not selected:raise ValueError('modes must not be empty')
    modes=[];constructed=False
    for item in selected:
        if not isinstance(item,(str,Mapping)):
            if not all(callable(getattr(item,name,None)) for name in ('neff','field','dimlimits')):
                raise TypeError('constructed modes must provide neff, field and dimlimits')
            modes.append(item if isinstance(item,Mode) else _ModeAdapter(item));constructed=True
            continue
        spec=mode_spec(item)
        if both and 'phi' in spec:raise ValueError('polarization pairs set their own mode orientation')
        orientations=([0.,math.pi/(2*operator.index(spec['n']))] if spec['kind']=='HE' else [0.]) if both else [spec.get('phi',0.)]
        for phi in orientations:
            options=dict(spec,phi=phi,model=kw['model'],loss=kw['loss'],temperature=kw['temperature'])
            if profile.kind=='constant' and isinstance(gas,str):
                mode=MarcatiliMode(radius,gas,profile.constant,**options)
            else:mode=MarcatiliMode(radius,core_index=profile.core_index,**options)
            modes.append(mode)
    return tuple(modes),constructed


def _indices(modes,selected,polarisation):
    if isinstance(selected,str) and selected=='lowest':
        indices=[0] if polarisation=='linear' else list(range(min(2,len(modes))))
    elif isinstance(selected,(int,np.integer)) and not isinstance(selected,(bool,np.bool_)):
        # Match Julia's low-level input mode indices (one-based).
        indices=[operator.index(selected)-1]
        if indices[0]<0 or indices[0]>=len(modes):raise ValueError('pulse mode index is out of range')
    else:
        spec=mode_spec(selected)
        indices=[i for i,m in enumerate(modes) if all(getattr(m,k,None)==spec[k] for k in ('kind','n','m'))]
    if not indices:raise ValueError('pulse mode was not found in the mode collection')
    if polarisation in ('linear','x'):return [(indices[0],1.,0.)]
    if len(indices)!=2:raise ValueError('polarisation requires exactly two selected modes')
    if polarisation=='y':return [(indices[1],1.,0.)]
    ellipticity=1. if polarisation=='circular' else float(polarisation)
    px=ellipticity**2/(1+ellipticity**2)
    return [(indices[0],1-px,0.),(indices[1],px,math.pi/2*np.sign(ellipticity))]


def _component(pulse,grid,factor,phase):
    if isinstance(pulse,PropagatedPulse):
        return _apply_propagator(_component(pulse.pulse,grid,factor,phase),grid,pulse.propagator)
    options=dict(pulse.options)
    coefficients=np.array(options['phi'],dtype=float,copy=True)
    if phase:
        if coefficients.size:coefficients[0]+=phase
        else:coefficients=np.array([phase])
    options['phi']=coefficients
    propagator=options.get('propagator');options['propagator']=None
    if isinstance(pulse,DataPulse):
        clone=copy.copy(pulse);clone.options=options;field=clone.spectrum(grid)
    elif isinstance(pulse,(GaussPulse,SechPulse)):
        field=_initial(grid,options|dict(pulse=None,pulse_domain='time',pulseshape=pulse.shape))
    else:field=_initial(grid,options)
    return _apply_propagator(field*math.sqrt(factor),grid,propagator)


def prepare_modal_inputs(grid,kw,modes):
    nmodes=len(modes);array=kw['pulse'];pulses=kw['pulses']
    if array is not None and pulses is not None:raise ValueError('pulse and pulses cannot be supplied together')
    if array is not None and np.ndim(array)==2:
        if kw['tau_fwhm'] is not None or kw['tau_w'] is not None:raise ValueError('pulse arrays cannot also specify a pulse duration')
        array=np.asarray(array)
        expected=(len(grid.omega) if kw['pulse_domain']=='frequency' else len(grid.t),nmodes)
        if array.shape!=expected:raise ValueError('modal pulse array must match (grid, modes)')
        fields=np.column_stack([_initial(grid,kw|dict(pulse=array[:,i],energy=None,power=None)) for i in range(nmodes)])
        energy,power=kw['energy'],kw['power']
        if energy is not None and power is not None:raise ValueError('specify only one of energy or power')
        if energy is not None or power is not None:
            time=np.fft.irfft(fields,n=len(grid.t),axis=0) if grid.is_real else np.fft.ifft(fields,axis=0)
            intensity=sum(_intensity(grid,time[:,i]) for i in range(nmodes))
            actual=_simpson(grid.t,intensity) if energy is not None else np.max(intensity)
            target=_positive('energy' if energy is not None else 'power',energy if energy is not None else power)
            if not np.isfinite(actual) or actual<=0:raise ValueError('cannot normalize a zero or nonfinite modal pulse')
            fields*=math.sqrt(target)/math.sqrt(actual)
        propagator=kw['propagator']
        if propagator is not None:
            if not callable(propagator):raise TypeError('propagator must be callable')
            incoming=fields.copy();supplied=propagator(incoming,copy.deepcopy(grid))
            fields=np.array(incoming if supplied is None else supplied,dtype=complex,copy=True)
            if fields.shape!=(len(grid.omega),nmodes):
                raise ValueError('modal propagator must preserve (frequency, modes)')
        if not np.all(np.isfinite(fields)):raise ValueError('nonfinite modal input')
        return fields
    if pulses is None:
        pulse=type('_Input',(),{})();pulse.options=kw|dict(mode='lowest');pulses=[pulse]
    else:
        pulses=pulses if isinstance(pulses,(tuple,list)) else [pulses]
        if not pulses:raise ValueError('pulses must not be empty')
    fields=np.zeros((len(grid.omega),nmodes),dtype=complex)
    for pulse in pulses:
        options=pulse_options(pulse) if kw['pulses'] is not None else pulse.options
        pol=options['polarisation'];vector_polarisation(pol)
        for index,factor,phase in _indices(modes,options['mode'],pol):
            fields[:,index]+=_component(pulse,grid,factor,phase)
    return fields


def mode_metadata(mode):
    if type(mode) is MarcatiliMode:
        return dict(type='MarcatiliMode',kind=mode.kind,n=mode.n,m=mode.m,phi=mode.phi,
                    radius=pulse_metadata(mode._radius),model=mode.model,loss=mode.loss)
    original=mode.mode if isinstance(mode,_ModeAdapter) else mode
    return dict(type=type(original).__qualname__)
