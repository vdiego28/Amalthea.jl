"""PPT ionisation rates and local, parameter-keyed acceleration tables."""
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
from scipy.special import dawsn, gamma, hyp1f1

from .grid import C, _positive
from .ionisation import AU_ENERGY, AU_EFIELD, HBAR, ELECTRON, BOHR_RADIUS
from .materials import ionisation_potential
from ._spline import _NormalizedCubic

AU_TIME=HBAR/AU_ENERGY
AU_POLARISABILITY=ELECTRON**2*BOHR_RADIUS**2/AU_ENERGY
_QUANTUM={'He':(1,0,1.),'HeJ':(1,0,1.),'HeB':(1,0,1.),'Ne':(2,1,1.),
          'Ar':(3,1,1.),'ArB':(3,1,1.),'Kr':(4,1,1.),'Xe':(5,1,1.),
          'O2':(2,0,.53),'N2':(2,0,.9)}
_STATIC={'He':(1.3207,.2811),'HeJ':(1.3207,.2811),'HeB':(1.3207,.2811),
         'Ne':(2.376,1.2417),'Ar':(10.762,6.807),'ArB':(10.762,6.807)}


def _phi_precise(m,x):
    # Context cloning prevents precision changes leaking to other simulations.
    from mpmath import mp
    ctx=mp.clone(); previous=None
    for precision in (80,120,180):
        ctx.dps=precision
        value=ctx.mpf(x)
        result=(ctx.sqrt(ctx.pi)*value**(2*m+1)*ctx.gamma(m+1)/(2*ctx.gamma(m+ctx.mpf('1.5')))
                *ctx.hyp1f1(m+1,m+ctx.mpf('1.5'),-value*value))
        if not ctx.isfinite(result) or result<0:
            raise ValueError('invalid high-precision PPT phi')
        if previous is not None and (result==previous or abs((result-previous)/result)<ctx.mpf('1e-60')):
            return float(result)
        previous=result
    raise ArithmeticError('PPT phi precision refinement did not converge')


def _phi(m,x):
    m=abs(operator.index(m));x=np.asarray(x,dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x<0):
        raise ValueError('PPT phi requires finite nonnegative arguments')
    if m==0:
        result=np.array(dawsn(x),copy=True)
    else:
        flat=x.reshape(-1);out=np.empty(flat.shape)
        moderate=flat<=26
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            v=flat[moderate]
            out[moderate]=(math.sqrt(math.pi)*v**(2*m+1)*gamma(m+1)/(2*gamma(m+1.5))
                           *hyp1f1(m+1,m+1.5,-v*v))
        for i in np.flatnonzero(~moderate):out[i]=_phi_precise(m,float(flat[i]))
        for i in np.flatnonzero(~np.isfinite(out)):
            out[i]=_phi_precise(m,float(flat[i]))
        result=out.reshape(x.shape).copy()
    if not np.all(np.isfinite(result)) or np.any(result<0):
        raise ValueError('nonfinite PPT phi result')
    return result.item() if result.ndim==0 else result


def _real_field(field):
    if np.iscomplexobj(field):raise ValueError('PPT requires a real field')
    field=np.asarray(field,dtype=float)
    if not np.all(np.isfinite(field)):raise ValueError('PPT requires a finite field')
    return field


def _occupancy(value):
    value=np.asarray(value)
    if value.ndim or np.iscomplexobj(value):raise ValueError('occupancy must be a real scalar')
    value=float(value)
    if not math.isfinite(value) or value<0:raise ValueError('occupancy must be finite and nonnegative')
    return value


class IonRatePPT:
    """PPT rate [1/s] for real fields [V/m]; numeric potentials are joules.

    Use IonRatePPT(material, lambda0) or IonRatePPT(ionpot, lambda0, Z, l).
    Occupancy callbacks receive an m quantum number serially in field order.
    """
    def __init__(self,material_or_ionpot,lambda0,Z=None,l=None,*,stark_shift=True,
                 dipole_corr=True,**kwargs):
        defaults=dict(delta_alpha=0.,alpha_ion=0.,sum_tol=1e-6,cycle_average=False,
                      sum_integral=False,msum=True,Cnl=None,occupancy=2,max_terms=1_000_000)
        supplied={}
        for key,value in kwargs.items():
            key={'Δα':'delta_alpha','α_ion':'alpha_ion'}.get(key,key)
            if key not in defaults:raise TypeError(f'unknown PPT option: {key}')
            if key in supplied:raise TypeError(f'duplicate PPT option: {key}')
            supplied[key]=value
        for value in (stark_shift,dipole_corr):
            if not isinstance(value,(bool,np.bool_)):raise TypeError('PPT correction flags must be booleans')
        if isinstance(material_or_ionpot,str):
            if Z is not None or l is not None:raise TypeError('material setup supplies Z and l')
            if material_or_ionpot not in _QUANTUM:raise ValueError('no PPT quantum numbers for material')
            _,l,Z=_QUANTUM[material_or_ionpot]
            potential=ionisation_potential(material_or_ionpot)
            neutral,cation=_STATIC.get(material_or_ionpot,(0.,0.))
            defaults['delta_alpha']=(neutral*AU_POLARISABILITY-cation*AU_POLARISABILITY) if stark_shift else 0.
            defaults['alpha_ion']=cation*AU_POLARISABILITY if dipole_corr else 0.
        else:potential=_positive('ionisation potential',material_or_ionpot)
        kw=defaults|supplied
        for name in ['cycle_average','sum_integral','msum']:
            if not isinstance(kw[name],(bool,np.bool_)):raise TypeError(f'{name} must be a boolean')
            kw[name]=bool(kw[name])
        for name in ['delta_alpha','alpha_ion']:
            kw[name]=float(0. if kw[name] is None else kw[name])
            if not math.isfinite(kw[name]):raise ValueError(f'{name} must be finite')
        if kw['Cnl'] is not None:kw['Cnl']=_positive('Cnl',kw['Cnl'])
        if not callable(kw['occupancy']):kw['occupancy']=_occupancy(kw['occupancy'])
        kw['sum_tol']=_positive('sum_tol',kw['sum_tol'])
        if kw['sum_tol']>=1:raise ValueError('sum_tol must be less than one')
        kw['max_terms']=operator.index(kw['max_terms'])
        if kw['max_terms']<1:raise ValueError('max_terms must be positive')
        self.ionpot=potential;self.lambda0=_positive('lambda0',lambda0)
        self.Z=_positive('Z',Z);self.l=operator.index(l)
        if self.l<0:raise ValueError('l must be nonnegative')
        self.kw=kw
        self.omega0_au=AU_TIME*(2*math.pi*C/self.lambda0)
        self.alpha_ion_au=kw['alpha_ion']/AU_POLARISABILITY
        if not math.isfinite(self.omega0_au) or not math.isfinite(self.alpha_ion_au):
            raise ValueError('nonfinite PPT setup')

    def _series(self,m,v,alpha,beta):
        total=np.zeros(v.shape);n=np.ceil(v);active=np.ones(v.shape,dtype=bool)
        if np.any(n+1==n):raise ValueError('PPT photon count exceeds floating-point resolution')
        for _ in range(self.kw['max_terms']):
            ids=np.flatnonzero(active)
            diff=n[ids]-v[ids]
            old=total[ids]
            new=old+np.exp(-alpha[ids]*diff)*_phi(m,np.sqrt(beta[ids]*diff))
            with np.errstate(invalid='ignore',divide='ignore'):
                done=2*abs(new-old)/abs(new+old)<self.kw['sum_tol']
            total[ids]=new;n[ids]+=1;active[ids[done]]=False
            if not np.any(active):return total
        raise ArithmeticError('PPT multiphoton series did not converge within max_terms')

    def _evaluate(self,field):
        kw=self.kw
        ip=(self.ionpot+kw['delta_alpha']/2*field**2)/AU_ENERGY
        if np.any(ip<=0) or not np.all(np.isfinite(ip)):raise ValueError('invalid Stark-shifted potential')
        ns=self.Z/np.sqrt(2*ip);ls=ns-1
        cn=(2**(2*ns)/(ns*gamma(ns+ls+1)*gamma(ns-ls)) if kw['Cnl'] is None else kw['Cnl']**2)
        e0=(2*ip)**1.5;ea=field/AU_EFIELD
        g=self.omega0_au*np.sqrt(2*ip)/ea;g2=g*g
        beta=2*g/np.sqrt(1+g2);alpha=2*(np.arcsinh(g)-g/np.sqrt(1+g2))
        v=(ip+ea**2/(4*self.omega0_au**2))/self.omega0_au
        if not all(np.all(np.isfinite(x)) for x in (ns,cn,g2,beta,alpha,v)) or np.any(alpha<=0):
            raise ValueError('PPT evaluation exceeds resolved floating-point domain')
        result=np.zeros(field.shape)
        for m in (range(-self.l,self.l+1) if kw['msum'] else [0]):
            ma=abs(m)
            flm=(2*self.l+1)*math.factorial(self.l+ma)/(2**ma*math.factorial(ma)*math.factorial(self.l-ma))
            partial=4*math.sqrt(2)/math.pi*cn
            partial=partial*(2*e0/(ea*np.sqrt(1+g2)))**(2*ns-ma-1.5)
            partial*=flm/math.factorial(ma)
            partial*=np.exp(-2*v*(np.arcsinh(g)-g*np.sqrt(1+g2)/(1+2*g2)))
            partial*=ip*g2/(1+g2)
            if not kw['cycle_average']:partial*=np.sqrt(math.pi*e0/(3*ea))
            if kw['sum_integral']:
                series=math.sqrt(math.pi)*math.factorial(ma)*beta**ma/(2*(alpha+beta)**(ma+1))*np.sqrt(beta/alpha)
            else:series=self._series(m,v,alpha,beta)
            partial*=series
            occ=kw['occupancy']
            if callable(occ):occ=_occupancy(occ(m))
            result+=occ*partial
        if self.alpha_ion_au!=0:result*=np.exp(-2*self.alpha_ion_au*ea)
        result/=AU_TIME
        if not np.all(np.isfinite(result)) or np.any(result<0):raise ValueError('invalid PPT rate result')
        return result

    def __call__(self,field):
        values=_real_field(field);magnitude=np.abs(values).reshape(-1)
        result=np.zeros(magnitude.shape);active=magnitude>0
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            if callable(self.kw['occupancy']):
                for i in np.flatnonzero(active):result[i]=self._evaluate(magnitude[i:i+1])[0]
            elif np.any(active):result[active]=self._evaluate(magnitude[active])
        result=result.reshape(values.shape).copy()
        return result.item() if result.ndim==0 else result

    def parameters(self):
        if callable(self.kw['occupancy']):raise ValueError('occupancy callbacks require cache=False')
        return dict(ionpot=self.ionpot,lambda0=self.lambda0,Z=self.Z,l=self.l,**self.kw)


def barrier_suppression(ionpot,Z):
    ip=_positive('ionpot',ionpot)/AU_ENERGY;Z=_positive('Z',Z)
    ns=Z/math.sqrt(2*ip)
    return Z**3/(16*ns**4)*AU_EFIELD


_CACHE_VERSION=2


def _digest(field,rate):
    h=hashlib.sha256();h.update(field.tobytes());h.update(rate.tobytes());return h.hexdigest()


class IonRatePPTAccel:
    """Locally generated log-spline PPT table with atomic, parameter-keyed caching."""
    def __init__(self,material_or_ionpot,lambda0,Z=None,l=None,*,N=65536,Emax=None,
                 cache=True,cachedir=None,**kwargs):
        if not isinstance(cache,(bool,np.bool_)):raise TypeError('cache must be boolean')
        self.model=IonRatePPT(material_or_ionpot,lambda0,Z,l,**kwargs)
        N=operator.index(N)
        if N<4 or N>1<<24:raise ValueError('N must be in 4..2**24')
        Emax=(2*barrier_suppression(self.model.ionpot,self.model.Z) if Emax is None else _positive('Emax',Emax))
        expected=np.linspace(Emax/5000,Emax,N)
        self.cache_hit=False;self.cache_path=None
        metadata=None
        if cache:
            metadata=json.dumps(dict(version=_CACHE_VERSION,N=N,Emax=Emax,
                                      constants=[AU_ENERGY,AU_EFIELD,AU_TIME,AU_POLARISABILITY,C],
                                      model=self.model.parameters()),sort_keys=True,separators=(',',':'))
            key=hashlib.sha256(metadata.encode()).hexdigest()
            base=(Path(cachedir) if cachedir is not None else
                  Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'amalthea_native'/'ppt')
            self.cache_path=base/f'{key}.npz'
            if self.cache_path.is_file():
                try:
                    with np.load(self.cache_path,allow_pickle=False) as data:
                        field=np.array(data['field'],dtype=float,copy=True)
                        rate=np.array(data['rate'],dtype=float,copy=True)
                        if (str(data['metadata'])!=metadata or str(data['digest'])!=_digest(field,rate)
                            or not np.array_equal(field,expected)):
                            raise ValueError('incompatible PPT cache')
                    self._setup(field,rate);self.cache_hit=True;return
                except (OSError,ValueError,KeyError,EOFError,BadZipFile):pass
        rate=self.model(expected)
        self._setup(expected,rate)
        if cache:
            self.cache_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=self.cache_path.parent,suffix='.npz',delete=False) as file:
                    temporary=Path(file.name)
                    np.savez_compressed(file,field=expected,rate=rate,metadata=metadata,digest=_digest(expected,rate))
                os.replace(temporary,self.cache_path)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)

    @classmethod
    def from_samples(cls,field,rate):
        obj=cls.__new__(cls);obj.model=None;obj.cache_hit=False;obj.cache_path=None
        obj._setup(field,rate);return obj

    def _setup(self,field,rate):
        if np.iscomplexobj(field) or np.iscomplexobj(rate):raise ValueError('PPT samples must be real')
        field=np.array(field,dtype=float,copy=True);rate=np.array(rate,dtype=float,copy=True)
        if (field.ndim!=1 or rate.shape!=field.shape or not np.all(np.isfinite(field))
            or not np.all(np.isfinite(rate)) or np.any(field<=0) or np.any(np.diff(field)<=0)
            or np.any(rate<0)):
            raise ValueError('invalid PPT table samples')
        selected=rate>0
        if np.count_nonzero(selected)<4:raise ValueError('PPT table requires four positive-rate samples')
        self._field=field[selected];self._rate=rate[selected]
        self.Emin=float(self._field[0]);self.Emax=float(self._field[-1])
        self._log_rate=np.log(self._rate)
        self._spline=_NormalizedCubic(self._field,self._log_rate)
        self._derivative=self._spline.derivative

    def _log_value(self,field):
        return self._spline(field)

    @property
    def field_nodes(self):return self._field.copy()

    @property
    def rate_nodes(self):return self._rate.copy()

    def __call__(self,field):
        values=_real_field(field);magnitude=np.abs(values)
        result=np.zeros(values.shape);selected=magnitude>=self.Emin
        result[selected]=np.exp(self._log_value(np.minimum(magnitude[selected],self.Emax)))
        if not np.all(np.isfinite(result)):raise ValueError('nonfinite interpolated PPT rate')
        return result.item() if result.ndim==0 else result


IonRatePPTCached=IonRatePPTAccel
