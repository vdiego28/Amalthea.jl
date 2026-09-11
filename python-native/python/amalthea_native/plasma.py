"""Scalar plasma polarization using the Julia cumulative-current convention."""
import math
import numpy as np
from .grid import _positive
from .ionisation import ELECTRON, M_E, IonRateADK
from .ppt import IonRatePPTAccel


def _cumtrapz(values,dt):
    result=np.empty(values.shape);result[0]=0.
    result[1:]=np.cumsum(.5*(values[:-1]+values[1:])*dt,axis=0)
    return result


class _PlasmaResponse:
    def __init__(self,dt,ratefunc,ionpot,preionfrac=0.):
        self.dt=_positive('plasma time step',dt)
        self.ionpot=_positive('ionisation potential',ionpot)
        if not callable(ratefunc):raise TypeError('ionisation rate must be callable')
        self.ratefunc=ratefunc
        self.preionfrac=float(preionfrac)
        if not math.isfinite(self.preionfrac) or not 0<=self.preionfrac<=1:
            raise ValueError('preionfrac must be in [0,1]')

    def components(self,field):
        if np.iscomplexobj(field):raise ValueError('plasma requires a real scalar field array')
        field=np.array(field,dtype=float,copy=True)
        if field.ndim!=1 or field.size<2 or not np.all(np.isfinite(field)):
            raise ValueError('plasma requires a finite scalar field array')
        supplied=self.ratefunc(field.copy())
        if np.iscomplexobj(supplied):raise ValueError('plasma rate must be real')
        rate=np.array(supplied,dtype=float,copy=True)
        if rate.shape!=field.shape or not np.all(np.isfinite(rate)) or np.any(rate<0):
            raise ValueError('plasma rate must match the field shape and be finite and nonnegative')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            fraction=self.preionfrac+1-np.exp(-_cumtrapz(rate,self.dt))
            phase=fraction*(ELECTRON**2/M_E)*field
            current=_cumtrapz(phase,self.dt)
            selected=field!=0
            current[selected]+=self.ionpot*rate[selected]*(1-fraction[selected])/field[selected]
            polarization=_cumtrapz(current,self.dt)
        if not all(np.all(np.isfinite(x)) for x in (fraction,current,polarization)):
            raise ValueError('nonfinite plasma response')
        return rate,fraction,current,polarization

    def __call__(self,field):
        return self.components(field)[-1]

    def vector_components(self, field):
        """Vector plasma: magnitude rates and componentwise current/loss."""
        if np.iscomplexobj(field):
            raise ValueError('vector plasma requires a real field')
        field=np.array(field,dtype=float,copy=True)
        if field.ndim!=2 or field.shape[1]!=2 or len(field)<2 or not np.all(np.isfinite(field)):
            raise ValueError('vector plasma requires finite (time, 2) fields')
        magnitude=np.hypot(field[:,0],field[:,1])
        supplied=self.ratefunc(magnitude.copy())
        if np.iscomplexobj(supplied):raise ValueError('plasma rate must be real')
        rate=np.array(supplied,dtype=float,copy=True)
        if rate.shape!=magnitude.shape or not np.all(np.isfinite(rate)) or np.any(rate<0):
            raise ValueError('plasma rate must match the time shape and be finite and nonnegative')
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            fraction=self.preionfrac+1-np.exp(-_cumtrapz(rate,self.dt))
            phase=fraction[:,None]*(ELECTRON**2/M_E)*field
            current=_cumtrapz(phase,self.dt)
            selected=magnitude!=0
            loss=self.ionpot*rate[selected]*(1-fraction[selected])/magnitude[selected]**2
            current[selected]+=loss[:,None]*field[selected]
            polarization=_cumtrapz(current,self.dt)
        if not all(np.all(np.isfinite(x)) for x in (fraction,current,polarization)):
            raise ValueError('nonfinite vector plasma response')
        return rate,fraction,current,polarization

    def vector(self, field):
        return self.vector_components(field)[-1]

    def native_config(self,density):
        rate=self.ratefunc
        if type(rate) is IonRateADK:
            if rate.thr<=0:
                return None,'threshold-free ADK uses Python for the zero-field limit'
            data=[rate.occupancy,rate.omega_p,rate.cn_sq,rate.nstar,rate.omega_t_prefac,rate.thr,rate.avfac]
            if not np.all(np.isfinite(data)) or np.any(np.asarray(data)<=0):
                raise ValueError('invalid ADK coefficients')
            prefix=('ADK',data,[],[])
        elif type(rate) is IonRatePPTAccel:
            field=rate._field;log_rate=rate._log_rate;derivative=rate._derivative
            h=np.diff(field);delta=np.diff(log_rate);d0=derivative[:-1];d1=derivative[1:]
            with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
                numerators=(d0,3*delta-2*d0-d1,2*(-delta)+d0+d1)
                coefficients=(d0/h,(numerators[1]/h)/h,((numerators[2]/h)/h)/h)
            if any(not np.all(np.isfinite(c)) or np.any((c==0)&(n!=0)) for c,n in zip(coefficients,numerators)):
                return None,'PPT polynomial scaling requires Python normalized-coordinate evaluation'
            prefix=('PPT',field.tolist(),log_rate.tolist(),derivative.tolist())
        else:
            return None,'direct or custom ionisation models require Python evaluation'
        return (*prefix,self.ionpot,ELECTRON**2/M_E,self.preionfrac,density),None
