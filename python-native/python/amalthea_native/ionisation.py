"""Ionisation setup in SI units using the project's CODATA2014 conventions."""
from functools import lru_cache
import math
import operator

import numpy as np

from .grid import C, _positive
from .materials import ionisation_potential

ELECTRON=1.6021766208e-19
M_E=9.10938356e-31
HBAR=1.0545718001391127e-34
BOHR_RADIUS=.52917721067e-10
FINE_STRUCTURE=7.2973525664e-3
AU_ENERGY=HBAR*C*FINE_STRUCTURE/BOHR_RADIUS
AU_EFIELD=AU_ENERGY/(ELECTRON*BOHR_RADIUS)


def _coefficients(ionpot):
    nstar=math.sqrt(.5/(ionpot/AU_ENERGY))
    cn_sq=2**(2*nstar)/(nstar*math.gamma(nstar+1)*math.gamma(nstar))
    omega_p=ionpot/HBAR
    omega_t_prefac=ELECTRON/math.sqrt(2*M_E*ionpot)
    if not all(math.isfinite(v) and v>0 for v in (nstar,cn_sq,omega_p,omega_t_prefac)):
        raise ValueError('ionisation potential produces invalid ADK coefficients')
    return nstar,cn_sq,omega_p,omega_t_prefac


@lru_cache(maxsize=128)
def _threshold(ionpot):
    nstar,cn_sq,omega_p,omega_t_prefac=_coefficients(ionpot)
    field=1e3
    for _ in range(100000):
        rate=(2*omega_p*cn_sq*(4*omega_p/(omega_t_prefac*field))**(2*nstar-1)
              *math.exp(-4/3*omega_p/(omega_t_prefac*field)))
        if rate!=0:
            if not math.isfinite(rate):
                raise ValueError('invalid ADK threshold calculation')
            return field
        field*=1.01
    raise ValueError('ADK threshold did not converge')


class IonRateADK:
    """ADK rate [1/s] for a material or ionisation potential [J].

    Call with a real electric field [V/m] or array. Threshold discovery follows
    Julia's sequential underflow search. At zero field the result is zero even
    when threshold=False, using the continuous physical limit.
    """
    def __init__(self,material_or_ionpot,*,occupancy=2,threshold=True,cycle_average=False):
        if not isinstance(threshold,(bool,np.bool_)) or not isinstance(cycle_average,(bool,np.bool_)):
            raise TypeError('threshold and cycle_average must be booleans')
        occupancy=operator.index(occupancy)
        if occupancy<=0:
            raise ValueError('occupancy must be a positive integer')
        self.ionpot=(ionisation_potential(material_or_ionpot) if isinstance(material_or_ionpot,str)
                     else _positive('ionisation potential',material_or_ionpot))
        self.occupancy=occupancy
        self.threshold=bool(threshold);self.cycle_average=bool(cycle_average)
        try:
            self.nstar,self.cn_sq,self.omega_p,self.omega_t_prefac=_coefficients(self.ionpot)
            self.thr=_threshold(self.ionpot) if threshold else 0.
            self.avfac=(math.sqrt(3/(math.pi*(2*self.ionpot/AU_ENERGY)**1.5*AU_EFIELD))
                        if cycle_average else 1.)
        except (OverflowError,ZeroDivisionError) as error:
            raise ValueError('ionisation potential is outside finite ADK setup range') from error
        if not math.isfinite(self.avfac) or self.avfac<=0:
            raise ValueError('ionisation potential produces an invalid cycle-average factor')

    def __call__(self,field):
        if np.iscomplexobj(field):
            raise ValueError('ADK requires a real electric field')
        field=np.asarray(field,dtype=float)
        if not np.all(np.isfinite(field)):
            raise ValueError('ADK requires a finite electric field')
        magnitude=np.abs(field)
        active=(magnitude>=self.thr)&(magnitude>0)
        selected=magnitude[active]
        result=np.zeros(field.shape)
        with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
            rate=(self.occupancy*self.omega_p*self.cn_sq*
                  (4*self.omega_p/(self.omega_t_prefac*selected))**(2*self.nstar-1)*
                  np.exp(-4/3*self.omega_p/(self.omega_t_prefac*selected)))
            if self.avfac!=1:
                rate*=self.avfac*np.sqrt(selected)
        if not np.all(np.isfinite(rate)) or np.any(rate<0):
            raise ValueError('ADK returned a nonfinite rate at the supplied field')
        result[active]=rate
        return result.item() if result.ndim==0 else result
