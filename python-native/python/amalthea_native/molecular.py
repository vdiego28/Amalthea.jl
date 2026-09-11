"""Molecular Raman setup with PhysData constants and density-dependent linewidths."""
from copy import deepcopy
import math
import operator
import numpy as np
from .grid import C,_positive,planck_taper
from .envelope import EPS0
from .ionisation import HBAR
from .materials import ROOMTEMP

K_B=1.38064852e-23
M_U=1.660539040e-27
AMG=101325/(K_B*273.15)


def _reduced_mass(mass):return (M_U*mass)**2/(2*M_U*mass)

_PARAMETERS={
    'N2':dict(rotation='nonrigid',vibration='sdo',B=199.,D=5.74e-4,qJodd=1,qJeven=2,
              delta_alpha=6.7e-31,B_r=3.3e9,A_r=0.,dalpha_dQ=1.75e-20,
              omega_v=2*math.pi*2330.*100.*C,mu=_reduced_mass(14.0067),tau_v=6e-12),
    'H2':dict(rotation='nonrigid',vibration='sdo',B=5890.,D=5.,qJodd=3,qJeven=1,
              delta_alpha=3e-31,B_r=114e6,A_r=6.15e6,dalpha_dQ=1.3e-20,
              omega_v=2*math.pi*124.5669e12,mu=_reduced_mass(1.00784),B_v=52.2e6,A_v=309e6),
    'D2':dict(rotation='nonrigid',vibration='sdo',B=2930.,D=2.1,qJodd=1,qJeven=2,
              delta_alpha=3e-31,B_r=4e-3*100.*C,A_r=0.,dalpha_dQ=1.4e-20,
              omega_v=2*math.pi*2987*100.*C,mu=_reduced_mass(2.014),B_v=120e6,A_v=101e6),
    'N2O':dict(rotation='nonrigid',vibration='none',B=41.,D=0.,qJodd=1,qJeven=1,
               delta_alpha=28.1e-31,tau_r=23.8e-12,omega_v=2*math.pi*1285*100.*C),
    'CH4':dict(rotation='none',vibration='sdo',dalpha_dQ=1.04e-20,
               omega_v=2*math.pi*2914*100.*C,mu=(1.00784*M_U)/4,B_v=384e6,A_v=0.,C_v=8220e6),
    'SF6':dict(rotation='none',vibration='sdo',dalpha_dQ=1.23e-20,
               omega_v=2*math.pi*775*100.*C,mu=(18.998403*M_U)/6,tau_v=6.6e-12),
    'O2':dict(rotation='nonrigid',vibration='sdo',B=144.,D=0.,qJodd=1,qJeven=0,
              delta_alpha=10.2e-31,dalpha_dQ=1.46e-20,omega_v=3e14,mu=1.3e-26),
}


class MolecularRaman:
    """Owned causal Raman samples; call with number density [m^-3].

    Time is a uniform array [s], temperature is kelvin. Oscillator data uses
    angular frequencies [rad/s], absolute couplings and dephasing times [s].
    """
    def __init__(self,time,gas,*,rotation=True,vibration=True,minJ=0,maxJ=50,temperature=ROOMTEMP):
        if np.iscomplexobj(time):raise ValueError('Raman time axis must be real')
        time=np.array(time,dtype=float,copy=True)
        if time.ndim!=1 or time.size<2 or time.size>1<<23 or not np.all(np.isfinite(time)):
            raise ValueError('Raman time axis must be a finite one-dimensional array')
        self.dt=_positive('Raman time step',time[1]-time[0])
        tolerance=64*np.finfo(float).eps*max(np.max(abs(time)),self.dt)
        if np.any(np.diff(time)<=0) or np.any(abs(np.diff(time)-self.dt)>tolerance):
            raise ValueError('Raman time axis must be uniform and increasing')
        if not isinstance(gas,str) or gas not in _PARAMETERS:raise ValueError('no molecular Raman data for gas')
        if not all(isinstance(v,(bool,np.bool_)) for v in (rotation,vibration)):
            raise TypeError('rotation and vibration must be booleans')
        self.temperature=_positive('temperature',temperature)
        minJ=operator.index(minJ);maxJ=operator.index(maxJ)
        if minJ<0 or maxJ<minJ+2 or maxJ>100000:raise ValueError('invalid rotational J range')
        self.gas=gas;self._parameters=rp=deepcopy(_PARAMETERS[gas])
        self._time=np.arange(len(time))*self.dt
        end=self._time[-1];self._window=planck_taper(self._time,-end,-end*.7,end*.7,end)
        self._groups=[];self._rotation_data=None
        if rotation and rp['rotation']!='none':
            self._require_linewidth('r')
            J=np.arange(minJ,maxJ+1,dtype=float)
            energy=2*math.pi*HBAR*C*(rp['B']*J*(J+1)-rp['D']*(J*(J+1))**2)
            decreasing=np.flatnonzero(np.diff(energy)<0)
            if decreasing.size:
                mj=int(decreasing[0])+1
                if max(mj-minJ+1,0)<=2:raise ValueError('Raman rotor cannot sum over levels (Julia truncation guard)')
                J=J[:mj];energy=energy[:mj]
            degeneracy=np.where(J%2!=0,rp['qJodd'],rp['qJeven'])
            population=degeneracy*(2*J+1)*np.exp(-energy/(K_B*self.temperature))
            total=np.sum(population)
            if not math.isfinite(total) or total<=0:raise ValueError('unresolved rotational populations')
            population/=total
            self._rotation_data=dict(J=J.copy(),energy=energy.copy(),population=population.copy())
            starts=J[:-2];omega=(energy[2:]-energy[:-2])/HBAR
            factor=-(4*math.pi*EPS0)**2*2/15*rp['delta_alpha']**2/HBAR
            coupling=(factor*(starts+1)*(starts+2)/(2*starts+3)*
                      (population[2:]/(2*starts+5)-population[:-2]/(2*starts+1)))
            self._add_group('rotation','r',omega,coupling)
        if vibration and rp['vibration']!='none':
            self._require_linewidth('v')
            omega=rp['omega_v'];coupling=(4*math.pi*EPS0)**2*rp['dalpha_dQ']**2/(4*rp['mu']*omega)
            self._add_group('vibration','v',np.array([omega]),np.array([coupling]))

    def _require_linewidth(self,suffix):
        if f'tau_{suffix}' not in self._parameters and not all(f'{k}_{suffix}' in self._parameters for k in ('A','B')):
            raise NotImplementedError(f'{self.gas} has no {suffix} Raman lifetime in the Julia oracle')

    def _add_group(self,name,suffix,omega,coupling):
        if not np.all(np.isfinite(omega)) or np.any(omega<=0) or not np.all(np.isfinite(coupling)):
            raise ValueError('nonfinite or invalid molecular oscillator data')
        pre=np.zeros(self._time.shape)
        for w,k in zip(omega,coupling):pre+=k*np.sin(w*self._time)
        self._groups.append(dict(name=name,suffix=suffix,omega=omega.copy(),coupling=coupling.copy(),pre=pre))

    def _density(self,density):
        if np.iscomplexobj(density) or np.ndim(density):raise ValueError('Raman density must be a real scalar')
        density=float(density)
        if not math.isfinite(density) or density<0:raise ValueError('Raman density must be finite and nonnegative')
        return density

    def _tau(self,suffix,density):
        rp=self._parameters
        if f'tau_{suffix}' in rp:return rp[f'tau_{suffix}']
        if density==0:
            if suffix=='r':return math.inf
            raise ValueError('zero-density vibrational broadening has a nonfinite Julia origin')
        width=rp.get(f'C_{suffix}',0.)+rp[f'A_{suffix}']/(density/AMG)+rp[f'B_{suffix}']*density/AMG
        if not math.isfinite(width) or width<0:raise ValueError('invalid Raman linewidth')
        return math.inf if width==0 else 1/(math.pi*width)

    def __call__(self,density):
        density=self._density(density);result=np.zeros(self._time.shape)
        for group in self._groups:
            result+=group['pre']*np.exp(-self._time/self._tau(group['suffix'],density))
        result*=self._window
        if not np.all(np.isfinite(result)):raise ValueError('nonfinite Raman response')
        return result

    def groups(self,density):
        density=self._density(density)
        return [dict(name=g['name'],omega=g['omega'].copy(),coupling=g['coupling'].copy(),
                     tau2=self._tau(g['suffix'],density)) for g in self._groups]

    def oscillators(self,density):
        groups=self.groups(density)
        return {key:(np.concatenate([np.full(len(g['omega']),g[key]) if key=='tau2' else g[key] for g in groups])
                     if groups else np.array([],dtype=float)) for key in ('omega','coupling','tau2')}

    @property
    def time(self):return self._time.copy()
    @property
    def window(self):return self._window.copy()
    @property
    def parameters(self):return deepcopy(self._parameters)
    @property
    def rotation_data(self):return deepcopy(self._rotation_data)
