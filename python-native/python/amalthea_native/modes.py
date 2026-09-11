"""Capillary modes with the physical conventions of Capillary.jl."""
from functools import lru_cache
from importlib.resources import files
import math
import operator

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep, splev
from scipy.special import jn_zeros, jv

from . import materials
from .differentiation import derivative
from .grid import C

_ELECTRON = 1.6021766208e-19
_HBAR = 1.0545718001391127e-34  # CODATA2014 h/(2pi), rounded from high precision.
_MU0 = 4*math.pi*1e-7


@lru_cache(maxsize=1)
def _silica_splines():
    data = np.loadtxt(files(__package__).joinpath('data/silica.txt'))
    wavelength = (2*math.pi*C)/(_ELECTRON*data[:,0]/_HBAR)*1e6
    order = np.argsort(wavelength)
    return tuple(splrep(wavelength[order],data[order,i],k=3,s=0) for i in (1,2))


def silica_index(wavelength):
    """Complex silica lookup index, at wavelength [m], with cubic extrapolation."""
    um = materials._values(wavelength,'wavelength',positive=True)*1e6
    real,imag = _silica_splines()
    return materials._owned(splev(um,real)+1j*splev(um,imag))


@lru_cache(maxsize=128)
def _mode_constants(kind, n, m):
    order = n-1 if kind=='HE' else 1
    roots = jn_zeros(order,m)
    zero = float(roots[-1])
    # Splitting at known zeros resolves oscillations at high radial order.
    integral,error = quad(lambda r:r*jv(n-1,zero*r)**4,0.,1.,epsabs=0.,epsrel=2e-13,
                          points=(roots[:-1]/zero).tolist(),limit=max(200,4*m))
    if integral<=0 or error>2e-13*abs(integral):
        raise ValueError('effective-area quadrature did not meet its error criterion')
    next_order = n if kind=='HE' else 2
    factor = float(jv(next_order,zero))
    return zero, factor, 2*math.pi*(.25*factor**4)/integral


def _scalar(value,name,*,positive=False,nonnegative=False):
    result = np.asarray(value)
    if result.ndim!=0 or np.iscomplexobj(result):
        raise ValueError(f'{name} must be a real scalar')
    result = float(result)
    if not math.isfinite(result) or (positive and result<=0) or (nonnegative and result<0):
        raise ValueError(f'invalid {name}')
    return result


class Mode:
    """Base for custom real spatial modes with complex dispersion.

    Implement neff(omega, *, z), field((x1, x2), *, z), and dimlimits(*, z).
    Batched field coordinates return an array with shape (2, points), x/y first.
    N is the physical power normalization; override it when known analytically.
    """
    def neff(self, omega, *, z=0.):
        raise NotImplementedError('a custom mode must implement neff')

    def field(self, coordinates, *, z=0.):
        raise NotImplementedError('a custom mode must implement field')

    def dimlimits(self, *, z=0.):
        raise NotImplementedError('a custom mode must implement dimlimits')

    def _checked_index(self, omega, z):
        omega = materials._values(omega, 'omega', positive=True)
        supplied = np.array(self.neff(np.array(omega, copy=True), z=z), dtype=complex, copy=True)
        if supplied.shape not in ((), np.shape(omega)) or not np.all(np.isfinite(supplied)):
            raise ValueError('mode neff must return a finite scalar or the omega shape')
        return np.asarray(omega), supplied

    def beta(self, omega, *, z=0.):
        omega, index = self._checked_index(omega, _scalar(z, 'z'))
        return materials._owned(omega / C * index.real)

    def alpha(self, omega, *, z=0.):
        omega, index = self._checked_index(omega, _scalar(z, 'z'))
        return materials._owned(2 * omega / C * index.imag)

    def dispersion(self, order, omega, *, z=0.):
        return derivative(lambda w: self.beta(w, z=z), omega, order)

    def N(self, *, z=0.):
        from .spatial import _normalization
        return _normalization(self, z=z)

    def Exy(self, coordinates, *, z=0.):
        return materials._owned(np.asarray(self.field(coordinates, z=z)) / math.sqrt(self.N(z=z)))

    def β(self, omega, *, z=0.):
        return self.beta(omega, z=z)

    def α(self, omega, *, z=0.):
        return self.alpha(omega, z=z)


class MarcatiliMode(Mode):
    """Hollow capillary mode; gas=None means a vacuum core.

    Custom index callbacks receive (omega, *, z), scalar or complete array.
    Radius and pressure may be callables of z. Frequency is rad/s, radius is m,
    pressure is bar, temperature is K. Polarized fields have x/y on axis zero.
    """
    def __init__(self,radius,gas=None,pressure=0.,**kwargs):
        defaults = dict(n=1,m=1,kind='HE',phi=0.,temperature=materials.ROOMTEMP,
                        model='full',loss=True,core_index=None,cladding_index=None)
        supplied = {}
        for key,value in kwargs.items():
            key = {'ϕ':'phi','φ':'phi','T':'temperature'}.get(key,key)
            if key not in defaults:
                raise TypeError(f'unknown mode keyword: {key}')
            if key in supplied:
                raise TypeError(f'duplicate mode keyword: {key}')
            supplied[key]=value
        options = defaults|supplied
        self.n,self.m = operator.index(options['n']),operator.index(options['m'])
        self.kind,self.model = options['kind'],options['model']
        if self.kind not in ('HE','TE','TM') or self.model not in ('full','reduced'):
            raise ValueError('unknown mode kind or model')
        if self.m<1 or (self.n<1 if self.kind=='HE' else self.n!=0):
            raise ValueError('HE requires n>=1; TE/TM require n=0; radial order m>=1')
        if not isinstance(options['loss'],(bool,np.bool_)):
            raise TypeError('loss must be a boolean')
        self.loss=bool(options['loss'])
        self.phi=_scalar(options['phi'],'phi')
        self.temperature=_scalar(options['temperature'],'temperature',positive=True)
        self._radius=radius if callable(radius) else _scalar(radius,'radius',positive=True)
        self._pressure=pressure if callable(pressure) else _scalar(pressure,'pressure',nonnegative=True)
        self.gas=None if gas is None else materials._gas(gas)
        self._core=options['core_index']; self._cladding=options['cladding_index']
        for name,callback in [('core_index',self._core),('cladding_index',self._cladding)]:
            if callback is not None and not callable(callback):
                raise TypeError(f'{name} must be callable')
        if self._core is not None and self.gas is not None:
            raise ValueError('choose a gas core or core_index callback')
        self.unm,self._bessel_norm,self._area_factor=_mode_constants(self.kind,self.n,self.m)

    def radius(self,z=0.):
        z=_scalar(z,'z')
        return _scalar(self._radius(z) if callable(self._radius) else self._radius,'radius',positive=True)

    def pressure(self,z=0.):
        z=_scalar(z,'z')
        return _scalar(self._pressure(z) if callable(self._pressure) else self._pressure,'pressure',nonnegative=True)

    @staticmethod
    def _index(callback,omega,z):
        incoming=np.array(omega,dtype=float,copy=True)
        value=np.array(callback(incoming,z=z),dtype=complex,copy=True)
        if value.ndim!=0 and value.shape!=np.shape(omega):
            raise ValueError('index callback must return a scalar or the omega shape')
        if not np.all(np.isfinite(value)):
            raise ValueError('index callback returned nonfinite values')
        return value

    def neff(self,omega,*,z=0.):
        omega=materials._values(omega,'omega',positive=True)
        z=_scalar(z,'z'); a=self.radius(z)
        cladding = self._index(self._cladding,omega,z) if self._cladding else silica_index(2*math.pi*C/omega)
        core = (self._index(self._core,omega,z) if self._core else
                (1.+0j if self.gas is None else materials.refractive_index(self.gas,2*math.pi*C/omega,self.pressure(z),self.temperature)))
        return self._neff_from_indices(omega,core,cladding,a)

    def _neff_from_indices(self,omega,core,cladding,a):
        ec,eco=np.asarray(cladding)**2,np.asarray(core)**2
        base=np.sqrt(np.asarray(ec-1,dtype=complex))
        vn={'HE':lambda:(ec+1)/(2*base),'TE':lambda:1/base,'TM':lambda:ec/base}[self.kind]()
        if self.model=='full':
            k=omega/C
            value=np.sqrt(eco-(self.unm/(k*a))**2*(1-1j*vn/(k*a))**2+0j)
            if self.loss:
                value=np.where(value.real<1e-3,1e-3+1j*np.maximum(value.imag,0),value)
            else:
                value=np.maximum(value.real,1e-3)
        else:
            value=1+(eco-1)/2-C**2*self.unm**2/(2*omega**2*a**2)
            if self.loss:
                value=value+1j*(C**3*self.unm**2)/(a**3*omega**3)*vn
            else:
                value=value.real
        return materials._owned(value)

    def beta(self,omega,*,z=0.):
        return materials._owned(np.asarray(omega)/C*np.real(self.neff(omega,z=z)))

    def alpha(self,omega,*,z=0.):
        return materials._owned(2*np.asarray(omega)/C*np.imag(self.neff(omega,z=z)))

    β=beta
    α=alpha

    def dispersion(self,order,omega,*,z=0.):
        return derivative(lambda w:self.beta(w,z=z),omega,order)

    def field(self,coordinates,*,z=0.,normalized=False):
        if len(coordinates)!=2:
            raise ValueError('field coordinates must be (r, theta)')
        r,theta=np.broadcast_arrays(np.asarray(coordinates[0],dtype=float),np.asarray(coordinates[1],dtype=float))
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(theta)):
            raise ValueError('field coordinates must be finite')
        a=self.radius(z)
        if self.kind=='HE':
            bessel=jv(self.n-1,r*self.unm/a)
            sin,cos=np.sin(self.n*(theta+self.phi)),np.cos(self.n*(theta+self.phi))
            out=np.array([bessel*(np.cos(theta)*sin-np.sin(theta)*cos),
                          bessel*(np.sin(theta)*sin+np.cos(theta)*cos)])
        else:
            bessel=jv(1,r*self.unm/a)
            out=np.array([-bessel*np.sin(theta),bessel*np.cos(theta)] if self.kind=='TE'
                         else [bessel*np.cos(theta),bessel*np.sin(theta)])
        if normalized:
            # Reuse this radius so an arbitrary stateful profile is called once.
            out /= math.sqrt(self._normalization(a))
        return materials._owned(out)

    def _normalization(self,a):
        return math.pi/2*a**2*self._bessel_norm**2*math.sqrt(materials.EPS0/_MU0)

    def N(self,*,z=0.):
        return self._normalization(self.radius(z))

    def effective_area(self,*,z=0.):
        return self.radius(z)**2*self._area_factor

    Aeff=effective_area

    def dimlimits(self,*,z=0.):
        return ('polar',(0.,0.),(self.radius(z),2*math.pi))
