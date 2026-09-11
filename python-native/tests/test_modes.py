import itertools
import math
import os
from importlib.resources import files
from pathlib import Path
import tomllib
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
from scipy.integrate import quad

from amalthea_native import MarcatiliMode
from amalthea_native import materials
from amalthea_native.differentiation import derivative, _stencil, _estimate, _step, _coefficients
from amalthea_native.modes import silica_index, _HBAR, _ELECTRON
from amalthea_native.grid import C


@pytest.fixture
def oracle():
    root = os.environ.get('AMALTHEA_MODE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_MODE_ORACLE for independent mode setup acceptance')
    return Path(root)


def relative(a,b):
    return np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-300)


def test_silica_table_and_interpolation(oracle):
    data=np.loadtxt(files('amalthea_native').joinpath('data/silica.txt'))
    np.testing.assert_array_equal(data,np.loadtxt(oracle/'silica-data.txt'))
    meta=tomllib.loads((oracle/'metadata.toml').read_text())
    assert _HBAR==meta['hbar'] and _ELECTRON==meta['electron']
    optical=np.loadtxt(oracle/'silica.txt'); ref=optical[:,1]+1j*optical[:,2]
    assert relative(silica_index(optical[:,0]),ref)<1e-13
    assert np.max(ref.imag)>.1  # Absorbing UV points are actually exercised.


CASES=list(itertools.product([('HE',1,1),('HE',2,3),('HE',8,8),('TE',0,1),('TM',0,4)],
                             ['full','reduced'],[False,True],[False,True]))


def test_julia_other_cores_and_cutoff(oracle):
    for gi,loss,omega,re,im in np.loadtxt(oracle/'extra-cores.txt'):
        gas=['He','N2',None][int(gi)-1]
        mode=MarcatiliMode(80e-6,gas,2.1,loss=bool(loss))
        assert abs(mode.neff(omega)/(re+1j*im)-1)<1e-13
    for loss,omega,re,im in np.loadtxt(oracle/'cutoff.txt'):
        mode=MarcatiliMode(100e-9,cladding_index=lambda w,z:20j,loss=bool(loss))
        assert re==1e-3 and im>=0
        assert abs(mode.neff(omega)/(re+1j*im)-1)<1e-13



@pytest.mark.parametrize('indices,model,loss,profile',CASES)
def test_julia_mode_setup(oracle,indices,model,loss,profile):
    kind,n,m=indices
    name='_'.join(map(str,(kind,n,m,model,str(loss).lower(),str(profile).lower())))
    root=oracle/name
    mode=MarcatiliMode((lambda z:125e-6*(1+.17*math.sin(3*z))) if profile else 125e-6,
                      'Ar',(lambda z:1.3+.7*z**2) if profile else 1.3,
                      kind=kind,n=n,m=m,model=model,loss=loss,phi=.37,T=305.,
                      **({'cladding_index':lambda w,z:1.45+.01*z+(.002+.003*z)*1j} if profile else {}))
    setup=np.loadtxt(root/'setup.txt'); optics=np.loadtxt(root/'optics.txt')
    fields=np.loadtxt(root/'fields.txt'); errors={}
    for z,a,unm,norm,area,refined,error_bound in setup:
        assert mode.radius(z)==a
        errors['root']=abs(mode.unm/unm-1)
        errors['normalization']=abs(mode.N(z=z)/norm-1)
        errors['area']=abs(mode.Aeff(z=z)/refined-1)
        # The default Julia quadrature is retained, not substituted as exact truth.
        assert abs(area/refined-1)<1e-8 and error_bound<=5e-14
        ix=optics[:,0]==z; om=optics[ix,1]
        for key,actual,expected in [
            ('neff',mode.neff(om,z=z),optics[ix,2]+1j*optics[ix,3]),
            ('beta',mode.beta(om,z=z),optics[ix,4]),
            ('alpha',mode.alpha(om,z=z),optics[ix,5])]:
            errors[key]=max(errors.get(key,0.),relative(actual,expected))
            assert actual.flags.owndata
        ix=fields[:,0]==z
        errors['field']=max(errors.get('field',0.),relative(
            mode.field((fields[ix,1],fields[ix,2]),z=z),fields[ix,3:5].T))
    print(name,errors)
    assert max(errors.values())<1e-13
    omega=2*math.pi*C/793e-9; disp=np.loadtxt(root/'dispersion.txt')
    # Higher derivatives are separately investigated; this asserts the moving frame.
    assert abs(mode.dispersion(1,omega,z=.137)/disp[1]-1)<1e-13
    assert mode.dispersion(0,omega,z=.137)==mode.beta(omega,z=.137)
    for q in range(1,8):
        samples=np.loadtxt(root/f'derivative{q}-samples.txt')
        step,_,scale=np.loadtxt(root/f'derivative{q}-step.txt')
        coefficients=_stencil(min(q+6,11),q)[1]
        same_samples=_estimate(samples[:,1],coefficients,step,q)/scale**q
        assert abs(same_samples/disp[q]-1)<1e-13


@pytest.mark.parametrize('order',range(1,8))
def test_julia_derivative_same_samples(oracle,order):
    root=oracle/f'derivative{order}'
    stencil=np.loadtxt(root/'stencil.txt'); estimate=np.loadtxt(root/'estimate.txt')
    samples=np.loadtxt(root/'samples.txt'); grid,coefficients,_,_=_stencil(min(order+6,11),order)
    np.testing.assert_array_equal(grid,stencil[:,0])
    np.testing.assert_array_equal(coefficients,stencil[:,1])
    value=_estimate(samples[:,1],coefficients,estimate[0],order)
    assert abs(value/estimate[2]-1)<1e-13
    p=min(order+6,11)
    bound=np.loadtxt(root/'bound.txt'); fs=np.loadtxt(root/'bound-samples.txt')[:,1]
    bound_grid,_,mult,emult=_stencil(p+2,p)
    assert abs(_step(p+2,p,10.,np.finfo(float).eps)/bound[0]-1)<1e-13
    np.testing.assert_allclose([mult,emult],bound[1:],rtol=1e-13)
    magnitude=max(abs(_estimate(fs,_coefficients(tuple(g+off for g in bound_grid),p),bound[0],p))
                  for off in (-1,0,1))
    assert abs(_step(p,order,magnitude,math.ulp(max(abs(fs))))/estimate[0]-1)<1e-13


@pytest.mark.parametrize('order',range(1,8))
def test_derivative_independent_high_precision_refinement(monkeypatch,order):
    from amalthea_native import differentiation as fd
    steps=[]
    original_step=fd._step
    def track_step(*args):
        value=original_step(*args); steps.append(value); return value
    monkeypatch.setattr(fd,'_step',track_step)
    samples=[]
    def exponential(x):
        value=math.exp(2*x); samples.append(value); return value
    computed=derivative(exponential,1.,order)
    step=min(steps[-1],1000*steps[-2])
    grid,coefficients,_,_=fd._stencil(min(order+6,11),order)
    # Recover the small exact rational coefficients independently of the
    # production Gaussian-elimination implementation.
    rational=[Fraction(c).limit_denominator(10**9) for c in coefficients]
    assert [float(c) for c in rational]==list(coefficients)
    with localcontext() as ctx:
        ctx.prec=100
        weights=[Decimal(c.numerator)/Decimal(c.denominator) for c in rational]
        truth=Decimal(2)**order*Decimal(2).exp()
        def high_precision(h):
            return sum(c*(2*(1+h*g)).exp() for c,g in zip(weights,grid))/h**order
        h=Decimal(step)
        errors=[abs(high_precision(h/2**j)-truth) for j in range(11)]
        assert all(b<a/4 for a,b in zip(errors,errors[1:]))
        assert errors[-1]/truth < Decimal('1e-13')
        # Measured perturbations account for rounded coordinates, function
        # samples, coefficients and denominator. A gamma_n bound then covers
        # the production products, left-fold sum and final division.
        products=[Decimal(v)*Decimal(c) for v,c in zip(samples[-len(grid):],coefficients)]
        denominator=Decimal(step**order)
        perturbed=sum(products)/denominator
        u=Decimal(2)**-53
        operations=2*len(grid)+2
        gamma=operations*u/(1-operations*u)
        roundoff=abs(perturbed-high_precision(h))+gamma*sum(abs(v) for v in products)/denominator
        total_bound=errors[0]+roundoff
        assert total_bound<truth/1000  # Derivative signal is decisively resolved.
        assert abs(Decimal(computed)-truth)<=total_bound
        print(f'derivative {order}: float64 relative error={float(abs(Decimal(computed)-truth)/truth):.3e}, '
              f'refined error={float(errors[-1]/truth):.3e}, bound={float(total_bound/truth):.3e}')


@pytest.mark.parametrize('kind,n,m',[('HE',1,1),('HE',4,8),('TE',0,3),('TM',0,8)])
def test_spatial_integral_normalization_and_area(kind,n,m):
    mode=MarcatiliMode(95e-6,kind=kind,n=n,m=m,phi=.63)
    # Integrate raw spatial vector intensities, independent of the Bessel formula for N.
    theta=np.arange(64)*2*math.pi/64
    def radial(r,power):
        intens=np.sum(mode.field((r,theta))**2,axis=0)
        return r*2*math.pi*np.mean(intens**power)
    i2,e2=quad(lambda x:radial(x*95e-6,1)*95e-6,0,1,epsabs=0,epsrel=1e-12,limit=300)
    i4,e4=quad(lambda x:radial(x*95e-6,2)*95e-6,0,1,epsabs=0,epsrel=1e-12,limit=300)
    assert e2<1e-12*i2 and e4<1e-12*i4
    assert abs(mode.N()/(i2*.5*math.sqrt(materials.EPS0/(4*math.pi*1e-7)))-1)<1e-13
    assert abs(mode.Aeff()/(i2*i2/i4)-1)<1e-13
    np.testing.assert_allclose(mode.field((20e-6,theta),normalized=True),
                               mode.field((20e-6,theta))/math.sqrt(mode.N()),rtol=1e-15)


def test_mode_profiles_callbacks_and_sensitivity():
    calls=[]
    def radius(z):
        calls.append(('radius',z)); return 100e-6*(1+z*z)
    def pressure(z):
        calls.append(('pressure',z)); return 1+z*z
    mode=MarcatiliMode(radius,'Ar',pressure)
    assert calls==[]
    omega=2*math.pi*C/np.array([400e-9,800e-9]); before=omega.copy()
    v=mode.neff(omega,z=.137)
    assert calls==[('radius',.137),('pressure',.137)]
    assert relative(v,mode.neff(omega,z=.71))>1e-5
    assert np.all(mode.alpha(omega)>.01)
    assert np.all(MarcatiliMode(100e-6,loss=False).alpha(omega)==0)
    custom=MarcatiliMode(radius,core_index=lambda w,z:materials.refractive_index('Ar',2*math.pi*C/w,1+z*z))
    np.testing.assert_array_equal(custom.neff(omega,z=.137),v)
    def mutate(w,*,z):
        w[:]=1.; return np.full(w.shape,1.45+.01j)
    changed=MarcatiliMode(100e-6,cladding_index=mutate).neff(omega)
    np.testing.assert_array_equal(omega,before)
    assert changed.shape==omega.shape and changed.flags.owndata
    calls.clear(); mode.field((0.,0.),z=.31,normalized=True)
    assert calls==[('radius',.31)]
    assert mode.dimlimits(z=.2)==('polar',(0.,0.),(radius(.2),2*math.pi))
    assert MarcatiliMode(100e-6,ϕ=.2).phi==MarcatiliMode(100e-6,φ=.2).phi==.2
    # Analytic lossless reduced vacuum beta = omega/c - K/omega.
    reduced=MarcatiliMode(100e-6,model='reduced',loss=False)
    expected=1/C+C*reduced.unm**2/(2*100e-6**2*omega**2)
    np.testing.assert_allclose(reduced.dispersion(1,omega),expected,rtol=1e-13)


def test_mode_invalid_outputs_and_exception_identity():
    omega=np.array([1e15,2e15]); boom=RuntimeError('custom model failed')
    def fail(*args,**kwargs): raise boom
    for mode in [MarcatiliMode(fail),MarcatiliMode(1e-4,core_index=fail),
                 MarcatiliMode(1e-4,'Ar',fail)]:
        with pytest.raises(RuntimeError) as exc: mode.neff(omega)
        assert exc.value is boom
    for bad in [np.zeros((2,1)),[np.inf,np.nan]]:
        with pytest.raises(ValueError): MarcatiliMode(1e-4,core_index=lambda w,z:bad).neff(omega)
    def resize(w,z):
        w.resize((1,2)); return w
    with pytest.raises(ValueError,match='shape'):
        MarcatiliMode(1e-4,core_index=resize).neff(omega)
    for bad in [lambda z:0,lambda z:np.nan,lambda z:[1e-4]]:
        with pytest.raises(ValueError): MarcatiliMode(bad).neff(omega)
    for kwargs in [dict(n=0),dict(m=0),dict(n=1,kind='TE'),dict(model='other'),
                   dict(kind='EH'),dict(T=0),dict(phi=0,ϕ=1),dict(loss=1)]:
        with pytest.raises((ValueError,TypeError)): MarcatiliMode(1e-4,**kwargs)
    for order in [-1,8]:
        with pytest.raises(ValueError): derivative(math.exp,1.,order)
    with pytest.raises(ValueError): derivative(lambda x:np.nan,1.)
    with pytest.raises(RuntimeError) as exc: derivative(fail,1.)
    assert exc.value is boom
