import gc
import json
import math
import os
from pathlib import Path
import threading
import tomllib
import weakref

import numpy as np
import pytest

from amalthea_native import prop_capillary, solve_precon
from amalthea_native.capillary import _Capillary
from amalthea_native.profiles import _GasProfile
from amalthea_native._spline import _NormalizedCubic
from amalthea_native import materials
from amalthea_native.solver import _range_positions

LENGTH = .0002
STEP = .00001
BASE = dict(lambda0=800e-9, lambda_lims=(200e-9,1700e-9), trange=300e-15,
            tau_fwhm=20e-15, energy=500e-6, plasma=False, saveN=7)
NAMES = [f'Ar-{grid}-{kind}' for grid in ('env','real') for kind in
         ('base','rise','fall','multi','pressure','taper','both','flat-pressure','flat-radius')]
NAMES += [f'{gas}-{grid}' for gas in ('N2','H2','D2','N2O','CH4','SF6') for grid in ('env','real')]
NAMES += [f'N2-real-{kind}' for kind in ('no-raman','plasma','rotation','hot','fourth','fine',
                                      'no-thg','no-kerr','reduced-taper')]
NAMES += ['N2-env-thg','H2-real-plasma']


def profiles(radius, pressure):
    radii = dict(constant=125e-6, flat=lambda z:125e-6,
                 linear=lambda z:125e-6*(1-.2*z/LENGTH),
                 curved=lambda z:125e-6*(1+.1*math.sin(math.pi*z/LENGTH)+.05*(z/LENGTH)**2))
    pressures = dict(constant=2., flat=lambda z:2., rise=(2.,4.), fall=(4.,2.),
                     multi=([0.,.00007,LENGTH],[2.,4.,1.]),
                     callable=lambda z:2*(1+.4*math.sin(math.pi*z/LENGTH)+.1*(z/LENGTH)**2))
    return radii[radius], pressures[pressure]


def relative(a,b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


def field(path):
    data=np.loadtxt(path);n=data.shape[1]//2
    return data[:,:n]+1j*data[:,n:]


@pytest.fixture
def oracle():
    root=os.environ.get('AMALTHEA_PROFILE_CAPILLARY_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_PROFILE_CAPILLARY_ORACLE for profile acceptance')
    return Path(root)


@pytest.mark.parametrize('name',NAMES)
def test_independent_profile_capillary(oracle,name,monkeypatch):
    root=oracle/name;kw=tomllib.loads((root/'parameters.toml').read_text())
    radius,pressure=profiles(kw.pop('radius_profile'),kw.pop('pressure_profile'))
    args=(radius,kw.pop('flength'),kw.pop('gas'),pressure)
    model=_Capillary(*args,**kw,backend='python')
    initial=field(root/'initial.txt')[:,0]
    errors={'input':relative(model.initial,initial)}
    for i in range(1,5):
        z,rho,area,beta1,beta0=np.loadtxt(root/f'material-{i}.txt')
        data=np.loadtxt(root/f'setup-{i}.txt');reference=data[:,0]+1j*data[:,1]
        actual=model.rhs(z,initial)
        errors[f'rhs-{i}']=relative(actual,reference)
        errors[f'density-{i}']=relative(model.density,rho)
        errors[f'area-{i}']=relative(model.area,area)
        errors[f'beta1-{i}']=relative(model.mode.dispersion(1,model.omega0,z=z),beta1)
        errors[f'beta0-{i}']=relative(model.mode.beta(model.omega0,z=z),beta0)
        neff=data[:,4]+1j*data[:,5]
        errors[f'neff-{i}']=relative(model.mode.neff(model.grid.omega[model.grid.sidx],z=z),neff[model.grid.sidx])
        if model.h is not None:
            errors[f'raman-{i}']=relative(model.h,np.loadtxt(root/f'raman-{i}.txt'))
        if model.variable:
            with monkeypatch.context() as patch:
                patch.setattr(model.mode,'neff',lambda omega,**kwargs:neff[model.grid.sidx])
                patch.setattr(model.mode,'dispersion',lambda *args,**kwargs:beta1)
                patch.setattr(model.mode,'beta',lambda *args,**kwargs:beta0)
                errors[f'linear-same-input-{i}']=relative(model.linear_operator(z),data[:,2]+1j*data[:,3])
    print(name,'setup',errors)
    assert max(errors.values())<1e-13
    samples=field(root/'linear-samples.txt');positions=np.loadtxt(root/'linear-positions.txt')
    transferred=dict(zip(positions,samples.T))
    short=solve_precon(model.rhs,lambda z:transferred[z],initial,STEP,dt=STEP,min_dt=STEP,max_dt=STEP,
                      rtol=1e-9,atol=1e-12,saveN=7,locextrap=kw['locextrap'],step_filter=model.window)
    np.testing.assert_array_equal(short.z,np.loadtxt(root/'interval-z.txt'))
    error=relative(short.field,field(root/'interval.txt'));print(name,'interval',error)
    assert error<1e-13
    for fixed in (True,False):
        result=prop_capillary(*args,**kw,backend='python',init_dz=STEP,min_dz=STEP if fixed else 1e-15,
                              max_dz=STEP,rtol=1e-9,atol=1e-12)
        suffix='fixed' if fixed else 'adaptive'
        np.testing.assert_array_equal(result.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(result.field,field(root/f'{suffix}.txt'));print(name,suffix,error)
        assert error<1e-6
    if (root/'entrypoint.txt').is_file():
        result=prop_capillary(*args,**kw,backend='auto')
        error=relative(result.field,field(root/'entrypoint.txt'));print(name,'entrypoint',error)
        assert error<1e-6 and result.metadata['backend']=='python'


def test_profile_oracle_effects(oracle):
    for grid in ('env','real'):
        base=field(oracle/f'Ar-{grid}-base'/'adaptive.txt')
        for kind in ('rise','fall','multi','pressure','taper','both'):
            effect=relative(field(oracle/f'Ar-{grid}-{kind}'/'adaptive.txt'),base)
            print('Ar',grid,kind,'effect',effect);assert effect>1e-5
        for kind in ('flat-pressure','flat-radius'):
            assert relative(field(oracle/f'Ar-{grid}-{kind}'/'adaptive.txt'),base)<1e-6
    for base,control in [('N2-real','N2-real-no-raman'),('N2-real','N2-real-plasma'),
                         ('H2-real','H2-real-plasma'),('N2-real','N2-real-rotation'),
                         ('N2-real','N2-real-hot'),('N2-real','N2-real-no-kerr'),
                         ('N2-real','N2-real-no-thg')]:
        effect=relative(field(oracle/control/'adaptive.txt'),field(oracle/base/'adaptive.txt'))
        print(base,control,'effect',effect);assert effect>1e-5


@pytest.mark.parametrize('name,gas,Z,P',[
    ('rise','Ar',[0.,LENGTH],[0.,4.]),('multi','H2',[0.,.00007,LENGTH],[2.,4.,1.]),
    ('equal','N2',[0.,LENGTH],[2.,2.])])
def test_pressure_gradient_spline_oracle(oracle,name,gas,Z,P):
    profile=_GasProfile(gas,(Z,P),LENGTH,materials.ROOMTEMP)
    reference=np.loadtxt(oracle/f'gradient-{name}'/'spline.txt')
    samples=np.loadtxt(oracle/f'gradient-{name}'/'samples.txt')
    assert relative(profile.spline.x,reference[:,0])<1e-13
    assert relative(profile.spline.y,reference[:,1])<1e-13
    same=_NormalizedCubic(reference[:,0],reference[:,1])
    assert relative(same.derivative,reference[:,2])<1e-13
    pressure=np.array([profile.pressure(z) for z in samples[:,0]])
    density=np.array([profile.density(z) for z in samples[:,0]])
    errors={'pressure':relative(pressure,samples[:,1]),'density':relative(density,samples[:,2])}
    print(name,'gradient',errors);assert max(errors.values())<1e-13
    if name=='rise':
        assert profile.density(0)==0 and profile.pressure(-1)==0 and profile.pressure(1)==4


def test_density_spline_refinement():
    pressures=np.r_[np.linspace(0,4,137),1e-5,1e-4,3.9999]
    exact=materials.density('Ar',pressures)
    errors=[]
    for n in (256,1024,4096):
        nodes=_range_positions(0.,4.,n)
        spline=_NormalizedCubic(nodes,materials.density('Ar',nodes))
        errors.append(relative(spline(pressures),exact))
    print('density spline refinement',errors)
    assert errors[-1]<1e-6 and errors[-1]<.1*errors[0]


@pytest.mark.parametrize('envelope',[False,True])
def test_profile_callbacks_positions_ownership_and_backend(envelope,tmp_path):
    seen=[]
    def radius(z):
        seen.append((z,threading.get_ident()))
        return 125e-6*(1+.03*z/LENGTH)
    positions=np.array([0.,.00007,LENGTH]);pressure=np.array([2.,4.,1.])
    options=BASE|dict(envelope=envelope,init_dz=STEP,min_dz=STEP,max_dz=STEP)
    reference=prop_capillary(radius,LENGTH,'N2',(positions,pressure),**options,backend='python')
    actual=prop_capillary(radius,LENGTH,'N2',(positions,pressure),**options)
    assert relative(actual.field,reference.field)<1e-13
    assert actual.metadata['backend']=='python' and 'profiles' in actual.metadata['backend_reason']
    assert any(0<z<STEP for z,_ in seen) and any(z>LENGTH for z,_ in seen)
    assert all(thread==threading.get_ident() for _,thread in seen)
    count=len(seen)
    with pytest.raises(NotImplementedError,match='profiles'):
        prop_capillary(radius,LENGTH,'N2',(positions,pressure),**options,backend='native')
    assert len(seen)==count
    positions[:]=10;pressure[:]=20
    actual.save_npz(tmp_path/'profile.npz')
    with np.load(tmp_path/'profile.npz',allow_pickle=False) as archive:
        parameters=json.loads(str(archive['parameters']))
        assert parameters['pressure'][1]==[2.,4.,1.] and 'callable' in parameters['radius']
        np.testing.assert_array_equal(archive['Eomega'],actual.field)


@pytest.mark.parametrize('parameter', ['radius','pressure'])
def test_profile_stage_exception_identity(parameter):
    marker=LookupError('original profile exception')
    def callback(z):
        if z>0:raise marker
        return 125e-6 if parameter=='radius' else 2.
    radius,pressure=(callback,2.) if parameter=='radius' else (125e-6,callback)
    for _ in range(3):
        with pytest.raises(LookupError) as caught:
            prop_capillary(radius,LENGTH,'Ar',pressure,**BASE)
        assert caught.value is marker


@pytest.mark.parametrize('parameter,bad',[(parameter,bad) for parameter in ('radius','pressure')
    for bad in ([1.],complex(1),np.nan,np.inf,-1.)]+[('radius',0.)])
def test_invalid_profile_callback_results(parameter,bad):
    radius,pressure=(lambda z:bad,2.) if parameter=='radius' else (125e-6,lambda z:bad)
    with pytest.raises(ValueError):
        prop_capillary(radius,LENGTH,'Ar',pressure,**BASE)


@pytest.mark.parametrize('pressure',[(0.,0.),(1.,-1.),(1.,np.inf),(1.,2.,3.),
    ([0.,0.],[1.,2.]),([0.,2.,1.],[1.,2.,3.]),([0.,1.],[1.]),([0.,1.],[1.,complex(2)])])
def test_invalid_gradient(pressure):
    with pytest.raises(ValueError):
        prop_capillary(125e-6,LENGTH,'Ar',pressure,**BASE)


def test_profile_teardown_and_unsampled_shape():
    class Pressure:
        def __call__(self,z):return 2.+.1*math.sin(z/LENGTH)
    for _ in range(5):
        profile=Pressure();ref=weakref.ref(profile)
        result=prop_capillary(125e-6,LENGTH,'Ar',profile,**BASE)
        del profile
        gc.collect()
        assert ref() is None and result.field.flags.owndata
    model=_GasProfile('Ar',lambda z:2.+z*z,LENGTH,materials.ROOMTEMP)
    for z in (.00000123,.137,.719):
        assert model.pressure(z)==2.+z*z
        assert model.density(z)==materials.density('Ar',2.+z*z)


@pytest.mark.parametrize('envelope',[False,True])
def test_profile_loss_clamp_and_zero_density(oracle,envelope):
    options=BASE|dict(envelope=envelope,kerr=False,energy=1e-15,init_dz=.00005,
                     min_dz=.00005,max_dz=.00005,backend='python')
    results=[]
    for variable in (False,True):
        radius=(lambda z:1e-6) if variable else 1e-6
        model=_Capillary(radius,LENGTH,'Ar',2.,**options,raman=False)
        name=f'{"env" if envelope else "real"}-{"variable" if variable else "constant"}'
        operator=model.linop(0.) if variable else model.linop
        limit=3000 if variable else 1500
        assert np.count_nonzero(operator.real==-limit)>10
        reference=field(oracle/'limits'/f'{name}-linop.txt')[:,0]
        np.testing.assert_allclose(operator.real,reference.real,rtol=1e-13,atol=0)
        result=prop_capillary(radius,LENGTH,'Ar',2.,**options,raman=False)
        error=relative(result.field,field(oracle/'limits'/f'{name}.txt'))
        print(name,'loss clamp solve',error);assert error<1e-6
        results.append(result.field)
    effect=relative(results[1],results[0]);print(envelope,'clamp effect',effect);assert effect>1e-5
    zero=_Capillary(125e-6,LENGTH,'N2',(0.,2.),**options,raman=True)
    name=f'zero-{"env" if envelope else "real"}'
    np.testing.assert_array_equal(zero.rhs(0.,zero.initial),field(oracle/'limits'/f'{name}-rhs.txt')[:,0])
    result=prop_capillary(125e-6,LENGTH,'N2',(0.,2.),**options,raman=True)
    error=relative(result.field,field(oracle/'limits'/f'{name}.txt'))
    print(name,'solve',error);assert error<1e-6
    vacuum=prop_capillary(125e-6,LENGTH,'N2',lambda z:0.,**options,raman=True)
    assert np.all(np.isfinite(vacuum.field))
    with pytest.raises(ValueError,match='zero-density'):
        prop_capillary(125e-6,LENGTH,'H2',(0.,2.),**options,raman=True)
