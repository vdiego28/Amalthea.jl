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

from amalthea_native import prop_capillary, solve_precon, _native, IonRateADK, IonRatePPTAccel
from amalthea_native.capillary import _Capillary
from amalthea_native import materials

LENGTH=.0002
STEP=.00001
BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=500e-6,plasma=False,saveN=7)
KINDS=['ArNe','ArNe-no-Ne','ArNe-no-kerr','Ar-single','Ar-split','ArNe-fourth',
       'ArNe-reduced','ArNe-thg','ArNe-gradient','ArNe-callable','N2H2','N2H2-no-raman',
       'N2H2-no-kerr','N2H2-components','N2H2-hot','N2H2-gradient','molecular-three']
NAMES=[f'{kind}-{grid}' for grid in ('env','real') for kind in KINDS]
NAMES += [f'N2H2-{kind}-real' for kind in ('plasma','plasma-preion','plasma-N2','plasma-H2',
                                         'plasma-gradient','plasma-nothg','fine')]
NATIVE={'ArNe','ArNe-no-Ne','ArNe-no-kerr','Ar-single','Ar-split','ArNe-fourth','ArNe-reduced','N2H2-no-raman'}


def relative(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


def field(path):
    data=np.loadtxt(path);n=data.shape[1]//2
    return data[:,:n]+1j*data[:,n:]


def pressure(name):
    return dict(zero=0.,one=1.,two=2.,**{'half-Ar':materials.pressure('Ar',materials.density('Ar',2.)/2)},
                rise=(2.,4.),multi=([0.,.00007,LENGTH],[1.,2.,1.5]),
                callable=lambda z:1.+.3*math.sin(math.pi*z/LENGTH)+.1*(z/LENGTH)**2)[name]


def setup(root):
    kw=tomllib.loads((root/'parameters.toml').read_text())
    rname=kw.pop('radius_profile');ps=tuple(pressure(p) for p in kw.pop('pressure_profiles'))
    radius=125e-6 if rname=='constant' else lambda z:125e-6*(1+.1*math.sin(math.pi*z/LENGTH)+.05*(z/LENGTH)**2)
    return (radius,LENGTH,kw.pop('gases'),ps),kw


@pytest.fixture
def oracle():
    value=os.environ.get('AMALTHEA_MIXTURE_CAPILLARY_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_MIXTURE_CAPILLARY_ORACLE for mixture acceptance')
    return Path(value)


@pytest.mark.parametrize('name,backend',[(name,backend) for name in NAMES
    for backend in (('python','native') if name.rsplit('-',1)[0] in NATIVE else ('python',))])
def test_independent_mixture(oracle,name,backend,monkeypatch,tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path))
    root=oracle/name;args,kw=setup(root);model=_Capillary(*args,**kw,backend=backend)
    initial=field(root/'initial.txt')[:,0];errors={'input':relative(model.initial,initial)}
    for i in range(1,5):
        z,area,beta1,beta0,*density=np.loadtxt(root/f'material-{i}.txt')
        data=np.loadtxt(root/f'setup-{i}.txt');operator=data[:,2]+1j*data[:,3]
        if backend=='python':
            actual=model.rhs(z,initial)
        else:
            function=_native.real_rhs if model.grid.is_real else _native.envelope_rhs
            actual=np.asarray(function(operator.tolist(),initial.tolist(),model.native_config()))
        errors[f'rhs-{i}']=relative(actual,data[:,0]+1j*data[:,1])
        errors[f'density-{i}']=relative(model.density,density)
        errors[f'area-{i}']=relative(model.area,area)
        errors[f'beta1-{i}']=relative(model.mode.dispersion(1,model.omega0,z=z),beta1)
        errors[f'beta0-{i}']=relative(model.mode.beta(model.omega0,z=z),beta0)
        neff=data[:,4]+1j*data[:,5]
        errors[f'neff-{i}']=relative(model.mode.neff(model.grid.omega[model.grid.sidx],z=z),neff[model.grid.sidx])
        for j,response in enumerate(model.responses,1):
            if response.h is not None:
                errors[f'raman-{i}-{j}']=relative(response.h,np.loadtxt(root/f'raman-{i}-{j}.txt'))
        if model.variable:
            with monkeypatch.context() as patch:
                patch.setattr(model.mode,'neff',lambda omega,**kwargs:neff[model.grid.sidx])
                patch.setattr(model.mode,'dispersion',lambda *args,**kwargs:beta1)
                patch.setattr(model.mode,'beta',lambda *args,**kwargs:beta0)
                errors[f'linear-same-input-{i}']=relative(model.linear_operator(z),operator)
    print(name,backend,'setup',errors);assert max(errors.values())<1e-13
    if backend=='python':
        operators=field(root/'linear-samples.txt');positions=np.loadtxt(root/'linear-positions.txt')
        transferred=dict(zip(positions,operators.T))
        result=solve_precon(model.rhs,lambda z:transferred[z],initial,STEP,dt=STEP,min_dt=STEP,max_dt=STEP,
                            rtol=1e-9,atol=1e-12,saveN=7,step_filter=model.window,locextrap=kw['locextrap'])
    else:
        short=_Capillary(args[0],STEP,*args[2:],**kw,backend=backend,init_dz=STEP,min_dz=STEP,
                          max_dz=STEP,rtol=1e-9,atol=1e-12)
        short.initial=initial.copy();short.linop=operator.copy()
        result=short.solve_native(STEP)
    np.testing.assert_array_equal(result.z,np.loadtxt(root/'interval-z.txt'))
    error=relative(result.field,field(root/'interval.txt'));print(name,backend,'interval',error);assert error<1e-13
    for fixed in (True,False):
        suffix='fixed' if fixed else 'adaptive'
        dz=STEP
        if not fixed and name=='N2H2-plasma-gradient-real':
            suffix='adaptive-fine';dz=1e-6
        result=prop_capillary(*args,**kw,backend=backend,init_dz=dz,min_dz=dz if fixed else 1e-15,
                              max_dz=dz,rtol=1e-9,atol=1e-12)
        np.testing.assert_array_equal(result.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(result.field,field(root/f'{suffix}.txt'))
        print(name,backend,suffix,error);assert error<1e-6
        assert result.metadata['backend']==backend


def test_mixture_oracle_effects_and_split_equivalence(oracle):
    for grid in ('env','real'):
        for baseline,controls in [('ArNe',['ArNe-no-Ne','ArNe-no-kerr','ArNe-gradient','ArNe-callable','ArNe-thg']),
                                 ('N2H2',['N2H2-no-raman','N2H2-no-kerr','N2H2-components','N2H2-hot','N2H2-gradient'])]:
            base=field(oracle/f'{baseline}-{grid}'/'adaptive.txt')
            for control in controls:
                effect=relative(field(oracle/f'{control}-{grid}'/'adaptive.txt'),base)
                print(grid,baseline,control,'effect',effect);assert effect>1e-5
        error=relative(field(oracle/f'Ar-split-{grid}'/'adaptive.txt'),field(oracle/f'Ar-single-{grid}'/'adaptive.txt'))
        print(grid,'split-gas equivalence',error);assert error<1e-6
    for baseline,control in [('N2H2','N2H2-plasma'),('N2H2','N2H2-plasma-N2'),('N2H2','N2H2-plasma-H2'),
                             ('N2H2-plasma','N2H2-plasma-preion'),('N2H2-plasma','N2H2-plasma-gradient'),
                             ('N2H2-plasma','N2H2-plasma-nothg')]:
        effect=relative(field(oracle/f'{control}-real'/'adaptive.txt'),field(oracle/f'{baseline}-real'/'adaptive.txt'))
        print(baseline,control,'effect',effect);assert effect>1e-5


@pytest.mark.parametrize('envelope',[False,True])
def test_mixture_native_callbacks_and_single_species(envelope,monkeypatch):
    options=BASE|dict(envelope=envelope,raman=False,init_dz=STEP,min_dz=STEP,max_dz=STEP)
    reference=prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,1.),**options,backend='python')
    single=prop_capillary(125e-6,LENGTH,'Ar',2.,**options,backend='python')
    def fail(*args,**kwargs):raise AssertionError('Python stage called on native path')
    for name in ('rhs','polarization','window'):monkeypatch.setattr(_Capillary,name,fail)
    for backend in ('auto','native'):
        actual=prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,1.),**options,backend=backend)
        assert relative(actual.field,reference.field)<1e-13
        assert actual.metadata['backend']=='native'
        one=prop_capillary(125e-6,LENGTH,('Ar',),(2.,),**options,backend=backend)
        assert relative(one.field,single.field)<1e-6


@pytest.mark.parametrize('envelope',[False,True])
def test_species_profiles_and_response_refresh(envelope,tmp_path):
    seen=[]
    def changing(z):
        seen.append((z,threading.get_ident()))
        return 1.+.2*math.sin(math.pi*z/LENGTH)
    ps=[(2.,4.),changing]
    opts=BASE|dict(envelope=envelope,init_dz=STEP,min_dz=STEP,max_dz=STEP)
    model=_Capillary(125e-6,LENGTH,('N2','H2'),ps,**opts)
    h0=model.responses[1].h.copy()
    for z in (.000037,.000137,.00024):
        model.rhs(z,model.initial)
        assert model.density[1]==materials.density('H2',changing(z))
        assert relative(model.responses[1].h,model.density[1]*model.responses[1].raman_response(model.density[1]))<1e-13
    assert relative(model.responses[1].h,h0)>1e-5
    result=prop_capillary(125e-6,LENGTH,('N2','H2'),ps,**opts)
    assert result.metadata['backend']=='python' and 'profiles' in result.metadata['backend_reason']
    assert any(0<z<STEP for z,_ in seen) and any(z>LENGTH for z,_ in seen)
    assert all(thread==threading.get_ident() for _,thread in seen)
    count=len(seen)
    with pytest.raises(NotImplementedError):
        prop_capillary(125e-6,LENGTH,('N2','H2'),ps,**opts,backend='native')
    assert len(seen)==count
    ps[0]=(50.,60.)
    result.save_npz(tmp_path/'mixture.npz')
    with np.load(tmp_path/'mixture.npz',allow_pickle=False) as archive:
        parameters=json.loads(str(archive['parameters']))
        assert parameters['gas']==['N2','H2'] and parameters['pressure'][0]==[2.,4.]
        assert 'callable' in parameters['pressure'][1]
        np.testing.assert_array_equal(archive['Eomega'],result.field)


def test_species_default_models_and_override_ownership(tmp_path):
    overrides=[{'PPT_options':{'cachedir':tmp_path}},{}]
    model=_Capillary(125e-6,LENGTH,('N2','H2'),(2.,1.),**(BASE|dict(plasma=True,species_options=overrides)))
    assert isinstance(model.responses[0].plasma.ratefunc,IonRatePPTAccel)
    assert isinstance(model.responses[1].plasma.ratefunc,IonRateADK)
    assert not model.native_eligible
    assert model.responses[0].plasma.ionpot!=model.responses[1].plasma.ionpot
    overrides[0]['PPT_options']['cachedir']='changed'
    assert model.parameters['species_options'][0]['PPT_options']['cachedir']==str(tmp_path)
    custom=IonRateADK('H2',threshold=False)
    model=_Capillary(125e-6,LENGTH,('N2','H2'),(2.,1.),**BASE,
                       species_options=[{},dict(plasma=custom,preionfrac=.003)])
    assert model.responses[0].plasma is None and model.responses[1].plasma.ratefunc is custom
    assert model.responses[1].plasma.preionfrac==.003


@pytest.mark.parametrize('bad',[(),('Ar','bad'),np.array('Ar'),3])
def test_invalid_gas_collection(bad):
    with pytest.raises(ValueError):prop_capillary(125e-6,LENGTH,bad,(2.,1.),**BASE)


@pytest.mark.parametrize('bad',[2.,(),(2.,),[2.,1.,3.],lambda z:[2.,1.],np.array(2.)])
def test_invalid_mixture_pressure_count(bad):
    with pytest.raises(ValueError):prop_capillary(125e-6,LENGTH,('Ar','Ne'),bad,**BASE)


@pytest.mark.parametrize('bad,exception',[
    ([{}],ValueError),({},ValueError),([{},3],ValueError),([{},dict(unknown=1)],TypeError),
    ([{},dict(plasma='unknown')],ValueError),([{},dict(kerr='false')],TypeError),
    ([{},dict(raman='false')],TypeError),([{},dict(preionfrac=np.nan)],ValueError),
    ([{},dict(PPT_options={'cachedir':'unused'})],ValueError),
    ([{},dict(ppt_options={},PPT_options={})],TypeError)])
def test_invalid_species_options(bad,exception):
    with pytest.raises(exception):
        prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,1.),**BASE,species_options=bad)


@pytest.mark.parametrize('bad',[[1.],complex(1),np.nan,np.inf,-1.])
def test_mixture_profile_invalid_result(bad):
    with pytest.raises(ValueError):
        prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,lambda z:bad),**BASE)


def test_mixture_exceptions_and_teardown():
    marker=LookupError('original species exception')
    class Pressure:
        def __call__(self,z):
            if z>0:raise marker
            return 1.
    for _ in range(4):
        callback=Pressure();reference=weakref.ref(callback)
        with pytest.raises(LookupError) as caught:
            prop_capillary(125e-6,LENGTH,('Ar','Ne'),(2.,callback),**BASE)
        assert caught.value is marker
        # Release traceback references as well as the original callback.
        marker.__traceback__=None
        del caught,callback
        gc.collect();assert reference() is None
    with pytest.raises(NotImplementedError,match='envelope plasma'):
        prop_capillary(125e-6,LENGTH,('Ar','H2'),(2.,1.),**BASE,envelope=True,
                       species_options=[{},dict(plasma=True)])
    with pytest.raises(NotImplementedError,match='lifetime'):
        prop_capillary(125e-6,LENGTH,('Ar','O2'),(2.,1.),**BASE,species_options=[{},dict(raman=True)])
    with pytest.raises(ValueError,match='zero-density'):
        prop_capillary(125e-6,LENGTH,('Ar','H2'),(2.,0.),**BASE)


def test_mixed_plasma_gradient_adaptive_refinement(oracle,monkeypatch,tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path))
    root=oracle/'N2H2-plasma-gradient-real';args,kw=setup(root)
    for suffix,dz in [('adaptive',1e-5),('adaptive-refined',2.5e-6),('adaptive-fine',1e-6)]:
        result=prop_capillary(*args,**kw,backend='python',init_dz=dz,max_dz=dz,rtol=1e-9,atol=1e-12)
        error=relative(result.field,field(root/f'{suffix}.txt'))
        controls=tomllib.loads((root/f'{suffix}-controls.toml').read_text())
        print('mixture adaptive refinement',suffix,'error',error,'Python steps',
              result.metadata['accepted_steps'],result.metadata['rejected_steps'],'Julia steps',
              controls['accepted'],controls['rejected'])
        assert result.metadata['rejected_steps']>0
        if suffix=='adaptive-fine':assert error<1e-6
    default=prop_capillary(*args,**kw)
    error=relative(default.field,field(root/'default.txt'))
    print('mixture default adaptive',error);assert error<1e-6
    assert default.metadata['backend']=='python'
