import json
import os
from pathlib import Path
import tomllib
import numpy as np
import pytest
from amalthea_native import prop_capillary,solve_precon,_native
from amalthea_native.capillary import _Capillary

BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=500e-6,plasma=False,saveN=7)
NAMES=[f'{gas}-{grid}' for gas in ('N2','H2','D2','N2O','CH4','SF6') for grid in ('env','real')]
NAMES += [f'N2-real-{name}' for name in ('off','rotation','vibration','no-kerr','hot','fourth','fine')]
NAMES += ['N2-env-off','N2-env-thg','N2-real-no-thg','N2-real-plasma','H2-real-plasma']
STEP=1e-5

def relative(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)

def complex_field(path):
    data=np.loadtxt(path);n=data.shape[1]//2
    return data[:,:n]+1j*data[:,n:]

@pytest.fixture
def oracle():
    root=os.environ.get('AMALTHEA_RAMAN_CAPILLARY_ORACLE')
    if root is None:pytest.skip('set AMALTHEA_RAMAN_CAPILLARY_ORACLE for Raman capillary acceptance')
    return Path(root)

@pytest.mark.parametrize('name,backend',[(name,backend) for name in NAMES
    for backend in (['python'] if name in ('N2-env-thg','N2-real-no-thg') else ['native','python'])])
def test_independent_raman_capillary(oracle,name,backend):
    root=oracle/name;kw=tomllib.loads((root/'parameters.toml').read_text())
    args=[kw.pop(k) for k in ('radius','flength','gas','pressure')];kw['backend']=backend
    model=_Capillary(*args,**kw)
    data=np.loadtxt(root/'setup.txt');initial=data[:,0]+1j*data[:,1];linop=data[:,2]+1j*data[:,3]
    if backend=='native':
        rhs=(np.array(_native.real_rhs(linop.tolist(),initial.tolist(),model.native_config(),model.native_plasma_config()))
             if model.grid.is_real else np.array(_native.envelope_rhs(linop.tolist(),initial.tolist(),model.native_config())))
    else:rhs=model.rhs(0.,initial)
    errors=dict(input=relative(model.initial,initial),rhs=relative(rhs,data[:,4]+1j*data[:,5]))
    if model.h is not None:errors['raman']=relative(model.h,np.loadtxt(root/'raman.txt'))
    print(name,backend,'setup',errors);assert max(errors.values())<1e-13
    short=_Capillary(args[0],STEP,*args[2:],**kw,init_dz=STEP,min_dz=STEP,max_dz=STEP,rtol=1e-9,atol=1e-12)
    short.initial=initial.copy();short.linop=linop.copy()
    interval=(short.solve_native(STEP) if backend=='native' else
              solve_precon(short.rhs,linop,initial,STEP,dt=STEP,min_dt=STEP,max_dt=STEP,
                           rtol=1e-9,atol=1e-12,saveN=7,step_filter=short.window,locextrap=kw['locextrap']))
    np.testing.assert_array_equal(interval.z,np.loadtxt(root/'interval-z.txt'))
    error=relative(interval.field,complex_field(root/'interval.txt'));print(name,backend,'interval',error)
    assert error<1e-13
    for fixed in (True,False):
        result=prop_capillary(*args,**kw,init_dz=STEP,min_dz=STEP if fixed else 1e-15,max_dz=STEP,rtol=1e-9,atol=1e-12)
        suffix='fixed' if fixed else 'adaptive'
        np.testing.assert_array_equal(result.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(result.field,complex_field(root/f'{suffix}.txt'));print(name,backend,suffix,error)
        assert error<1e-6 and result.metadata['backend']==backend
    if (root/'entrypoint.txt').is_file():
        result=prop_capillary(*args,**kw)
        error=relative(result.field,complex_field(root/'entrypoint.txt'));print(name,backend,'entrypoint',error)
        assert error<1e-6


def test_raman_capillary_effects(oracle):
    for base,control in [('N2-real','N2-real-off'),('N2-env','N2-env-off'),
                         ('N2-real','N2-real-rotation'),('N2-real','N2-real-vibration'),
                         ('N2-real','N2-real-hot'),('N2-real','N2-real-no-kerr'),
                         ('N2-real','N2-real-no-thg'),('N2-env','N2-env-thg'),
                         ('N2-real','N2-real-plasma'),('H2-real','H2-real-plasma')]:
        a=complex_field(oracle/base/'adaptive.txt');b=complex_field(oracle/control/'adaptive.txt')
        effect=relative(b,a);print(base,control,'oracle effect',effect);assert effect>1e-5


@pytest.mark.parametrize('envelope',[False,True])
def test_raman_native_callback_avoidance(monkeypatch,envelope,tmp_path):
    kw=BASE|dict(envelope=envelope)
    expected=prop_capillary(125e-6,.0002,'N2',2.,**kw,backend='python')
    def fail(*args):raise AssertionError('Python Raman callback on native path')
    monkeypatch.setattr(_Capillary,'rhs',fail);monkeypatch.setattr(_Capillary,'polarization',fail)
    monkeypatch.setattr(_Capillary,'window',fail)
    for backend in ('native','auto'):
        actual=prop_capillary(125e-6,.0002,'N2',2.,**kw,backend=backend)
        assert relative(actual.field,expected.field)<1e-13
        assert actual.metadata['backend']=='native'
    actual.save_npz(tmp_path/'raman.npz')
    with np.load(tmp_path/'raman.npz',allow_pickle=False) as data:
        np.testing.assert_array_equal(data['Eomega'],actual.field)
        assert json.loads(str(data['parameters']))['gas']=='N2'


def test_raman_thg_fallback_and_empty_components():
    for envelope,kerr in [(True,True),(False,True),(False,False)]:
        kw=BASE|dict(envelope=envelope,thg=envelope,kerr=kerr)
        actual=prop_capillary(125e-6,.0002,'N2',2.,**kw)
        expected=prop_capillary(125e-6,.0002,'N2',2.,**kw,backend='python')
        assert relative(actual.field,expected.field)<1e-13 and actual.metadata['backend']=='python'
        assert 'THG' in actual.metadata['backend_reason'] or 'thg=False' in actual.metadata['backend_reason']
        with pytest.raises(NotImplementedError):prop_capillary(125e-6,.0002,'N2',2.,**kw,backend='native')
    kw=BASE|dict(envelope=True,kerr=False,thg=True)
    assert prop_capillary(125e-6,.0002,'N2',2.,**kw,backend='native').metadata['backend']=='native'
    for envelope in (False,True):
        empty=prop_capillary(125e-6,.0002,'H2',2.,**BASE,envelope=envelope,rotation=False,vibration=False)
        off=prop_capillary(125e-6,.0002,'H2',2.,**BASE,envelope=envelope,raman=False)
        assert relative(empty.field,off.field)<1e-13


def test_raman_invalid_and_undefined_oracle_combinations():
    with pytest.raises(NotImplementedError,match='Julia oracle'):
        prop_capillary(125e-6,.0002,'O2',2.,**BASE,raman=True)
    with pytest.raises(ValueError,match='zero-density'):
        prop_capillary(125e-6,.0002,'H2',0.,**BASE)
    for key in ('rotation','vibration'):
        with pytest.raises(TypeError):prop_capillary(125e-6,.0002,'N2',2.,**BASE,**{key:1})
    vacuum=prop_capillary(125e-6,.0002,'H2',0.,**BASE,vibration=False)
    assert np.all(np.isfinite(vacuum.field))
