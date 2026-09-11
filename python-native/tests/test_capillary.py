import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from amalthea_native import prop_capillary, solve_precon, GaussPulse, _native
from amalthea_native.capillary import _Capillary
from amalthea_native.pulses import energy_t

BASE=dict(lambda0=800e-9,lambda_lims=(400e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=10e-6,envelope=True,saveN=7)


def relative(a,b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


@pytest.fixture
def oracle():
    root=os.environ.get('AMALTHEA_CAPILLARY_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_CAPILLARY_ORACLE for independent capillary acceptance')
    return Path(root)


def complex_field(path):
    data=np.loadtxt(path); count=data.shape[1]//2
    return data[:,:count]+1j*data[:,count:]


@pytest.mark.parametrize('name',['base','reduced','no-loss','no-kerr','he12','fine','molecular-kerr'])
@pytest.mark.parametrize('backend',['native','python'])
def test_independent_julia_capillary(oracle,name,backend):
    root=oracle/name
    kw=tomllib.loads((root/'parameters.toml').read_text())
    radius,length,gas,pressure=[kw.pop(k) for k in ('radius','flength','gas','pressure')]
    kw['backend']=backend
    model=_Capillary(radius,length,gas,pressure,**kw)
    data=np.loadtxt(root/'setup.txt')
    initial=data[:,0]+1j*data[:,1]; linop=data[:,2]+1j*data[:,3]
    reference_rhs=data[:,4]+1j*data[:,5]
    material=np.loadtxt(root/'material.txt')
    rhs=(model.rhs(0.,initial) if backend=='python' else
         np.array(_native.envelope_rhs(linop.tolist(),initial.tolist(),model.native_config())))
    errors={'input':relative(model.initial,initial), 'beta':relative(model.beta,data[:,6]),
            'rhs':relative(rhs,reference_rhs),'density':abs(model.density/material[0]-1),
            'area':abs(model.area/material[1]-1),
            'energy':abs(energy_t(model.grid,np.fft.ifft(model.initial))/material[2]-1)}
    print(name,backend,'setup',errors)
    assert max(errors.values())<1e-13
    # Identical initial field/operator isolates stepping from cancellation in
    # the independently prepared moving-frame subtraction.
    short=_Capillary(radius,.001,gas,pressure,**kw,
                             init_dz=.001,min_dz=.001,max_dz=.001,rtol=1e-9,atol=1e-12)
    short.initial=initial.copy(); short.linop=linop.copy()
    interval=(short.solve_native(.001) if backend=='native' else
              solve_precon(short.rhs,linop,initial,.001,dt=.001,min_dt=.001,max_dt=.001,
                           rtol=1e-9,atol=1e-12,saveN=7,step_filter=short.window))
    np.testing.assert_array_equal(interval.z,np.loadtxt(root/'interval-z.txt'))
    error=relative(interval.field,complex_field(root/'interval.txt'))
    print(name,backend,'same-input interval',error)
    assert error<1e-13
    for fixed in (False,True):
        suffix='fixed' if fixed else 'adaptive'
        result=prop_capillary(radius,length,gas,pressure,**kw,init_dz=.001,
                              min_dz=.001 if fixed else 1e-15,max_dz=.001,rtol=1e-9,atol=1e-12)
        np.testing.assert_array_equal(result.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(result.field,complex_field(root/f'{suffix}.txt'))
        print(name,backend,suffix,error)
        assert error<1e-6
        assert result.metadata['backend']==backend
    if name=='base':
        result=prop_capillary(radius,length,gas,pressure,**kw)
        error=relative(result.field,complex_field(root/'entrypoint.txt'))
        print(backend,'public capillary entrypoint',error)
        assert error<1e-6


def test_oracle_capillary_effects(oracle):
    base=complex_field(oracle/'base/adaptive.txt')
    for name in ('no-kerr','no-loss','he12','molecular-kerr'):
        effect=relative(complex_field(oracle/name/'adaptive.txt'),base)
        print('Julia capillary',name,'effect',effect)
        assert effect>1e-5  # At least ten times the complete-trajectory tolerance.


def test_capillary_native_path_aliases_pulses_and_output(monkeypatch,tmp_path):
    reference=prop_capillary(125e-6,.02,'Ar',2.,**BASE,backend='python')
    def fail(*args,**kwargs): raise AssertionError('Python stage called on native path')
    monkeypatch.setattr(_Capillary,'rhs',fail)
    monkeypatch.setattr(_Capillary,'window',fail)
    alias=BASE.copy()
    for ascii,unicode in [('lambda0','λ0'),('lambda_lims','λlims'),('tau_fwhm','τfwhm')]:
        alias[unicode]=alias.pop(ascii)
    for backend in ('auto','native'):
        result=prop_capillary(125e-6,.02,'Ar',2.,**alias,backend=backend)
        assert relative(result.field,reference.field)<1e-13
        assert result.field.flags.owndata and result.metadata['stepper']=='rust-resident'
    pulse=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=10e-6)
    rich=prop_capillary(125e-6,.02,'Ar',2.,**(BASE|dict(pulses=[pulse])))
    assert relative(rich.field,reference.field)<1e-13
    supplied=reference.field[:,0].copy(); before=supplied.copy()
    result=prop_capillary(125e-6,.02,'Ar',2.,**(BASE|dict(pulse=supplied,pulse_domain='frequency',
                                                       energy=None,tau_fwhm=None)))
    np.testing.assert_array_equal(supplied,before)
    assert relative(result.field,reference.field)<1e-13
    out=tmp_path/'capillary.npz'; result.save_npz(out)
    with np.load(out,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['Eomega'],result.field)
        assert json.loads(str(saved['parameters']))['gas']=='Ar'
    np.testing.assert_allclose(result.temporal_field(),np.fft.ifft(result.field,axis=0),rtol=1e-15)


@pytest.mark.parametrize('change,exception',[
    ({'backend':'bad'},ValueError),({'shotnoise':True},NotImplementedError),
    ({'plasma':True},NotImplementedError),
    ({'raman':True},ValueError),
    ({'modes':[]},ValueError),({'modes':'TE11'},ValueError),
    ({'polarisation':'unknown'},ValueError),({'loss':1},TypeError),
    ({'raman':'false'},TypeError),({'λ0':800e-9},TypeError),({'unknown':1},TypeError),
])
def test_capillary_explicit_rejections(change,exception):
    with pytest.raises(exception): prop_capillary(125e-6,.02,'Ar',2.,**(BASE|change))


def test_capillary_profile_errors_and_molecular_default():
    def fail(z): raise AssertionError('profile must not be sampled to infer constancy')
    for args in [(fail,.02,'Ar',2.),(125e-6,.02,'Ar',fail)]:
        with pytest.raises(AssertionError,match='profile'):
            prop_capillary(*args,**BASE)
        with pytest.raises(NotImplementedError,match='profiles'):
            prop_capillary(*args,**BASE,backend='native')
    raman=prop_capillary(125e-6,.02,'N2',2.,**BASE)
    result=prop_capillary(125e-6,.02,'N2',2.,**BASE,raman=False)
    assert np.all(np.isfinite(result.field))
    assert relative(raman.field,result.field)>1e-5
