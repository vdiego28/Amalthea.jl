import json
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from amalthea_native import prop_capillary, solve_precon, GaussPulse, _native
from amalthea_native.capillary import _Capillary
from amalthea_native.pulses import energy_t

BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=10e-6,envelope=False,plasma=False,saveN=7)


def relative(a,b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


@pytest.fixture
def oracle():
    root=os.environ.get('AMALTHEA_REAL_CAPILLARY_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_REAL_CAPILLARY_ORACLE for independent capillary acceptance')
    return Path(root)


def complex_field(path):
    data=np.loadtxt(path); count=data.shape[1]//2
    return data[:,:count]+1j*data[:,count:]


@pytest.mark.parametrize('name,backend',[(name,backend) for name in
    ['base','reduced','no-loss','no-kerr','he12','fine','no-thg','fourth']
    for backend in (['python'] if name=='no-thg' else ['native','python'])])
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
         np.array(_native.real_rhs(linop.tolist(),initial.tolist(),model.native_config())))
    errors={'input':relative(model.initial,initial), 'beta':relative(model.beta,data[:,6]),
            'rhs':relative(rhs,reference_rhs),'density':abs(model.density/material[0]-1),
            'area':abs(model.area/material[1]-1),
            'energy':abs(energy_t(model.grid,np.fft.irfft(model.initial,n=model.grid.t.size))/material[2]-1)}
    print(name,backend,'setup',errors)
    assert max(errors.values())<1e-13
    # Identical initial field/operator isolates stepping from cancellation in
    # the independently prepared moving-frame subtraction.
    short=_Capillary(radius,.001,gas,pressure,**kw,
                             init_dz=.001,min_dz=.001,max_dz=.001,rtol=1e-9,atol=1e-12)
    short.initial=initial.copy(); short.linop=linop.copy()
    interval=(short.solve_native(.001) if backend=='native' else
              solve_precon(short.rhs,linop,initial,.001,dt=.001,min_dt=.001,max_dt=.001,
                           rtol=1e-9,atol=1e-12,saveN=7,step_filter=short.window,locextrap=kw['locextrap']))
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
    for name in ('no-kerr','no-loss','he12','no-thg'):
        effect=relative(complex_field(oracle/name/'adaptive.txt'),base)
        print('Julia capillary',name,'effect',effect)
        assert effect>1e-5  # At least ten times the complete-trajectory tolerance.


def test_real_capillary_native_output_and_inputs(monkeypatch,tmp_path):
    expected=prop_capillary(125e-6,.02,'Ar',2.,**BASE,backend='python')
    def forbidden(*args): raise AssertionError('Python stage on native path')
    monkeypatch.setattr(_Capillary,'rhs',forbidden)
    monkeypatch.setattr(_Capillary,'window',forbidden)
    aliases=BASE.copy()
    for ascii,unicode in [('lambda0','λ0'),('lambda_lims','λlims'),('tau_fwhm','τfwhm')]:
        aliases[unicode]=aliases.pop(ascii)
    for backend in ['native','auto']:
        actual=prop_capillary(125e-6,.02,'Ar',2.,**aliases,backend=backend)
        assert relative(actual.field,expected.field)<1e-13
        assert actual.metadata['backend']=='native' and actual.metadata['stepper']=='rust-resident'
        assert actual.field.flags.owndata
    time=actual.temporal_field()
    assert time.shape==(actual.grid.t.size,BASE['saveN']) and not np.iscomplexobj(time)
    for domain,pulse in [('time',time[:,0]),('frequency',actual.field[:,0])]:
        before=pulse.copy()
        result=prop_capillary(125e-6,.02,'Ar',2.,**(BASE|dict(pulse=pulse,pulse_domain=domain,
                                                        energy=None,tau_fwhm=None)))
        np.testing.assert_array_equal(pulse,before)
        assert relative(result.field,expected.field)<1e-13
    rich=prop_capillary(125e-6,.02,'Ar',2.,**(BASE|dict(pulses=[
        GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=10e-6)])))
    assert relative(rich.field,expected.field)<1e-13
    path=tmp_path/'carrier.npz';actual.save_npz(path)
    with np.load(path,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['Eomega'],actual.field)
        assert json.loads(str(saved['parameters']))['envelope'] is False
        assert len(saved['omega'])==len(saved['t'])//2+1


def test_real_capillary_thg_selection_and_plasma_default():
    expected=prop_capillary(125e-6,.02,'Ar',2.,**BASE,backend='python',thg=False)
    automatic=prop_capillary(125e-6,.02,'Ar',2.,**BASE,thg=False)
    assert relative(automatic.field,expected.field)<1e-13
    assert automatic.metadata['backend']=='python'
    assert 'thg=False' in automatic.metadata['backend_reason']
    with pytest.raises(NotImplementedError,match='thg=False'):
        prop_capillary(125e-6,.02,'Ar',2.,**BASE,thg=False,backend='native')
    plain=prop_capillary(125e-6,.02,'Ar',2.,**BASE,thg=False,kerr=False,backend='native')
    assert plain.metadata['backend']=='native'
