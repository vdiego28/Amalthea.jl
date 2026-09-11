"""Carrier resident lifecycle, including independently solvable cubic dynamics."""
import copy

import numpy as np
import pytest

from amalthea_native import _native, solve_precon


def relative(a,b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


def setup(windows=False):
    n=16
    time=.5+.1*np.cos(2*np.pi*np.arange(n)/n)
    initial=np.fft.rfft(time)
    linop=np.full(initial.size,-.08,dtype=complex)
    twin=np.linspace(.94,1.,n) if windows else np.ones(n)
    owin=np.linspace(1.,.97,initial.size) if windows else np.ones(initial.size)
    config=([1+0j]*initial.size,np.ones(n).tolist(),twin.tolist(),owin.tolist(),1.,1.,None,.1)
    def rhs(z,field): return np.fft.rfft(np.fft.irfft(field,n=n)**3)
    def window(z,field): return np.fft.rfft(twin*np.fft.irfft(owin*field,n=n))
    return initial,linop,config,rhs,window


def native_run(initial,linop,config,positions,*,fifth=True,dt=.01,min_dt=1e-14,
               max_dt=.1,rtol=1e-10,max_attempts=10000,repeat_limit=10):
    samples,accepted,rejected,endpoints=_native.solve_real(
        linop.tolist(),initial.tolist(),config,list(positions),dt,rtol,1e-12,
        .9,min_dt,max_dt,fifth,max_attempts,repeat_limit)
    return np.array(samples).T,accepted,rejected,np.array(endpoints)


@pytest.mark.parametrize('fifth',[False,True])
@pytest.mark.parametrize('windows',[False,True])
def test_real_resident_lifecycle(fifth,windows):
    initial,linop,config,rhs,window=setup(windows)
    expected=solve_precon(rhs,linop,initial,.3,dt=.01,min_dt=.01,max_dt=.01,
                         saveN=37,locextrap=fifth,step_filter=window)
    actual,accepted,rejected,endpoints=native_run(initial,linop,config,expected.z,
                                                 fifth=fifth,min_dt=.01,max_dt=.01)
    error=relative(actual,expected.field)
    print('carrier fixed',fifth,windows,error)
    assert error<1e-13
    np.testing.assert_array_equal(endpoints,expected.metadata['accepted_positions'])
    assert accepted==expected.metadata['accepted_steps'] and rejected==0
    if windows:
        plain=native_run(initial,linop,setup()[2],expected.z,fifth=fifth,min_dt=.01,max_dt=.01)[0]
        assert relative(actual,plain)>1e-2


@pytest.mark.parametrize('fifth',[False,True])
def test_real_resident_rejection_restart_stopping(fifth):
    initial,linop,config,rhs,window=setup()
    expected=solve_precon(rhs,linop,initial,.3,dt=.3,max_dt=.3,saveN=19,
                         rtol=1e-10,atol=1e-12,locextrap=fifth,step_filter=window)
    actual,accepted,rejected,endpoints=native_run(initial,linop,config,expected.z,
                                                fifth=fifth,dt=.3,max_dt=.3)
    error=relative(actual,expected.field)
    print('carrier adaptive',fifth,error,'rejected',rejected)
    assert error<1e-6 and rejected>0 and accepted>2
    assert endpoints[-1]>expected.z[-1]
    restarted=native_run(actual[:,9],linop,config,expected.z[9:],fifth=fifth)[0]
    assert relative(restarted,actual[:,9:])<1e-6
    with pytest.raises(RuntimeError,match='maximum step attempts'):
        native_run(initial,linop,config,[0,1],max_attempts=1)
    with pytest.raises(RuntimeError,match='repetition limit'):
        native_run(initial,linop,config,[0,1],dt=1.,max_dt=1.,repeat_limit=0)


@pytest.mark.parametrize('fifth,min_ratio',[(False,20),(True,35)])
def test_real_resident_analytic_dense(fifth,min_ratio):
    initial,linop,config,_,_=setup()
    time=np.fft.irfft(initial,n=16); a=linop[0].real
    errors=[]
    for h in [.4,.2,.1]:
        positions=np.array([0.,.37*h,.999999*h,h])
        actual=native_run(initial,linop,config,positions,fifth=fifth,dt=h,min_dt=h,max_dt=h)[0]
        exact=time[:,None]*np.exp(a*positions)/np.sqrt(1-time[:,None]**2*np.expm1(2*a*positions)/a)
        reference=np.fft.rfft(exact,axis=0)
        errors.append(relative(actual,reference))
        assert relative(reference,initial[:,None]*np.exp(a*positions))>1e-3
    ratios=np.array(errors[:-1])/errors[1:]
    print('carrier analytic dense',fifth,'errors',errors,'ratios',ratios)
    assert np.all(ratios>min_ratio)


def test_real_resident_invalid_configs_and_ownership():
    initial,linop,config,_,_=setup()
    preserved=copy.deepcopy(config)
    for index,value in [(0,[1j]),(1,[1.]),(1,[1.]*17),(2,[1.]),(2,[1.]*15),
                        (3,[1.]),(4,float('nan')),(5,0.),(6,[1.]*15)]:
        bad=list(config);bad[index]=value
        with pytest.raises(ValueError): native_run(initial,linop,tuple(bad),[0,.1])
    for value in [np.full(9,np.nan),np.ones(8)]:
        with pytest.raises(ValueError): native_run(value,linop,config,[0,.1])
    for controls in [dict(dt=0),dict(min_dt=.2,max_dt=.1),dict(rtol=float('nan'))]:
        with pytest.raises(ValueError): native_run(initial,linop,config,[0,.1],**controls)
    for _ in range(4):
        assert np.all(np.isfinite(native_run(initial,linop,config,[0,.1])[0]))
    assert config==preserved
