import os
from pathlib import Path

import numpy as np
import pytest

from amalthea_native import RealGrid, GaussPulse, SechPulse, DataPulse, PropagatedPulse
from amalthea_native.pulses import _analytic_signal, _initial, energy_t, energy_omega


def grid():
    return RealGrid(.1,800e-9,(150e-9,1800e-9),100e-15)


def relative(a,b):
    return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


@pytest.mark.parametrize('name',['gauss','cep-power','sech','multi','data','complex-data','propagated','mixed'])
def test_independent_julia_real_pulses(name):
    path=os.environ.get('AMALTHEA_REAL_PULSE_ORACLE')
    if path is None: pytest.skip('set AMALTHEA_REAL_PULSE_ORACLE for carrier pulse acceptance')
    root=Path(path); g=grid()
    gauss=GaussPulse(lambda0=780e-9,tau_fwhm=4e-15,energy=30e-9)
    cep=GaussPulse(lambda0=800e-9,tau_fwhm=2.5e-15,power=1e7,phi=[1.1,3e-15,1e-30])
    sech=SechPulse(lambda0=880e-9,tau_w=2e-15,energy=20e-9,phi=[.3,-4e-15])
    source=np.loadtxt(root/'source.txt'); omega,I,phase=source.T
    data=DataPulse(omega,I,phase,energy=10e-9,phi=[.3,2e-15,2e-30])
    complex_data=DataPulse(omega,np.sqrt(I)*np.exp(1j*phase),energy=10e-9,phi=[.3,2e-15,2e-30])
    def gain(field,grid): field*=.8*np.exp(-1j*grid.omega*2e-15)
    propagated=PropagatedPulse(gauss,gain)
    cases={'gauss':[gauss],'cep-power':[cep],'sech':[sech],'multi':[gauss,sech],
           'data':[data],'complex-data':[complex_data],'propagated':[propagated],'mixed':[propagated,data]}
    spectrum=sum(p.spectrum(g) for p in cases[name])
    time=np.fft.irfft(spectrum,n=len(g.t)); intensity=abs(_analytic_signal(time))**2
    ref=np.loadtxt(root/name/'spectrum.txt'); time_ref=np.loadtxt(root/name/'time.txt')
    energy_ref=np.loadtxt(root/name/'energy.txt')
    errors={'t':relative(g.t,np.loadtxt(root/'time-axis.txt')),
            'omega':relative(g.omega,np.loadtxt(root/'omega-axis.txt')),
            'spectrum':relative(spectrum,ref[:,0]+1j*ref[:,1]),
            'time':relative(time,time_ref[:,0]),'intensity':relative(intensity,time_ref[:,1]),
            'energy_t':abs(energy_t(g,time)/energy_ref[0]-1),
            'energy_omega':abs(energy_omega(g,spectrum)/energy_ref[1]-1),
            'power':abs(np.max(intensity)/energy_ref[2]-1)}
    print(name,errors)
    assert max(errors.values())<1e-13
    assert relative(time**2,intensity)>.1
    if name=='cep-power':
        assert abs(max(intensity)/1e7-1)<1e-13
        assert abs(max(time**2)/1e7-1)>1e-3


@pytest.mark.parametrize('count',[15,16,31,32])
def test_analytic_signal_dc_nyquist_and_ownership(count):
    phase=2*np.pi*np.arange(count)/count
    field=.3+np.cos(3*phase)
    expected=.3+np.exp(3j*phase)
    if count%2==0:
        field+=.2*(-1.)**np.arange(count)
        expected+=.2*(-1.)**np.arange(count)
    before=field.copy(); actual=_analytic_signal(field)
    np.testing.assert_array_equal(before,field)
    assert actual.flags.owndata
    assert relative(actual,expected)<1e-13


def test_real_array_inputs_projection_shapes_and_normalization():
    g=grid(); spectrum=GaussPulse(lambda0=800e-9,tau_fwhm=4e-15,energy=30e-9).spectrum(g)
    options=dict(pulse_domain='time',pulse=None,tau_fwhm=None,tau_w=None,pulseshape='gauss',
                 lambda0=800e-9,phi=(),energy=None,power=None)
    for domain,value in [('time',np.fft.irfft(spectrum,n=len(g.t))),('frequency',spectrum)]:
        original=value.copy(); actual=_initial(g,options|dict(pulse_domain=domain,pulse=value))
        np.testing.assert_array_equal(value,original)
        assert relative(actual,spectrum)<1e-13
        normalized=_initial(g,options|dict(pulse_domain=domain,pulse=value,energy=11e-9))
        assert abs(energy_t(g,np.fft.irfft(normalized,n=len(g.t)))/11e-9-1)<1e-13
    # FFTW ignores imaginary DC/Nyquist components on real inverse transforms.
    supplied=spectrum.copy(); supplied[[0,-1]]+=2j
    actual=_initial(g,options|dict(pulse_domain='frequency',pulse=supplied))
    assert relative(actual,spectrum)<1e-13
    for value in [np.zeros(len(g.omega)),np.zeros(g.t.shape,dtype=complex)]:
        with pytest.raises(ValueError): _initial(g,options|dict(pulse=value))
    with pytest.raises(ValueError):
        _initial(g,options|dict(pulse_domain='frequency',pulse=np.zeros(len(g.t))))


def test_real_propagator_callbacks():
    g=grid(); before=g.omega.copy(); calls=[]
    def callback(field,copy_grid):
        calls.append(field.shape)
        copy_grid.omega[:]=0
        field*=.5
    p=GaussPulse(lambda0=800e-9,tau_fwhm=4e-15,energy=30e-9,propagator=callback)
    for _ in range(4): assert p.spectrum(g).shape==g.omega.shape
    assert calls==[g.omega.shape]*4
    np.testing.assert_array_equal(before,g.omega)
    boom=RuntimeError('pulse callback failed')
    def fail(*args): raise boom
    with pytest.raises(RuntimeError) as error:
        PropagatedPulse(p,fail).spectrum(g)
    assert error.value is boom
    for invalid in [lambda f,g:np.zeros(len(g.t)),lambda f,g:np.full(f.shape,np.nan)]:
        with pytest.raises(ValueError): PropagatedPulse(p,invalid).spectrum(g)
