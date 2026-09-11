import json
import os
from pathlib import Path

import numpy as np
import pytest

from amalthea_native import EnvGrid, GaussPulse, SechPulse, DataPulse, PropagatedPulse, prop_gnlse
from amalthea_native.gnlse import _Gnlse
from amalthea_native.pulses import energy_omega

SETTINGS=dict(lambda0=800e-9,lambda_lims=(500e-9,1800e-9),trange=600e-15,
              raman=False,saveN=7)
BETAS=[0.,0.,-20e-27]


def gain(field,grid):
    offset=grid.omega-grid.omega0
    field *= .8*np.exp(-1j*(offset*3e-15+5e-30*offset**2))


def pulses(source=None):
    gauss=GaussPulse(lambda0=780e-9,tau_fwhm=24e-15,energy=30e-12,phi=[.2,-20e-15,1e-29])
    sech=SechPulse(lambda0=880e-9,tau_w=15e-15,power=800.,phi=[1.1,40e-15])
    results={'gauss':gauss,'sech':sech,'multi':[gauss,sech],
             'propagated':PropagatedPulse(gauss,gain)}
    if source is not None:
        omega,intensity,phase=source.T
        data=DataPulse(omega,intensity,phase,energy=20e-12,phi=[.3,5e-15,2e-29])
        results['data']=data
        results['complex-data']=DataPulse(omega,np.sqrt(intensity)*np.exp(1j*phase),energy=20e-12,phi=[.3,5e-15,2e-29])
        results['mixed']=[PropagatedPulse(gauss,gain),data]
    return results


def relative(actual,reference):
    return np.linalg.norm(actual-reference)/np.linalg.norm(reference)


@pytest.mark.parametrize('backend', ['python', 'native'])
def test_julia_pulse_inputs_and_trajectories(backend):
    root=os.environ.get('AMALTHEA_PULSE_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_PULSE_ORACLE for independent pulse acceptance')
    root=Path(root)
    for name,pulse in pulses(np.loadtxt(root/'source.txt')).items():
        model=_Gnlse(.01,.02,BETAS,**SETTINGS,pulses=pulse,backend=backend)
        setup=np.loadtxt(root/name/'setup.txt')
        for label,value,ref in [('field',model.initial,setup[:,0]+1j*setup[:,1]),
                                ('rhs',model.rhs(0,model.initial),setup[:,2]+1j*setup[:,3])]:
            error=relative(value,ref); print(f'{name} {label}: {error:.3e}'); assert error<1e-13
        result=prop_gnlse(.01,.001,BETAS,**(SETTINGS|dict(saveN=3)),pulses=pulse,backend=backend,
                          init_dz=.001,min_dz=.001,max_dz=.001)
        ref=np.loadtxt(root/name/'interval.txt'); error=relative(result.field,ref[:,:3]+1j*ref[:,3:])
        print(f'{name} interval: {error:.3e}'); assert error<1e-13
        result=prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=pulse,backend=backend)
        ref=np.loadtxt(root/name/'full.txt'); error=relative(result.field,ref[:,:7]+1j*ref[:,7:])
        print(f'{name} full: {error:.3e}'); assert error<1e-6
        np.testing.assert_array_equal(result.z,np.loadtxt(root/name/'z.txt'))
    one=np.loadtxt(root/'gauss/full.txt')
    for name in ('multi','propagated'):
        effect=relative(np.loadtxt(root/name/'full.txt'),one)
        print(f'Julia {name} effect: {effect:.3e}'); assert effect>1e-3


def test_coherent_sum_aliases_and_gain_normalization(tmp_path):
    grid=EnvGrid(.02,SETTINGS['lambda0'],SETTINGS['lambda_lims'],SETTINGS['trange'])
    pulse=GaussPulse(λ0=800e-9,τfwhm=20e-15,energy=30e-12,ϕ=[.3])
    # Coherent doubling gives four times the energy, not twice.
    one=_Gnlse(.01,.02,BETAS,**SETTINGS,pulses=pulse)
    both=_Gnlse(.01,.02,BETAS,**SETTINGS,pulses=[pulse,pulse])
    np.testing.assert_array_equal(both.initial,2*one.initial)
    assert energy_omega(grid,both.initial)/energy_omega(grid,one.initial)==pytest.approx(4.)
    modified=PropagatedPulse(pulse,lambda field,grid:2*field)
    np.testing.assert_array_equal(modified.spectrum(grid),2*pulse.spectrum(grid))
    # Top-level simple pulse custom propagator uses the same after-normalization order.
    simple=_Gnlse(.01,.02,BETAS,**SETTINGS,tau_fwhm=20e-15,energy=30e-12,phi=[.3],propagator=gain)
    wrapped=_Gnlse(.01,.02,BETAS,**SETTINGS,pulses=PropagatedPulse(pulse,gain))
    np.testing.assert_array_equal(simple.initial,wrapped.initial)
    result=prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=[pulse,modified])
    result.save_npz(tmp_path/'pulses.npz')
    with np.load(tmp_path/'pulses.npz',allow_pickle=False) as saved:
        metadata=json.loads(str(saved['parameters']))
        assert metadata['pulses'][0]['type']=='GaussPulse'
        assert 'callable' in metadata['pulses'][1]['propagator']


def test_data_ownership_and_interpolation_refinement():
    grid=EnvGrid(.02,SETTINGS['lambda0'],SETTINGS['lambda_lims'],SETTINGS['trange'])
    errors=[]
    for count in (41,81,161):
        omega=np.linspace(1.4e15,3.4e15,count)
        intensity=np.exp(-((omega-2.4e15)/1.5e14)**2)
        before=intensity.copy()
        data=DataPulse(omega,intensity,np.zeros(count),energy=20e-12)
        value=data.spectrum(grid)
        reference_intensity=np.exp(-((grid.omega-2.4e15)/1.5e14)**2)*grid.omega_win
        reference_intensity[(grid.omega<=omega[0])|(grid.omega>=omega[-1])]=0
        ref=np.sqrt(reference_intensity).astype(complex)
        ref*=np.sqrt(20e-12/energy_omega(grid,ref))
        # Compare spectral intensity: centering is independently covered by Julia.
        errors.append(relative(abs(value)**2,abs(ref)**2))
        assert energy_omega(grid,value)==pytest.approx(20e-12,rel=1e-14)
        np.testing.assert_array_equal(intensity,before)
        intensity[:]=0;omega[:]=0
        np.testing.assert_array_equal(data.spectrum(grid),value)
    print('DataPulse intensity interpolation refinement:',errors)
    assert errors[1]<errors[0]/8 and errors[2]<errors[1]/8
    assert errors[-1]<1e-5


def test_input_callback_ownership_exceptions_shapes_and_order():
    retained=[];calls=[]
    grid=EnvGrid(.02,SETTINGS['lambda0'],SETTINGS['lambda_lims'],SETTINGS['trange'])
    def callback(field,copy_grid):
        calls.append(len(retained));retained.append(field)
        copy_grid.omega[:]=0
        return field
    pulse=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,power=1000.,propagator=callback)
    result=prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=[pulse,pulse])
    assert calls==[0,1]
    np.testing.assert_array_equal(result.grid.omega,grid.omega)
    before=result.field.copy();retained[0][:]=0
    np.testing.assert_array_equal(result.field,before)
    error=RuntimeError('original input propagator error')
    def fail(field,grid): raise error
    with pytest.raises(RuntimeError) as caught:
        prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=PropagatedPulse(pulse,fail))
    assert caught.value is error
    for invalid in (lambda field,grid:field[:-1],lambda field,grid:field*np.nan):
        with pytest.raises(ValueError):
            prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=PropagatedPulse(pulse,invalid))
    for _ in range(5):
        assert np.all(np.isfinite(pulse.spectrum(grid)))


def test_pulse_validation():
    with pytest.raises(TypeError): GaussPulse(lambda0=800e-9,λ0=800e-9)
    with pytest.raises(TypeError): GaussPulse(lambda0=800e-9,phi=[0],ϕ=[0])
    with pytest.raises(TypeError): GaussPulse(lambda0=800e-9,unexpected=True)
    for omega,intensity,phase in [([1,2,2,4],[1]*4,[0]*4),([1,2,3,4],[-1]*4,[0]*4),
                                  ([1,2,3,4],[1]*3,[0]*4),([1,2,3,4],[1]*4,[np.nan]*4)]:
        with pytest.raises(ValueError):DataPulse(omega,intensity,phase,energy=1)
    for collection in ([],[object()]):
        with pytest.raises((TypeError,ValueError)):
            prop_gnlse(.01,.02,BETAS,**SETTINGS,pulses=collection)


def test_unordered_data_and_nonoverlapping_source():
    omega=np.array([1.8e15,2.1e15,2.3e15,2.7e15,3.e15])
    intensity=np.array([0.,.3,1.,.4,0.])
    phase=np.array([.2,.5,.3,.1,-.1])
    order=np.array([2,0,4,1,3])
    grid=EnvGrid(.02,SETTINGS['lambda0'],SETTINGS['lambda_lims'],SETTINGS['trange'])
    expected=DataPulse(omega,intensity,phase,energy=1e-12).spectrum(grid)
    actual=DataPulse(omega[order],intensity[order],phase[order],energy=1e-12).spectrum(grid)
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(ValueError,match='no finite energy'):
        DataPulse(omega*10,intensity,phase,energy=1e-12).spectrum(grid)
