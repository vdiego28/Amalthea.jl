"""Independent modal setup, complete point arrays, dense intervals and solves."""
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

from amalthea_native import (Mode,MarcatiliMode,GaussPulse,SechPulse,DataPulse,
                             PropagatedPulse,prop_capillary,solve_precon,EnvGrid,RealGrid)
from amalthea_native.capillary import _Capillary
from amalthea_native.modal import _ModalCapillary
from amalthea_native.plasma import _PlasmaResponse, _cumtrapz
from amalthea_native.ionisation import ELECTRON,M_E

LENGTH=1e-5;STEP=2.5e-6
BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=100e-15,
          tau_fwhm=20e-15,energy=500e-6,plasma=False,raman=False,saveN=7)
KINDS=['radial','radial-linear','radial-fourth','radial-reduced','radial-thg',
       'circular','elliptic','full','full-linear','full-x','raman','raman-off',
       'gradient','mixture','custom','default','average-TE','average-TM','average-HE21']
NAMES=[f'{kind}-{grid}' for grid in ('env','real') for kind in KINDS]
NAMES += [f'{kind}-real' for kind in ('vector-ADK','vector-PPT','vector-preion','vector-gradient',
                                    'vector-plasma-only','full-plasma','mixture-plasma')]


def relative(a,b):return np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-300)


def readfield(path,shape):
    array=np.loadtxt(path,ndmin=2)
    return (array[:,0]+1j*array[:,1]).reshape(shape,order='F')


class EquivalentMode(Mode):
    def __init__(self,wrapped):self.wrapped=wrapped
    def neff(self,omega,*,z=0.):return self.wrapped.neff(omega,z=z)
    def field(self,xy,*,z=0.):return self.wrapped.field(xy,z=z)
    def dimlimits(self,*,z=0.):return self.wrapped.dimlimits(z=z)
    def N(self,*,z=0.):return self.wrapped.N(z=z)


def setup(root):
    cfg=tomllib.loads((root/'parameters.toml').read_text())
    radius=(lambda z:125e-6*(1+.08*math.sin(math.pi*z/LENGTH)+.03*(z/LENGTH)**2)) if cfg['gradient'] else 125e-6
    pressure=(2.,4.) if cfg['gradient'] else cfg['pressure']
    gas=cfg['gas']
    kw=BASE|{key:cfg[key] for key in ('modes','polarisation','envelope','thg','raman','kerr',
                                    'plasma','preionfrac','model','loss','locextrap','radial_integral_rtol')}
    kw['modal_maxevals']=2_000_000  # the independent Julia export uses this refinement budget
    for key in ('modal_components','modal_full'):
        if key in cfg:kw[key]=cfg[key]
    variant=cfg['pulse_variant']
    if variant=='one':
        kw['pulses']=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=500e-6,polarisation=cfg['polarisation'])
    elif variant=='two':
        kw['pulses']=[GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=400e-6,mode='HE11'),
                      SechPulse(lambda0=790e-9,tau_fwhm=17e-15,energy=100e-6,mode='HE12',phi=[.2,12e-15])]
    else:
        kw['pulses']=[GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=500e-6/len(cfg['modes']),mode=mode,
                                phi=[.1*(i+1),i*5e-15]) for i,mode in enumerate(cfg['modes'])]
    for pulse in kw['pulses'] if isinstance(kw['pulses'],list) else [kw['pulses']]:
        pulse.options['energy']*=cfg.get('energy_scale',1.)
    if cfg['custom']:
        kw['modes']=[EquivalentMode(MarcatiliMode(radius,gas,pressure,m=m)) for m in range(1,cfg['modes']+1)]
        # Mode selection by signifier requires mode metadata; use the public one-based indices.
        for i,pulse in enumerate(kw['pulses'],1):pulse.options['mode']=i
    return (radius,LENGTH,gas,pressure),kw,cfg


@pytest.fixture
def oracle():
    value=os.environ.get('AMALTHEA_MODAL_CAPILLARY_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_MODAL_CAPILLARY_ORACLE to the independent Julia fixtures')
    return Path(value)


def fixed_nodes(model,rule):
    def rhs(z,field):
        if isinstance(model,_Capillary):return model.rhs(z,field)
        model._refresh(z);geometry=model.space.at(z);ndim=2 if model.space.full else 1
        half=(geometry.upper[:ndim]-geometry.lower[:ndim])/2
        mid=(geometry.upper[:ndim]+geometry.lower[:ndim])/2
        points=rule[:,:ndim]*half+mid
        if ndim==1:points=np.column_stack((points[:,0],np.zeros(len(points))))
        matrix=geometry.matrix(points)
        polarized=model.point_response(field[None,:,:]@matrix)
        projected=polarized@matrix.transpose(0,2,1)
        weight=rule[:,-1]*np.prod(half)
        if geometry.kind=='polar':weight=weight*points[:,0]*(1 if model.space.full else 2*math.pi)
        return np.sum(projected*weight[:,None,None],axis=0)
    return rhs


def spectrum_from_polarization(model,polarization):
    scale=len(model.grid.t)/model.no
    if model.grid.is_real:transformed=np.fft.rfft(polarization.real,axis=0)[:model.n]
    else:
        transformed=np.fft.fft(polarization,axis=0);half=model.n//2
        transformed=np.concatenate((transformed[:half],transformed[-half:]),axis=0)
    return transformed*scale*model.pre[:,None]


def check_plasma_point(model,root,zi,pi,time,expected_time,expected_p):
    """Isolate the cancellation-prone integration, with a forward-error bound."""
    bound=np.zeros_like(time);isolated=np.zeros_like(time);nonplasma=np.zeros_like(time)
    errors=[]
    for si,response in enumerate(model.responses,1):
        plasma=response.plasma
        try:
            response.plasma=None
            if model.npol==1:
                response.accumulate(isolated[:,0],expected_time[:,0])
                response.accumulate(nonplasma[:,0],time[:,0])
            else:
                response.accumulate(isolated,expected_time)
                response.accumulate(nonplasma,time)
        finally:response.plasma=plasma
        if plasma is None:continue
        rate,fraction=np.loadtxt(root/f'plasma-rate-{zi}-{pi}-{si}.txt').T
        current=np.loadtxt(root/f'plasma-current-{zi}-{pi}-{si}.txt',ndmin=2)
        polarization=np.loadtxt(root/f'plasma-polarization-{zi}-{pi}-{si}.txt',ndmin=2)
        field=time[:,0] if model.npol==1 else time
        parts=plasma.components(field) if model.npol==1 else plasma.vector_components(field)
        reference_field=expected_time[:,0] if model.npol==1 else expected_time
        same_input=plasma.components(reference_field) if model.npol==1 else plasma.vector_components(reference_field)
        actual_current=parts[2].reshape(current.shape)
        errors.append(relative(same_input[0],rate))
        same_current=_cumtrapz(current,plasma.dt)
        errors.append(relative(same_current,polarization))
        if root.name=='vector-ADK-real' and zi==1 and pi==1:
            from decimal import Decimal,localcontext
            with localcontext() as context:
                context.prec=100;step=Decimal(float(plasma.dt));refined=np.zeros_like(current)
                for column in range(model.npol):
                    total=Decimal(0)
                    for row in range(1,model.no):
                        total+=(Decimal(float(current[row-1,column]))+Decimal(float(current[row,column])))*step/2
                        refined[row,column]=float(total)
            errors.append(relative(same_current,refined))
            print('vector plasma Decimal refinement',errors[-1],flush=True)
        same_fraction=_cumtrapz(fraction[:,None]*(ELECTRON**2/M_E)*expected_time,plasma.dt)
        magnitude=np.abs(expected_time[:,0]) if model.npol==1 else np.hypot(*expected_time.T)
        selected=magnitude!=0
        prefactor=plasma.ionpot*rate[selected]*(1-fraction[selected])/magnitude[selected]**2
        same_fraction[selected]+=prefactor[:,None]*expected_time[selected]
        errors.append(relative(same_fraction,current))
        eps=np.finfo(float).eps;gamma=4*model.no*eps/(1-4*model.no*eps)
        integrated=_cumtrapz(parts[0],plasma.dt);reference_integral=_cumtrapz(rate,plasma.dt)
        fraction_bound=np.abs(integrated-reference_integral)+gamma*(integrated+reference_integral)+4*eps
        assert np.all(np.abs(parts[1]-fraction)<=fraction_bound)
        phase=parts[1][:,None]*(ELECTRON**2/M_E)*time
        reference_phase=fraction[:,None]*(ELECTRON**2/M_E)*expected_time
        def loss(values,rates,fractions):
            magnitude=np.abs(values[:,0]) if model.npol==1 else np.hypot(*values.T)
            selected=magnitude!=0;out=np.zeros_like(values)
            out[selected]=(plasma.ionpot*rates[selected]*(1-fractions[selected])/magnitude[selected]**2)[:,None]*values[selected]
            return out
        losses=loss(time,parts[0],parts[1]);reference_losses=loss(expected_time,rate,fraction)
        current_bound=_cumtrapz(np.abs(phase-reference_phase),plasma.dt)+np.abs(losses-reference_losses)
        current_bound+=gamma*(_cumtrapz(np.abs(phase)+np.abs(reference_phase),plasma.dt)+np.abs(losses)+np.abs(reference_losses))
        assert np.all(np.abs(actual_current-current)<=current_bound+np.finfo(float).tiny)
        forward=_cumtrapz(np.abs(actual_current-current),plasma.dt)
        forward+=gamma*_cumtrapz(np.abs(actual_current)+np.abs(current),plasma.dt)
        assert np.all(np.abs(parts[3].reshape(current.shape)-polarization)<=forward+np.finfo(float).tiny)
        bound+=response.density*forward
        isolated+=response.density*polarization
    isolated*=model.grid.towin[:,None]
    errors.append(relative(isolated,expected_p))
    assert max(errors)<1e-13
    # Bound differences in the ordinary response from the two FFT-prepared inputs.
    for response in model.responses:
        plasma=response.plasma
        try:
            response.plasma=None
            if model.npol==1:response.accumulate(nonplasma[:,0],-expected_time[:,0])
            else:response.accumulate(nonplasma,-expected_time)
        finally:response.plasma=plasma
    bound=(bound+np.abs(nonplasma))*model.grid.towin[:,None]
    bound_norm=np.linalg.norm(bound)+1e-13*np.linalg.norm(expected_p)
    assert bound_norm/max(np.linalg.norm(expected_p),1e-300)<1e-6
    return max(errors),bound_norm


@pytest.mark.parametrize('name,backend',[(name,backend) for name in NAMES
    for backend in (('python','native') if name.startswith('average-') else ('python',))])
def test_independent_modal(oracle,name,backend,tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path))
    root=oracle/name;args,kw,cfg=setup(root)
    shape=tuple(cfg['field_shape'])
    constructor=_Capillary if cfg['average'] else _ModalCapillary
    model=constructor(*args,**kw,backend='python')
    initial=readfield(root/'initial.txt',shape)
    metrics=dict(initial=relative(model.initial,initial),linear=[],rhs=[],points=[])
    assert metrics['initial']<1e-13
    for zi in range(1,4):
        z,*densities=np.loadtxt(root/f'density-{zi}.txt')
        expected_linear=readfield(root/f'linear-{zi}.txt',shape)
        actual_linear=model.linop(z) if callable(model.linop) else model.linop
        linear_error=relative(actual_linear,expected_linear);metrics['linear'].append(linear_error)
        assert linear_error<1e-9  # independent finite-difference frame cancellation; exact L below
        actual=model.rhs(z,initial)
        error=relative(actual,readfield(root/f'rhs-{zi}.txt',shape));metrics['rhs'].append(error)
        assert error<3e-7 if not cfg['average'] else error<1e-13
        actual_densities=[r.density for r in model.responses]
        assert relative(actual_densities,densities)<1e-13
        if cfg['average']:continue
        assert model.last_integral.error_norm<=model.last_integral.tolerance
        geometry=model.space.at(z);radius=geometry.upper[0]
        points=np.array([[.13*radius,.2],[.51*radius,.73],[.87*radius,2.1]])
        spectrum=geometry.synthesize(initial,points)
        time=model.to_time(spectrum)
        polarization=model.polarization(time)*model.grid.towin[None,:,None]
        response=model.point_response(spectrum)
        projected=response@geometry.matrix(points).transpose(0,2,1)
        errors=[];conditioned=[]
        for pi in range(1,4):
            expected_time=readfield(root/f'point-time-{zi}-{pi}.txt',(model.no,model.npol))
            expected_p=readfield(root/f'point-polarization-{zi}-{pi}.txt',(model.no,model.npol))
            expected_s=readfield(root/f'point-spectrum-{zi}-{pi}.txt',(model.n,model.npol))
            expected_projection=readfield(root/f'point-projection-{zi}-{pi}.txt',shape)
            errors.append(relative(time[pi-1],expected_time))
            if any(item.plasma is not None for item in model.responses):
                kernel,bound=check_plasma_point(model,root,zi,pi,time[pi-1],expected_time.real,expected_p.real)
                difference=np.linalg.norm(polarization[pi-1]-expected_p)
                assert difference<=bound
                spectral_bound=bound*np.sqrt(model.no)*(len(model.grid.t)/model.no)*max(abs(model.pre))
                assert np.linalg.norm(response[pi-1]-expected_s)<=spectral_bound+1e-13*np.linalg.norm(expected_s)
                isolated_s=spectrum_from_polarization(model,expected_p)
                isolated_projection=expected_s@geometry.matrix(points)[pi-1].T
                errors.extend([kernel,relative(isolated_s,expected_s),relative(isolated_projection,expected_projection)])
                conditioned.append(dict(error=difference/max(np.linalg.norm(expected_p),1e-300),
                                         bound=bound/max(np.linalg.norm(expected_p),1e-300)))
            else:
                errors.extend([relative(polarization[pi-1],expected_p),relative(response[pi-1],expected_s),
                               relative(projected[pi-1],expected_projection)])
        metrics['points'].append(max(errors));assert max(errors)<1e-13
        if conditioned:print('conditioned modal point',name,zi,json.dumps(conditioned),flush=True)
    positions=np.loadtxt(root/'linear-positions.txt')
    operators=readfield(root/'linear-samples.txt',(np.prod(shape),len(positions)))
    transferred={z:operators[:,i].reshape(shape,order='F') for i,z in enumerate(positions)}
    result=solve_precon(fixed_nodes(model,np.loadtxt(root/'rule.txt',ndmin=2)),lambda z:transferred[z],initial,STEP,
                        dt=STEP,min_dt=STEP,max_dt=STEP,rtol=1e-9,atol=1e-12,saveN=7,
                        step_filter=model.window,locextrap=kw['locextrap'])
    assert np.array_equal(result.z,np.loadtxt(root/'interval-z.txt'))
    metrics['interval']=relative(result.field,readfield(root/'interval.txt',(*shape,7)))
    assert metrics['interval']<1e-13
    for fixed in (True,False):
        suffix='fixed' if fixed else 'adaptive'
        controls=tomllib.loads((root/f'{suffix}-controls.toml').read_text())
        # Test the public spatial default against Julia's refined reference.
        run_options=kw if cfg['average'] else kw|dict(radial_integral_rtol=1e-3)
        actual=prop_capillary(*args,**run_options,backend=backend,init_dz=controls['dt'],max_dz=controls['max_dt'],
                              min_dz=STEP if fixed else 1e-15,rtol=controls['rtol'],atol=controls['atol'])
        assert np.array_equal(actual.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(actual.field,readfield(root/f'{suffix}.txt',(*shape,7)));metrics[suffix]=error
        assert error<1e-6
        assert actual.metadata['backend']==backend
        if not cfg['average']:
            assert actual.field.ndim==3 and actual.metadata['quadrature']=='scipy'
            assert actual.metadata['max_quadrature_error_ratio']<=1.
    print('modal',name,backend,json.dumps(metrics),flush=True)


def test_oracle_physics_effects(oracle):
    def field(name):
        root=oracle/name;cfg=tomllib.loads((root/'parameters.toml').read_text())
        return readfield(root/'adaptive.txt',(*cfg['field_shape'],7))
    for grid in ('env','real'):
        for base,control in [('radial','radial-linear'),('radial','radial-thg'),('full','full-linear'),
                             ('raman','raman-off'),('raman','gradient'),('circular','elliptic')]:
            if grid=='env' and control=='radial-thg':
                reference=field('radial-thg-env')
                effect=relative(readfield(oracle/'radial-thg-env/thg-control.txt',reference.shape),reference)
            else:effect=relative(field(f'{control}-{grid}'),field(f'{base}-{grid}'))
            print('modal effect',grid,base,control,effect);assert effect>1e-5
    for base,control in [('circular','vector-ADK'),('circular','vector-PPT'),('vector-ADK','vector-preion'),
                         ('vector-ADK','vector-gradient'),('full','full-plasma'),('mixture','mixture-plasma')]:
        if control=='mixture-plasma':
            reference=field('mixture-plasma-real')
            effect=relative(reference,readfield(oracle/'mixture-plasma-real/plasma-control.txt',reference.shape))
        else:effect=relative(field(f'{control}-real'),field(f'{base}-real'))
        print('modal effect',base,control,effect);assert effect>1e-5


@pytest.mark.parametrize('envelope',[True,False])
def test_modal_array_inputs_output_and_ownership(envelope,tmp_path):
    kw=BASE|dict(envelope=envelope,modes=2,init_dz=STEP,max_dz=STEP)
    reference=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw)
    array=reference.field[:,:,0].copy();before=array.copy()
    supplied=prop_capillary(125e-6,LENGTH,'Ar',2.,**(kw|dict(pulse=array,pulse_domain='frequency',tau_fwhm=None,energy=None)))
    assert relative(supplied.field,reference.field)<1e-13 and np.array_equal(array,before)
    assert supplied.temporal_field().shape==(len(supplied.grid.t),2,7)
    target=tmp_path/'modal.npz';supplied.save_npz(target)
    with np.load(target,allow_pickle=False) as data:
        assert np.array_equal(data['Eomega'],supplied.field)
        assert len(json.loads(str(data['parameters']))['modes'])==2
    time=reference.temporal_field()[:,:,0]
    scaled=prop_capillary(125e-6,LENGTH,'Ar',2.,**(kw|dict(pulse=time,pulse_domain='time',tau_fwhm=None,energy=500e-6)))
    assert relative(scaled.field,reference.field)<1e-13


@pytest.mark.parametrize('envelope',[True,False])
def test_custom_mode_equivalence_and_actual_positions(envelope):
    calls=[]
    class Custom(EquivalentMode):
        def neff(self,omega,*,z=0.):calls.append(z);return super().neff(omega,z=z)
    wrapped=[Custom(MarcatiliMode(125e-6,'Ar',2.,m=m)) for m in (1,2)]
    kw=BASE|dict(envelope=envelope,init_dz=STEP,max_dz=STEP,modal_full=False,modal_components='y')
    # Callable constant radius selects Julia's variable, masked operator too.
    reference=prop_capillary(lambda z:125e-6,LENGTH,'Ar',2.,**kw,modes=2)
    custom=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,modes=wrapped)
    assert relative(custom.field,reference.field)<1e-6
    assert any(0<z<LENGTH for z in calls) and max(calls)>LENGTH
    assert custom.metadata['linear_operator']=='python'
    assert all(isinstance(m,dict) for m in custom.parameters['modes'])


@pytest.mark.parametrize('pol',['x','y',0.,.6,-.6,'circular'])
def test_polarization_energy_and_phase(pol):
    kw=BASE|dict(envelope=True,modes=1,polarisation=pol)
    model=_ModalCapillary(125e-6,LENGTH,'Ar',2.,**kw)
    assert model.initial.shape[1]==2
    norm=np.sum(np.abs(model.initial)**2,axis=0)
    if pol=='x':assert norm[0]>0 and norm[1]==0
    elif pol=='y':assert norm[1]>0 and norm[0]==0
    else:
        eps=1. if pol=='circular' else pol
        assert abs(norm[1]/norm[0]-eps**2)<1e-14
        if eps:
            assert relative(model.initial[:,1],-1j*eps*model.initial[:,0])<1e-13


def test_modal_callbacks_and_native_rejection():
    failure=LookupError('custom modal dispersion')
    class Broken(EquivalentMode):
        def neff(self,*args,**kwargs):raise failure
    modes=[Broken(MarcatiliMode(125e-6,'Ar',2.))]
    with pytest.raises(NotImplementedError,match='quadrature'):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=modes,backend='native')
    with pytest.raises(LookupError) as caught:
        prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=modes,modal_components='y')
    assert caught.value is failure


@pytest.mark.parametrize('change,exception',[
    (dict(modes=[]),ValueError),(dict(modes=0),ValueError),(dict(modes=True),TypeError),
    (dict(modes=['HE11','HE11']),ValueError),(dict(modes=['HE11','HE12'],modal_components='bad'),ValueError),
    (dict(modes=2,polarisation=1.1),ValueError),(dict(modes=2,polarisation=[0.]),ValueError),
    (dict(modes=2,modal_maxevals=1),RuntimeError),
    (dict(modes=2,polarisation='circular',raman=True),NotImplementedError),
    (dict(modes=2,polarisation='circular',thg=False),NotImplementedError),
    (dict(modes=2,envelope=True,plasma=True),NotImplementedError),
    (dict(modes=2,shotnoise=True),NotImplementedError),
])
def test_explicit_modal_errors(change,exception):
    with pytest.raises(exception):prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|change))


def test_vector_plasma_scalar_limit_and_original_rate_exception():
    dt=1e-17;axis=np.arange(128)*dt
    x=3e10*np.sin(2.3e15*axis);rate=lambda e:1e12*np.abs(e/3e10)**4
    model=_PlasmaResponse(dt,rate,2e-18,.003)
    vector=model.vector_components(np.column_stack((x,np.zeros_like(x))))
    scalar=model.components(x)
    for actual,expected in zip(vector[:2],scalar[:2]):assert relative(actual,expected)<1e-13
    for actual,expected in zip(vector[2:],scalar[2:]):
        assert relative(actual[:,0],expected)<1e-13 and np.all(actual[:,1]==0)
    failure=RuntimeError('rate failed')
    def broken(field):raise failure
    with pytest.raises(RuntimeError) as caught:_PlasmaResponse(dt,broken,2e-18).vector(np.ones((128,2)))
    assert caught.value is failure


def test_modal_input_propagator_is_serial_and_owned():
    retained=[];threads=[]
    def propagator(field,grid):
        retained.append((field,field.copy()));threads.append(threading.get_ident())
        field*=np.exp(-1j*.2)
    pulse=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=500e-6,polarisation='circular',propagator=propagator)
    result=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=1,pulses=pulse)))
    assert len(retained)==2 and len(set(threads))==1
    for field,before in retained:assert np.array_equal(field,before*np.exp(-1j*.2))
    assert result.field.shape[1]==2


def test_repeated_modal_construction_and_callback_release():
    for _ in range(5):
        mode=EquivalentMode(MarcatiliMode(125e-6,'Ar',2.));reference=weakref.ref(mode)
        result=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=[mode],modal_components='y',modal_full=False)))
        del mode;gc.collect();assert reference() is None
        assert np.all(np.isfinite(result.field))


@pytest.mark.parametrize('envelope',[False,True])
def test_generic_custom_normalization_and_dispersion(envelope):
    # Duck-typed mode deliberately supplies none of the optional defaults.
    class Custom:
        wrapped=MarcatiliMode(125e-6,'Ar',2.)
        def neff(self,w,*,z=0.):return self.wrapped.neff(w,z=z)
        def field(self,xy,*,z=0.):return self.wrapped.field(xy,z=z)
        def dimlimits(self,*,z=0.):return self.wrapped.dimlimits(z=z)
    kw=BASE|dict(envelope=envelope,modal_full=False,modal_components='y')
    builtin=prop_capillary(lambda z:125e-6,LENGTH,'Ar',2.,**kw,modes=1)
    custom=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,modes=[Custom()])
    assert relative(custom.field,builtin.field)<1e-13


@pytest.mark.parametrize('envelope',[False,True])
def test_modal_data_and_wrapped_pulse_assignment(envelope):
    omega=np.linspace(1e15,4e15,300)
    data=DataPulse(omega,np.exp(-((omega-2.35e15)/2e14)**2),energy=300e-6,mode='HE12')
    called=[]
    def phase(field,grid):called.append(field.copy());return field*np.exp(-.3j)
    wrapped=PropagatedPulse(data,phase)
    model=_ModalCapillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=envelope,modes=2,pulses=wrapped)))
    assert len(called)==1 and np.all(model.initial[:,0]==0)
    assert relative(model.initial[:,1],data.spectrum(model.grid)*np.exp(-.3j))<1e-13
    actual=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=envelope,modes=2,pulses=wrapped)))
    supplied=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=envelope,modes=2,
        pulse=model.initial,pulse_domain='frequency',tau_fwhm=None,energy=None)))
    assert relative(actual.field,supplied.field)<1e-13


def test_complete_modal_array_propagator():
    model=_ModalCapillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=2)))
    before=model.initial.copy();seen=[]
    def swap(field,grid):
        seen.append(field.shape);field[:]=field[:,::-1].copy();grid.omega[:]=1.
    result=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=2,
        pulse=before,pulse_domain='frequency',tau_fwhm=None,energy=None,propagator=swap)))
    assert seen==[before.shape] and np.array_equal(before,model.initial)
    assert relative(result.field[:,:,0],before[:,::-1])<1e-13
    assert np.array_equal(result.grid.omega,model.grid.omega)


@pytest.mark.parametrize('propagator',[lambda f,g:np.ones(3),lambda f,g:f*np.nan])
def test_invalid_complete_modal_array_propagator(propagator):
    model=_ModalCapillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=2)))
    with pytest.raises(ValueError):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,modes=2,
            pulse=model.initial,pulse_domain='frequency',tau_fwhm=None,energy=None,propagator=propagator)))


@pytest.mark.parametrize('method', ['neff','field','N','dimlimits'])
def test_invalid_custom_mode_methods(method):
    class Invalid(EquivalentMode):pass
    bad={'neff':lambda *a,**k:np.nan,'field':lambda *a,**k:np.ones((3,)),
         'N':lambda *a,**k:-1.,'dimlimits':lambda *a,**k:('polar',(0,0),(0,1))}
    setattr(Invalid,method,bad[method])
    with pytest.raises(ValueError):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(modes=[Invalid(MarcatiliMode(125e-6,'Ar',2.))],
                                                                       modal_components='y',modal_full=False)))


def test_scalar_pulse_selection_and_polarisation_precedence():
    pulse=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=500e-6,mode='HE12')
    with pytest.raises(ValueError,match='mode'):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,pulses=pulse)
    actual=prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,pulses=pulse,modes='HE12',polarisation='circular')
    expected=prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,pulses=pulse,modes='HE12')
    assert relative(actual.field,expected.field)<1e-13 and actual.field.ndim==2


def test_mode_mapping_large_index_and_alias():
    pulse=GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=500e-6,
                    mode={'kind':'HE','n':1,'m':10},polarization='linear')
    result=prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=True,pulses=pulse,
                                          modes=[dict(kind='HE',n=1,m=10,ϕ=0.)])))
    assert result.field.shape[1]==1 and result.parameters['modes'][0]['m']==10


@pytest.mark.parametrize('name',['full-env','raman-real','full-plasma-real'])
def test_modal_quadrature_refinement(oracle,name,tmp_path,monkeypatch):
    monkeypatch.setenv('XDG_CACHE_HOME',str(tmp_path))
    root=oracle/name;args,kw,cfg=setup(root);shape=tuple(cfg['field_shape'])
    initial=readfield(root/'initial.txt',shape)
    expected=readfield(root/'rhs-2.txt',shape)
    model=_ModalCapillary(*args,**(kw|dict(radial_integral_rtol=1e-3)))
    coarse=model.rhs(.37*LENGTH,initial)
    model.quadrature.update(rtol=1e-8)
    fine=model.rhs(.37*LENGTH,initial)
    # Different tensor rule and a tighter absolute component budget.
    model.quadrature.update(rule='gk15',rtol=1e-10,maxevals=500000)
    refined=model.rhs(.37*LENGTH,initial)
    errors=[relative(value,expected) for value in (coarse,fine,refined)]
    assert relative(fine,refined)<1e-8 and errors[2]<3e-8
    assert errors[2]<=max(errors[0],1e-13)
    controls=dict(init_dz=STEP,min_dz=STEP,max_dz=STEP,rtol=1e-9,atol=1e-12)
    runs=[prop_capillary(*args,**(kw|controls|dict(radial_integral_rtol=tolerance))) for tolerance in (1e-3,1e-8)]
    expected_field=readfield(root/'fixed.txt',(*shape,7))
    trajectories=[relative(run.field,expected_field) for run in runs]
    assert max(trajectories)<1e-6 and relative(runs[0].field,runs[1].field)<1e-6
    print('modal refinement',name,json.dumps(dict(rhs=errors,trajectories=trajectories)),flush=True)
