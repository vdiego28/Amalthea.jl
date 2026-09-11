"""Complete-field callback math, context ownership and solver transport."""
import gc
import json
import os
from pathlib import Path
import threading
import tomllib
import weakref

import numpy as np
import pytest

from amalthea_native import Mode,ResponseContext,prop_capillary,prop_gnlse,solve_precon
from amalthea_native import materials
from amalthea_native.envelope import EPS0
from amalthea_native.gnlse import _Gnlse
from amalthea_native.capillary import _Capillary
from amalthea_native.modal import _ModalCapillary
from amalthea_native.pulses import _analytic_signal
from test_modal import BASE,LENGTH,STEP,setup as modal_setup,readfield,relative


def kerr_response(thg,omega0):
    def response(field,context):
        coefficient=EPS0*sum(materials.gamma3(gas)*density for gas,density in zip(context.gases,context.densities))
        if context.is_real:
            intensity=np.sum(field**2,axis=1) if thg else .75*abs(_analytic_signal(field[:,0]))**2
            return coefficient*field*intensity[:,None]
        coefficient*=.75
        if thg:
            return coefficient*(field*abs(field)**2+np.exp(2j*omega0*context.t)[:,None]*field**3/3)
        if field.shape[1]==1:return coefficient*field*abs(field)**2
        x,y=field.T
        return coefficient*np.column_stack(((abs(x)**2+2/3*abs(y)**2)*x+np.conj(x)*y*y/3,
                                             (abs(y)**2+2/3*abs(x)**2)*y+np.conj(y)*x*x/3))
    return response


@pytest.mark.parametrize('name',['base','no-raman','sio2'])
def test_custom_gnlse_against_independent_julia(name):
    path=os.environ.get('AMALTHEA_GNLSE_ORACLE')
    if path is None:pytest.skip('set AMALTHEA_GNLSE_ORACLE')
    root=Path(path)/name;kw=tomllib.loads((root/'parameters.toml').read_text())
    gamma,length,betas=[kw.pop(k) for k in ('gamma','flength','betas')]
    baseline=_Gnlse(gamma,length,betas,**kw)
    # The impulse response is separately checked against the independent setup.
    kernel=baseline.h
    if kernel is not None:assert relative(kernel,np.loadtxt(root/'raman.txt'))<1e-13
    coefficient=baseline.kerr
    def response(field,context):
        assert isinstance(context,ResponseContext) and context.gases==() and context.densities.size==0
        intensity=abs(field[:,0])**2
        polarization=coefficient*field[:,0]*intensity
        if kernel is not None:
            polarization+=.5*field[:,0]*np.convolve(intensity,kernel)[:len(field)]*(context.t[1]-context.t[0])
        return polarization[:,None]
    options=kw|dict(raman=False,responses=response)
    model=_Gnlse(0.,length,betas,**options)
    data=np.loadtxt(root/'setup.txt');initial=data[:,0]+1j*data[:,1]
    assert relative(model.rhs(0.,initial),data[:,4]+1j*data[:,5])<1e-13
    interval=prop_gnlse(0.,.001,betas,**(options|dict(saveN=3,init_dz=.001,min_dz=.001,max_dz=.001)))
    reference=np.loadtxt(root/'interval.txt')
    assert relative(interval.field,reference[:,:3]+1j*reference[:,3:])<1e-13
    for fixed in (True,False):
        suffix='fixed' if fixed else 'adaptive'
        result=prop_gnlse(0.,length,betas,**options,init_dz=.001,min_dz=.001 if fixed else 1e-15,
                          max_dz=.001,rtol=1e-9,atol=1e-12)
        reference=np.loadtxt(root/f'{suffix}.txt')
        error=relative(result.field,reference[:,:7]+1j*reference[:,7:])
        assert error<(1e-13 if fixed else 1e-6)
        assert result.metadata['backend']=='python' and result.metadata['custom_response_calls']>0
        print('custom GNLSE',name,suffix,error,flush=True)


@pytest.mark.parametrize('envelope',[False,True])
@pytest.mark.parametrize('modes',['HE11',2])
def test_scalar_and_modal_kerr_callback_equivalence(envelope,modes):
    kw=BASE|dict(envelope=envelope,modes=modes,energy=100e-6,init_dz=STEP,max_dz=STEP)
    constructor=_Capillary if isinstance(modes,str) else _ModalCapillary
    baseline=constructor(125e-6,LENGTH,'Ar',2.,**kw)
    response=kerr_response(not envelope,baseline.omega0)
    model=constructor(125e-6,LENGTH,'Ar',2.,**(kw|dict(kerr=False,responses=response)))
    assert relative(model.rhs(.37*LENGTH,model.initial),baseline.rhs(.37*LENGTH,baseline.initial))<1e-13
    reference=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,backend='python')
    actual=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,kerr=False,responses=response)
    assert relative(actual.field,reference.field)<1e-13
    linear=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,kerr=False)
    assert relative(actual.field,linear.field)>1e-5


@pytest.mark.parametrize('name',['radial-env','radial-real','radial-thg-env','radial-thg-real',
    'circular-env','circular-real','full-env','full-real','gradient-env','gradient-real','mixture-env','mixture-real'])
def test_modal_custom_kerr_oracle_and_response_mixtures(name):
    value=os.environ.get('AMALTHEA_MODAL_CAPILLARY_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_MODAL_CAPILLARY_ORACLE')
    root=Path(value)/name;args,kw,cfg=modal_setup(root)
    options=kw|dict(kerr=False,responses=kerr_response(cfg['thg'],2*np.pi*materials.C/kw['lambda0']))
    model=_ModalCapillary(*args,**options)
    initial=readfield(root/'initial.txt',cfg['field_shape'])
    # The callback replaces Kerr while existing per-species Raman remains active.
    assert relative(model.rhs(.37*LENGTH,initial),readfield(root/'rhs-2.txt',cfg['field_shape']))<3e-7
    result=prop_capillary(*args,**(options|dict(init_dz=STEP,max_dz=STEP,rtol=1e-9,atol=1e-12)))
    error=relative(result.field,readfield(root/'adaptive.txt',(*cfg['field_shape'],7)))
    print('custom modal',name,error,flush=True);assert error<1e-6


def test_callback_context_ownership_order_and_profiles(tmp_path):
    held=[];order=[];threads=[];profile_calls=[]
    def pressure(z):profile_calls.append(z);return 2.+z/LENGTH
    def first(field,context):
        order.append(1);threads.append(threading.get_ident());held.append((field,field.copy(),context,context.t.copy()))
        value=field.copy();field[:]=0;context.densities[:]=0;context.t[:]=0
        return value*0
    def second(field,context):
        order.append(2);assert np.any(field) and np.any(context.t)
        assert context.gases==('Ar',) and context.densities[0]>0
        assert context.coordinates is not None and context.coordinate_system=='polar'
        assert context.components==('y',)
        return field*0
    kw=BASE|dict(envelope=True,modes=1,init_dz=STEP,max_dz=STEP,kerr=False)
    result=prop_capillary(125e-6,LENGTH,'Ar',pressure,**kw,responses=[first,second])
    calls=profile_calls.copy();profile_calls.clear()
    reference=prop_capillary(125e-6,LENGTH,'Ar',pressure,**kw)
    assert calls==profile_calls  # Context uses the already-refreshed densities.
    assert np.array_equal(result.field,reference.field)
    assert order==[1,2]*(len(order)//2) and set(threads)=={threading.get_ident()}
    assert max(context.z for _,_,context,_ in held)>LENGTH
    assert all(np.all(field==0) for field,_,_,_ in held)
    assert all(np.any(before) and np.any(t) for _,before,_,t in held)
    result.save_npz(tmp_path/'custom.npz')
    with np.load(tmp_path/'custom.npz',allow_pickle=False) as saved:
        assert len(json.loads(str(saved['parameters']))['responses'])==2


@pytest.mark.parametrize('envelope',[True,False])
@pytest.mark.parametrize('modes',['HE11',1])
@pytest.mark.parametrize('bad',[lambda f,c:np.ones(len(f)),lambda f,c:f*np.nan,lambda f,c:None])
def test_invalid_response_arrays(envelope,modes,bad):
    with pytest.raises(ValueError):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(envelope=envelope,modes=modes,responses=bad)))


def test_callback_errors_native_rejection_and_lifetime():
    failure=LookupError('original custom response error');calls=[]
    class Callback:
        def __call__(self,field,context):calls.append(context.z);raise failure
    callback=Callback();reference=weakref.ref(callback)
    for api,args,kw in [(prop_capillary,(125e-6,LENGTH,'Ar',2.),BASE),
                         (prop_gnlse,(0.,LENGTH,[0,0]),{k:v for k,v in BASE.items() if k!='plasma'})]:
        with pytest.raises(NotImplementedError):api(*args,**kw,responses=callback,backend='native')
        assert calls==[]
        with pytest.raises(LookupError) as caught:api(*args,**kw,responses=callback)
        assert caught.value is failure and len(calls)==1
        calls.clear()
    # The preserved exception owns its traceback, including the callback frame.
    failure.__traceback__=None
    del callback,caught;gc.collect();assert reference() is None


def test_carrier_response_real_validation():
    with pytest.raises(ValueError,match='real'):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,responses=lambda f,c:1j*f)


@pytest.mark.parametrize('value',[3,[lambda f,c:f,None],{'response':lambda f,c:f}])
def test_response_selection_validation(value):
    with pytest.raises(TypeError):prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,responses=value)


@pytest.mark.parametrize('modes',['HE11',1,2])
def test_retained_response_arrays_teardown_and_nested_simulation(modes):
    held=[];references=[]
    class Callback:
        def __call__(self,field,context):
            if not held:
                inner=prop_gnlse(0.,LENGTH,[0,0],**{k:v for k,v in BASE.items() if k!='plasma'})
                assert np.all(np.isfinite(inner.field))
            field[:]=0
            held.append(field)
            return field
    kw=BASE|dict(envelope=True,modes=modes,kerr=False,init_dz=STEP,max_dz=STEP)
    reference=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw)
    results=[]
    for _ in range(3):
        callback=Callback();references.append(weakref.ref(callback))
        result=prop_capillary(125e-6,LENGTH,'Ar',2.,**kw,responses=callback)
        assert result.metadata['backend']=='python'
        assert relative(result.field,reference.field)<1e-13
        del callback;gc.collect();assert all(ref() is None for ref in references)
        results.append((result,result.field.copy()))
    for field in held:field[:]=np.nan
    assert all(np.array_equal(result.field,before) for result,before in results)


@pytest.mark.parametrize('bad',[lambda f,c:np.full(f.shape,'1'),lambda f,c:np.full(f.shape,object()),
                               lambda f,c:np.full(f.shape,np.inf)])
def test_nonnumeric_and_infinite_response_rejected(bad):
    with pytest.raises(ValueError):prop_gnlse(0.,LENGTH,[0,0],**{k:v for k,v in BASE.items() if k!='plasma'},responses=bad)


@pytest.mark.parametrize('modes',['HE11',2])
def test_empty_responses_leave_backend_selection_unchanged(modes):
    reference=prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=modes)
    actual=prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=modes,responses=[])
    assert actual.metadata['backend']==reference.metadata['backend']
    assert np.array_equal(actual.field,reference.field)


@pytest.mark.parametrize('name',['vector-ADK-real','vector-gradient-real','vector-preion-real'])
def test_custom_vector_plasma_complete_arrays(name):
    from scipy.integrate import cumulative_trapezoid
    from amalthea_native import IonRateADK
    from amalthea_native.ionisation import ELECTRON,M_E
    value=os.environ.get('AMALTHEA_MODAL_CAPILLARY_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_MODAL_CAPILLARY_ORACLE')
    root=Path(value)/name;args,kw,cfg=modal_setup(root)
    ratefun=IonRateADK('Ar');potential=materials.ionisation_potential('Ar')
    def response(field,context):
        assert field.shape==(len(context.t),2) and context.is_real
        magnitude=np.hypot(field[:,0],field[:,1]);rate=ratefun(magnitude)
        dt=context.t[1]-context.t[0]
        cumulative=lambda a:cumulative_trapezoid(a,dx=dt,axis=0,initial=0.)
        fraction=cfg['preionfrac']+1-np.exp(-cumulative(rate))
        current=cumulative(fraction[:,None]*(ELECTRON**2/M_E)*field)
        selected=magnitude!=0
        loss=potential*rate[selected]*(1-fraction[selected])/magnitude[selected]**2
        current[selected]+=loss[:,None]*field[selected]
        return context.densities[0]*cumulative(current)
    options=kw|dict(plasma=False,responses=response)
    model=_ModalCapillary(*args,**options);baseline=_ModalCapillary(*args,**kw)
    # Same floating inputs isolate callback transport from plasma cancellation.
    assert relative(model.rhs(.37*LENGTH,model.initial),baseline.rhs(.37*LENGTH,model.initial))<1e-13
    result=prop_capillary(*args,**(options|dict(radial_integral_rtol=1e-3,init_dz=STEP,max_dz=STEP,rtol=1e-9,atol=1e-12)))
    error=relative(result.field,readfield(root/'adaptive.txt',(*cfg['field_shape'],7)))
    assert error<1e-6
    print('custom vector plasma',name,error,flush=True)


class CartesianCallbackMode(Mode):
    """Synthetic orthogonal polynomial fields; use generic normalization."""
    def __init__(self,index,varying=True):self.index=index;self.varying=varying
    def position(self,z):return z if self.varying else 0.
    def radius(self,z):
        z=self.position(z)/LENGTH
        return 125e-6*(1+.08*np.sin(np.pi*z)+.03*z*z)
    def dimlimits(self,*,z=0.):
        a=self.radius(z);return 'cartesian',(-a,-a/2),(a,a/2)
    def field(self,xy,*,z=0.):
        a=self.radius(z);x,y=np.asarray(xy[0])/a,np.asarray(xy[1])/(a/2)
        value=(1-x*x)*(1-y*y)*(1. if self.index==1 else x)
        return np.stack((value,.3*value))
    def neff(self,omega,*,z=0.):
        return np.full(np.shape(omega),1.0001+(self.index-1)*2e-6+1e-7j+self.position(z)/LENGTH*1e-5)


def callback_fixed_nodes(model,rule):
    def rhs(z,field):
        model._refresh(z);geometry=model.space.at(z)
        half=(geometry.upper-geometry.lower)/2;mid=(geometry.upper+geometry.lower)/2
        points=rule[:,:2]*half+mid;matrix=geometry.matrix(points)
        polarized=model.point_response(field[None,:,:]@matrix,points,z=z,coordinate_system=geometry.kind)
        return np.sum((polarized@matrix.transpose(0,2,1))*(rule[:,-1]*np.prod(half))[:,None,None],axis=0)
    return rhs


@pytest.mark.parametrize('envelope',[True,False])
def test_cartesian_custom_modes_responses_and_exact_profiles(envelope):
    from amalthea_native import GaussPulse,SechPulse
    value=os.environ.get('AMALTHEA_CALLBACK_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_CALLBACK_ORACLE')
    root=Path(value)/('cartesian-varying-'+('env' if envelope else 'real'))
    cfg=tomllib.loads((root/'parameters.toml').read_text());shape=cfg['field_shape']
    modes=[CartesianCallbackMode(i) for i in (1,2)]
    pressure=(lambda z:2.+z/LENGTH,lambda z:1.+.4*np.sin(np.pi*z/LENGTH))
    pulses=[GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=300e-6,mode=1),
            SechPulse(lambda0=790e-9,tau_fwhm=17e-15,energy=100e-6,mode=2,phi=[.2,12e-15])]
    base_response=kerr_response(not envelope,2*np.pi*materials.C/800e-9)
    positions=[]
    def response(field,context):
        assert context.coordinate_system=='cartesian' and context.components==('x','y')
        assert context.gases==('Ar','Ne')
        a=modes[0].radius(context.z);x,y=context.coordinates
        assert -a<x<a and -a/2<y<a/2
        positions.append(context.z)
        assert relative(context.densities,[materials.density(g,p(context.z)) for g,p in zip(context.gases,pressure)])<1e-13
        return base_response(field,context)
    kw=BASE|dict(envelope=envelope,kerr=False,modes=modes,pulses=pulses,responses=response,
                modal_components='xy',modal_full=True,radial_integral_rtol=1e-9)
    args=(modes[0].radius,LENGTH,('Ar','Ne'),pressure)
    model=_ModalCapillary(*args,**kw);initial=readfield(root/'initial.txt',shape)
    metrics={'initial':relative(model.initial,initial),'rhs':[],'linear':[],'normalization':[],'rhs_refinement':[]}
    assert metrics['initial']<1e-13
    for zi,z in enumerate((0.,.37*LENGTH,1.2*LENGTH),1):
        metrics['normalization'].append(relative([m.N(z=z) for m in modes],np.loadtxt(root/f'normalization-{zi}.txt')))
        metrics['rhs'].append(relative(model.rhs(z,initial),readfield(root/f'rhs-{zi}.txt',shape)))
        metrics['rhs_refinement'].append(relative(readfield(root/f'rhs-coarse-{zi}.txt',shape),readfield(root/f'rhs-{zi}.txt',shape)))
        errors=np.loadtxt(root/f'quadrature-{zi}.txt')
        assert errors[0]<=errors[1] and errors[2]<=errors[3]
        metrics['linear'].append(relative(model.linear_operator(z),readfield(root/f'linear-{zi}.txt',shape)))
    assert max(metrics['normalization'])<1e-13 and max(metrics['rhs'])<1e-13 and max(metrics['linear'])<1e-9
    assert max(metrics['rhs_refinement'])<1e-9
    zs=np.loadtxt(root/'linear-positions.txt')
    operators=readfield(root/'linear-samples.txt',(np.prod(shape),len(zs)))
    transferred={z:operators[:,i].reshape(shape,order='F') for i,z in enumerate(zs)}
    interval=solve_precon(callback_fixed_nodes(model,np.loadtxt(root/'rule.txt')),lambda z:transferred[z],initial,STEP,
                         dt=STEP,min_dt=STEP,max_dt=STEP,rtol=1e-9,atol=1e-12,saveN=7,step_filter=model.window)
    metrics['interval']=relative(interval.field,readfield(root/'interval.txt',(*shape,7)))
    assert metrics['interval']<1e-13
    for fixed in (True,False):
        suffix='fixed' if fixed else 'adaptive'
        result=prop_capillary(*args,**kw,init_dz=STEP,min_dz=STEP if fixed else 1e-15,max_dz=STEP,rtol=1e-9,atol=1e-12)
        metrics[suffix]=relative(result.field,readfield(root/f'{suffix}.txt',(*shape,7)))
        assert metrics[suffix]<1e-6 and result.metadata['backend']=='python'
    assert max(positions)>LENGTH
    reference=readfield(root/'adaptive.txt',(*shape,7))
    for control in ('linear','constant'):
        other=root.parent/(f'cartesian-{control}-'+('env' if envelope else 'real'))
        assert np.array_equal(initial,readfield(other/'initial.txt',shape))
        metrics[control+'-effect']=relative(reference,readfield(other/'adaptive.txt',(*shape,7)))
        assert metrics[control+'-effect']>1e-5
    print('Cartesian custom callbacks',envelope,json.dumps(metrics),flush=True)
