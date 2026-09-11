import json
import os
from pathlib import Path
import tomllib
import numpy as np
import pytest
from amalthea_native import prop_capillary, solve_precon, IonRateADK, IonRatePPTAccel, IonRatePPT, _native
from amalthea_native.capillary import _Capillary
from amalthea_native.plasma import _PlasmaResponse, _cumtrapz
from amalthea_native.ionisation import ELECTRON,M_E

BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=500e-6,raman=False,saveN=7)
STEP=1e-5

def relative(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)

def complex_field(path):
    data=np.loadtxt(path);n=data.shape[1]//2
    return data[:,:n]+1j*data[:,n:]

@pytest.fixture
def oracle():
    path=os.environ.get('AMALTHEA_PLASMA_ORACLE')
    if path is None:pytest.skip('set AMALTHEA_PLASMA_ORACLE for plasma trajectory acceptance')
    return Path(path)

@pytest.mark.parametrize('name,backend',[(name,backend) for name in
    ['adk','ppt','no-plasma','no-kerr','no-thg','preion','fine','fourth']
    for backend in (['python'] if name=='no-thg' else ['python','native'])])
def test_independent_plasma_capillary(oracle,name,backend):
    root=oracle/name;kw=tomllib.loads((root/'parameters.toml').read_text())
    radius,length,gas,pressure=[kw.pop(k) for k in ('radius','flength','gas','pressure')]
    kw['backend']=backend
    model=_Capillary(radius,length,gas,pressure,**kw)
    data=np.loadtxt(root/'setup.txt');initial=data[:,0]+1j*data[:,1];linop=data[:,2]+1j*data[:,3]
    rhs=(model.rhs(0.,initial) if backend=='python' else np.array(_native.real_rhs(
        linop.tolist(),initial.tolist(),model.native_config(),model.native_plasma_config())))
    errors=dict(input=relative(model.initial,initial),rhs=relative(rhs,data[:,4]+1j*data[:,5]))
    if model.plasma is not None:
        components=np.loadtxt(root/'plasma.txt')
        parts=model.plasma.components(components[:,0])
        for key,actual,expected in zip(('rate','fraction','current'),parts[:3],components[:,1:4].T):
            errors[key]=relative(actual,expected)
        dt=model.plasma.dt
        isolated=_cumtrapz(components[:,3],dt)
        errors['same-current polarization']=relative(isolated,components[:,4])
        current=_cumtrapz(components[:,2]*(ELECTRON**2/M_E)*components[:,0],dt)
        selected=components[:,0]!=0
        current[selected]+=model.plasma.ionpot*components[selected,1]*(1-components[selected,2])/components[selected,0]
        errors['same-fraction current']=relative(current,components[:,3])
        discrepancy=relative(parts[3],components[:,4])
        eps=np.finfo(float).eps;n=len(current);gamma=4*n*eps/(1-4*n*eps)
        bound=_cumtrapz(abs(parts[2]-components[:,3]),dt)+gamma*_cumtrapz(abs(parts[2])+abs(components[:,3]),dt)
        relative_bound=np.linalg.norm(bound)/np.linalg.norm(components[:,4])
        print(name,'conditioned polarization',discrepancy,'bound',relative_bound)
        assert discrepancy<=relative_bound<1e-6
        if name=='adk':
            from decimal import Decimal,localcontext
            with localcontext() as ctx:
                ctx.prec=100;step=Decimal(float(dt));values=[Decimal(float(x)) for x in components[:,3]]
                total=Decimal(0);refined=[0.]
                for left,right in zip(values[:-1],values[1:]):
                    total+=(left+right)*step/2;refined.append(float(total))
            error=relative(isolated,np.array(refined));print('Decimal polarization',error)
            assert error<1e-13
    print(name,backend,'plasma setup',errors);assert max(errors.values())<1e-13
    short=_Capillary(radius,STEP,gas,pressure,**kw,init_dz=STEP,min_dz=STEP,max_dz=STEP,rtol=1e-9,atol=1e-12)
    short.initial=initial.copy();short.linop=linop.copy()
    interval=(short.solve_native(STEP) if backend=='native' else
              solve_precon(model.rhs,linop,initial,STEP,dt=STEP,min_dt=STEP,max_dt=STEP,
                           rtol=1e-9,atol=1e-12,saveN=7,step_filter=model.window,locextrap=kw['locextrap']))
    np.testing.assert_array_equal(interval.z,np.loadtxt(root/'interval-z.txt'))
    error=relative(interval.field,complex_field(root/'interval.txt'))
    print(name,backend,'same-input interval',error);assert error<1e-13
    for fixed in (True,False):
        suffix='fixed' if fixed else 'adaptive'
        result=prop_capillary(radius,length,gas,pressure,**kw,init_dz=STEP,min_dz=STEP if fixed else 1e-15,
                              max_dz=STEP,rtol=1e-9,atol=1e-12)
        np.testing.assert_array_equal(result.z,np.loadtxt(root/f'{suffix}-z.txt'))
        error=relative(result.field,complex_field(root/f'{suffix}.txt'))
        print(name,backend,suffix,error);assert error<1e-6
        assert result.metadata['backend']==backend
    if name in ('adk','ppt'):
        result=prop_capillary(radius,length,gas,pressure,**kw)
        error=relative(result.field,complex_field(root/'entrypoint.txt'))
        print(name,backend,'entrypoint',error);assert error<1e-6


def test_plasma_oracle_effects(oracle):
    adk=complex_field(oracle/'adk/adaptive.txt');ppt=complex_field(oracle/'ppt/adaptive.txt')
    for name,reference in [('no-plasma',adk),('no-kerr',adk),('no-thg',adk),('preion',adk),('no-plasma',ppt)]:
        effect=relative(complex_field(oracle/name/'adaptive.txt'),reference)
        print('plasma effect',name,effect);assert effect>1e-5
    difference=relative(adk,ppt);print('ADK/PPT difference',difference);assert difference>1e-5


def test_plasma_selection_models_and_output(tmp_path):
    options={'cachedir':tmp_path}
    expected=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma='PPT',PPT_options=options)
    assert expected.metadata['backend']=='native'
    expected.save_npz(tmp_path/'ppt.npz')
    for selection in (None,True,IonRatePPTAccel('Ar',800e-9,**options)):
        kw={} if callable(selection) else dict(ppt_options=options)
        result=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=selection,**kw)
        assert relative(result.field,expected.field)<1e-13
    adk=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma='ADK')
    constructed=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=IonRateADK('Ar'))
    assert relative(adk.field,constructed.field)<1e-13
    constructed.save_npz(tmp_path/'plasma.npz')
    with np.load(tmp_path/'plasma.npz',allow_pickle=False) as out:
        assert json.loads(str(out['parameters']))['plasma']['callable']=='IonRateADK'
        np.testing.assert_array_equal(out['Eomega'],constructed.field)
    for gas in ('H2','D2','N2O','CH4','SF6'):
        model=_Capillary(125e-6,.0002,gas,2.,**BASE)
        assert isinstance(model.plasma.ratefunc,IonRateADK)
    custom=IonRatePPT('Ar',800e-9,sum_integral=True)
    model=_Capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=custom)
    assert model.plasma.ratefunc is custom
    table=IonRatePPTAccel.from_samples([1,2,3,4],[1,2,3,4])
    assert _Capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=table).plasma.ionpot==IonRateADK('Ar').ionpot


@pytest.mark.parametrize('change,exception',[
    ({'plasma':'bad'},ValueError),({'plasma':1},TypeError),
    ({'plasma':'PPT','envelope':True},NotImplementedError),
    ({'PPT_options':[]},TypeError),({'plasma':'ADK','PPT_options':{'N':8}},ValueError),
    ({'preionfrac':-.1},ValueError),({'preionfrac':1.1},ValueError),({'preionfrac':float('nan')},ValueError),
    ({'PPT_options':{},'ppt_options':{}},TypeError)])
def test_invalid_plasma_configuration(change,exception):
    with pytest.raises(exception):_Capillary(125e-6,.0002,'Ar',2.,**BASE,**change)


def test_plasma_components_callbacks_and_ownership():
    field=np.array([0.,2e10,-3e10,0.,1e10]);before=field.copy();rate=np.ones(5)*1e13
    calls=[]
    def response(values):
        calls.append(values.copy());values[:]=0;return rate
    plasma=_PlasmaResponse(1e-15,response,2e-18,.01)
    parts=plasma.components(field)
    np.testing.assert_array_equal(field,before);np.testing.assert_array_equal(calls[0],field)
    assert all(x.flags.owndata for x in parts)
    rate[:]=0;assert parts[0][0]==1e13
    for bad in (np.ones(4),np.ones((5,1)),np.ones(5)*np.nan,np.ones(5)*-1,np.ones(5)*1j):
        with pytest.raises(ValueError):_PlasmaResponse(1e-15,lambda x:bad,2e-18)(field)
    exception=RuntimeError('rate callback failed')
    def fail(x):raise exception
    with pytest.raises(RuntimeError) as caught:_PlasmaResponse(1e-15,fail,2e-18)(field)
    assert caught.value is exception
    for bad in ([1,np.nan],[1j,0],[[1,2]],[1]):
        with pytest.raises(ValueError):plasma(bad)
    # Exact zero contributes no division term, while a zero rate retains preionisation.
    zero=_PlasmaResponse(1e-15,lambda x:np.zeros(x.shape),2e-18,1.)
    parts=zero.components(field);np.testing.assert_array_equal(parts[1],np.ones(5))
    assert np.linalg.norm(parts[-1])>0


@pytest.mark.parametrize('backend',['native','python'])
def test_plasma_rejected_trials(oracle,backend):
    root=oracle/'ppt';kw=tomllib.loads((root/'parameters.toml').read_text())
    args=[kw.pop(k) for k in ('radius','flength','gas','pressure')]
    result=prop_capillary(*args,**kw,init_dz=args[1],max_dz=args[1],rtol=1e-12,atol=1e-14,backend=backend)
    np.testing.assert_array_equal(result.z,np.loadtxt(root/'rejections-z.txt'))
    error=relative(result.field,complex_field(root/'rejections.txt'))
    print(backend,'plasma rejected trials',result.metadata['rejected_steps'],'accepted',result.metadata['accepted_steps'],'matched trajectory',error)
    counts=np.loadtxt(root/'rejections-counts.txt')
    print('Julia accepted/rejected',counts)
    assert counts[1]>0
    assert result.metadata['rejected_steps']>0 and error<1e-6
