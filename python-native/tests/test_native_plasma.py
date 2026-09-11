import copy
import numpy as np
import pytest
from amalthea_native import IonRateADK,IonRatePPT,IonRatePPTAccel,prop_capillary,_native
from amalthea_native.capillary import _Capillary
from amalthea_native.plasma import _PlasmaResponse
from amalthea_native.ionisation import ELECTRON,M_E

BASE=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
          tau_fwhm=20e-15,energy=500e-6,raman=False,saveN=7)

def relative(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)


def native_rhs(response,field):
    plasma,reason=response.native_config(2e25);assert reason is None
    n=len(field);spectrum=np.fft.rfft(field);linop=np.zeros(len(spectrum),complex)
    config=([1+0j]*len(spectrum),[1.]*n,[1.]*n,[1.]*len(spectrum),0.,1.,None,response.dt)
    actual=np.array(_native.real_rhs(linop.tolist(),spectrum.tolist(),config,plasma))
    return actual,config,plasma,linop,spectrum


@pytest.mark.parametrize('average',[False,True])
@pytest.mark.parametrize('occupancy',[1,2])
def test_native_adk_options_and_physical_zero(average,occupancy):
    rate=IonRateADK('Ar',cycle_average=average,occupancy=occupancy)
    field=np.array([0,1e8,-1e10,2e10,4e10,-6e10,2e10,0,0,2e10,4e10,-3e10,1e10,0,0,0.])
    response=_PlasmaResponse(1e-15,rate,rate.ionpot,.01)
    actual,*_=native_rhs(response,field)
    expected=np.fft.rfft(response(field)*2e25)
    error=relative(actual,expected);print('native ADK options',average,occupancy,error)
    assert error<1e-13 and np.linalg.norm(expected)>0


def test_native_nonuniform_ppt_clamp_and_owned_arrays():
    nodes=np.array([1e8,1e9,8e9,1e10,2e10,3e10,5e10])
    rates=np.array([1e-10,0,1e5,1e8,1e12,1e14,1e16])
    table=IonRatePPTAccel.from_samples(nodes,rates)
    field=np.array([0,5e7,-1e8,2e9,1e10,3e10,5e10,8e10,0,-8e10,2e10,0,1e10,0,0,0.])
    response=_PlasmaResponse(1e-15,table,IonRateADK('Ar').ionpot)
    actual,config,plasma,linop,spectrum=native_rhs(response,field)
    expected=np.fft.rfft(response(field)*2e25)
    error=relative(actual,expected);print('native nonuniform PPT',error);assert error<1e-13
    preserved=copy.deepcopy(plasma)
    for _ in range(8):
        again=np.array(_native.real_rhs(linop.tolist(),spectrum.tolist(),config,plasma))
        np.testing.assert_array_equal(again,actual)
    assert plasma==preserved
    zeros=np.zeros_like(field);zero,*_=native_rhs(response,zeros)
    np.testing.assert_array_equal(zero,np.zeros_like(zero))


@pytest.mark.parametrize('name',['ADK','PPT'])
def test_native_plasma_avoids_callbacks(monkeypatch,tmp_path,name):
    extra={'PPT_options':{'cachedir':tmp_path}} if name=='PPT' else {}
    expected=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=name,backend='python',**extra)
    def forbidden(*args):raise AssertionError('Python callback on resident plasma path')
    monkeypatch.setattr(_Capillary,'rhs',forbidden)
    monkeypatch.setattr(_Capillary,'window',forbidden)
    monkeypatch.setattr(_PlasmaResponse,'components',forbidden)
    for backend in ('native','auto'):
        actual=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=name,backend=backend,**extra)
        assert actual.metadata['backend']=='native' and actual.metadata['stepper']=='rust-resident'
        assert relative(actual.field,expected.field)<1e-13


def test_native_plasma_explicit_fallbacks():
    class CustomADK(IonRateADK):
        def __call__(self,field):return 1.1*super().__call__(field)
    rates=[IonRateADK('Ar',threshold=False),IonRatePPT('Ar',800e-9,sum_integral=True),CustomADK('Ar'),
           IonRatePPTAccel.from_samples([1e-200,2e-200,3e-200,4e-200],[1e8,2e8,4e8,8e8]),
           IonRatePPTAccel.from_samples([1e200,2e200,3e200,4e200],[1e8,2e8,4e8,8e8])]
    for rate in rates:
        model=_Capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=rate)
        assert not model.native_eligible and 'Python' in model.backend_reason
        with pytest.raises(NotImplementedError,match='Python'):
            _Capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=rate,backend='native')
    custom=rates[2]
    actual=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=custom)
    expected=prop_capillary(125e-6,.0002,'Ar',2.,**BASE,plasma=custom,backend='python')
    assert actual.metadata['backend']=='python' and relative(actual.field,expected.field)<1e-13


def test_private_native_plasma_validation():
    response=_PlasmaResponse(1e-15,IonRateADK('Ar'),IonRateADK('Ar').ionpot)
    _,config,plasma,linop,spectrum=native_rhs(response,np.ones(16)*2e10)
    invalid=[(0,'other'),(1,[1.]*6),(2,[1.]),(3,[1.]),(4,0.),(5,float('nan')),(6,-.1),(7,-1.)]
    for i,value in invalid:
        bad=list(plasma);bad[i]=value
        with pytest.raises(ValueError):_native.real_rhs(linop.tolist(),spectrum.tolist(),config,tuple(bad))
    for badvalues in ([1,1,1,1,1,0,1],[1,1,1,1,1,float('inf'),1]):
        bad=list(plasma);bad[1]=badvalues
        with pytest.raises(ValueError):_native.real_rhs(linop.tolist(),spectrum.tolist(),config,tuple(bad))
    for fields,logs,derivatives in [([1,2,3],[0,1,2],[1,1,1]),([1,2,3,4],[0]*3,[1]*4),
                                   ([1,2,3,4],[0]*4,[np.nan]*4),([1,2,2,4],[0]*4,[1]*4)]:
        bad=('PPT',fields,logs,derivatives,*plasma[4:])
        with pytest.raises(ValueError):_native.real_rhs(linop.tolist(),spectrum.tolist(),config,bad)
