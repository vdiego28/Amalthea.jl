import os
from pathlib import Path
import tomllib
import numpy as np
import pytest
from amalthea_native import MolecularRaman
from amalthea_native.molecular import AMG,K_B,M_U,HBAR,EPS0,C

CASES=[f'{gas}-{kind}' for gas in ('N2','H2','D2','N2O','CH4','SF6') for kind in ('base','rotation','vibration','hot')]+['H2-range','N2O-short','O2-empty']

def relative(a,b):return np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)

@pytest.fixture
def oracle():
    root=os.environ.get('AMALTHEA_MOLECULAR_ORACLE')
    if root is None:pytest.skip('set AMALTHEA_MOLECULAR_ORACLE for independent molecular Raman acceptance')
    return Path(root)

@pytest.mark.parametrize('name',CASES)
def test_independent_molecular_setup(oracle,name):
    root=oracle/name;settings=tomllib.loads((root/'settings.toml').read_text())
    model=MolecularRaman(np.loadtxt(oracle/'input-time.txt'),**settings)
    parameters=tomllib.loads((root/'parameters.toml').read_text())
    for key,value in parameters.items():
        actual=model.parameters[key]
        if isinstance(value,(str,int)) or value==0:assert actual==value
        else:assert abs(actual/value-1)<1e-13
    grid=np.loadtxt(root/'grid.txt')
    assert relative(model.time,grid[:,0])<1e-13 and relative(model.window,grid[:,1])<1e-13
    rows=(np.loadtxt(root/'oscillators.txt',ndmin=2) if (root/'oscillators.txt').stat().st_size else np.empty((0,5)))
    errors={};rho=np.loadtxt(root/'densities.txt')
    for index,density in enumerate(rho):
        reference=np.loadtxt(root/f'response-{index+1}.txt')
        if not np.all(np.isfinite(reference)):
            assert density==0 and settings['vibration'] and settings['gas'] in ('H2','D2','CH4')
            with pytest.raises(ValueError,match='zero-density'):model(density)
            continue
        actual=model(density)
        errors[f'response-{index}']=relative(actual,reference)
        assert errors[f'response-{index}']<1e-13
        assert actual.flags.owndata and actual[0]==0 and actual[-1]==0
        if index:
            oscillators=model.oscillators(density)
            for key,expected in [('omega',rows[:,0]),('coupling',rows[:,1]),('tau2',rows[:,index+1])]:
                actual=oscillators[key];error=relative(actual,expected);errors[f'{key}-{index}']=error
                assert error<1e-13
                np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=0)
    print(name,'molecular setup',errors)


def test_molecular_constants_and_oracle_rejections(oracle):
    constants=tomllib.loads((oracle/'constants.toml').read_text())
    for key in ('AMG','K_B','M_U','HBAR','EPS0','C'):assert abs(globals()[key]/constants[key]-1)<1e-13
    errors=tomllib.loads((oracle/'errors.toml').read_text())
    for name,error in errors.items():assert error!='unexpected success'
    time=np.loadtxt(oracle/'input-time.txt')
    for opts in [dict(rotation=False),dict(vibration=False)]:
        with pytest.raises(NotImplementedError,match='Julia oracle'):MolecularRaman(time,'O2',**opts)
    with pytest.raises(ValueError,match='guard'):MolecularRaman(time,'H2',minJ=20,maxJ=50)


def test_molecular_effects(oracle):
    for gas in ('N2','H2','D2'):
        base=np.loadtxt(oracle/f'{gas}-base/response-3.txt')
        for option in ('rotation','vibration','hot'):
            effect=relative(np.loadtxt(oracle/f'{gas}-{option}/response-3.txt'),base)
            print(gas,option,'effect',effect);assert effect>1e-5
    for gas in ('N2','H2','D2','CH4'):
        low=np.loadtxt(oracle/f'{gas}-base/response-2.txt');high=np.loadtxt(oracle/f'{gas}-base/response-4.txt')
        effect=relative(high,low);print(gas,'density effect',effect);assert effect>1e-5


def test_molecular_rotor_high_precision():
    from mpmath import mp
    ctx=mp.clone();ctx.dps=100
    model=MolecularRaman(np.arange(64)*1e-15,'N2O',vibration=False,maxJ=8,temperature=350.)
    p=model.parameters
    f=lambda x:ctx.mpf(float(x))
    energy=[2*ctx.pi*f(HBAR)*f(C)*f(p['B'])*j*(j+1) for j in range(9)]
    population=[(2*j+1)*ctx.exp(-e/(f(K_B)*350)) for j,e in enumerate(energy)]
    total=sum(population);population=[x/total for x in population]
    omega=[(energy[j+2]-energy[j])/f(HBAR) for j in range(7)]
    factor=-(4*ctx.pi*f(EPS0))**2*2/15*f(p['delta_alpha'])**2/f(HBAR)
    coupling=[factor*(j+1)*(j+2)/(2*j+3)*(population[j+2]/(2*j+5)-population[j]/(2*j+1)) for j in range(7)]
    result=model.oscillators(AMG)
    errors=[relative(model.rotation_data['population'],np.array(list(map(float,population))))]
    for key,expected in [('omega',omega),('coupling',coupling)]:errors.append(relative(result[key],np.array(list(map(float,expected)))))
    print('100-digit rotor errors',errors);assert max(errors)<1e-13


def test_molecular_owned_data_and_fresh_density():
    time=np.arange(1024)*1e-15;model=MolecularRaman(time,'H2');first=model(AMG);saved=first.copy()
    assert len(model.oscillators(AMG)['omega'])>3
    first[:]=0;time[:]=np.nan
    params=model.parameters;params['B']=0
    group=model.groups(AMG)[0];group['omega'][:]=0
    rotations=model.rotation_data;rotations['population'][:]=0
    model.time[:]=0;model.window[:]=0
    np.testing.assert_array_equal(model(AMG),saved)
    assert relative(model(20*AMG),saved)>1e-5
    for _ in range(4):np.testing.assert_array_equal(MolecularRaman(np.arange(1024)*1e-15,'H2')(AMG),saved)
    for gas in ('N2O','CH4','SF6'):
        empty=MolecularRaman(np.arange(32)*1e-15,gas,rotation=False,vibration=False)
        assert all(len(v)==0 for v in empty.oscillators(0).values())
        np.testing.assert_array_equal(empty(0),np.zeros(32))


@pytest.mark.parametrize('time',[[0],[0,0],[0,1,3],[0,np.nan],[0,1j],[[0,1]]])
def test_invalid_molecular_axes(time):
    with pytest.raises(ValueError):MolecularRaman(time,'N2')

@pytest.mark.parametrize('options,exception',[(dict(rotation=1),TypeError),(dict(vibration='yes'),TypeError),
    (dict(minJ=-1),ValueError),(dict(maxJ=1),ValueError),(dict(minJ=1.5),TypeError),
    (dict(temperature=0),ValueError),(dict(temperature=np.nan),ValueError)])
def test_invalid_molecular_options(options,exception):
    with pytest.raises(exception):MolecularRaman([0,1e-15],'N2',**options)


def test_invalid_molecular_density_and_material():
    for gas in ('Ar','CO2',None):
        with pytest.raises(ValueError):MolecularRaman([0,1e-15],gas)
    model=MolecularRaman([0,1e-15],'N2')
    for density in (-1,np.nan,np.inf,1j,[AMG]):
        with pytest.raises(ValueError):model(density)
