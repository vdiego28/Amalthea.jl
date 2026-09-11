"""Independent Julia setup and high-precision ADK rate controls."""
from decimal import Decimal, localcontext
import math
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from amalthea_native import IonRateADK
from amalthea_native import ionisation as ion
from amalthea_native.materials import ionisation_potential

MATERIALS=['He','HeJ','HeB','Ne','Ar','ArB','Kr','Xe','H','N2','H2','O2','CH4','N2O','SF6','D2']


@pytest.mark.parametrize('material',MATERIALS)
def test_independent_adk_oracle(material):
    path=os.environ.get('AMALTHEA_ADK_ORACLE')
    if path is None: pytest.skip('set AMALTHEA_ADK_ORACLE for independent ADK acceptance')
    root=Path(path)
    constants=tomllib.loads((root/'constants.toml').read_text())
    for name,value in [('electron',ion.ELECTRON),('mass',ion.M_E),('hbar',ion.HBAR),
                       ('energy',ion.AU_ENERGY),('field',ion.AU_EFIELD)]:
        assert abs(value/constants[name]-1)<1e-13
    max_coeff=0.;max_rate=0.;underflow_count=0
    for threshold in [True,False]:
        for average in [True,False]:
            for occupancy in [1,2]:
                case=root/f'{material}-{str(threshold).lower()}-{str(average).lower()}-{occupancy}'
                params=tomllib.loads((case/'parameters.toml').read_text())
                rate=IonRateADK(material,threshold=threshold,cycle_average=average,occupancy=occupancy)
                for name in ['ionpot','nstar','cn_sq','omega_p','omega_t_prefac','avfac']:
                    error=abs(getattr(rate,name)/params[name]-1)
                    max_coeff=max(max_coeff,error)
                    assert error<1e-13
                assert rate.thr==params['thr']
                for unit,key in [('SI','ionpot'),('atomic','potential_atomic'),('eV','potential_eV')]:
                    assert abs(ionisation_potential(material,unit)/params[key]-1)<1e-13
                field,reference=np.loadtxt(case/'rates.txt').T
                actual=rate(field)
                zero=field==0
                assert np.all(actual[zero]==0)
                if not threshold: assert np.all(np.isnan(reference[zero]))
                active=(abs(field)>0)&(abs(field)>=rate.thr)
                magnitude=abs(field[active]);expected=reference[active];observed=actual[active]
                exponent=-4/3*params['omega_p']/(params['omega_t_prefac']*magnitude)
                expfactor=np.exp(exponent)
                resolved=expfactor>=np.finfo(float).tiny
                error=np.max(abs(observed[resolved]/expected[resolved]-1))
                max_rate=max(max_rate,error)
                assert error<1e-13
                # Once the exponential underflows, rate quantization is its
                # spacing amplified by the positive prefactor, not rate ULPs.
                low=~resolved;underflow_count+=np.count_nonzero(low)
                prefactor=(occupancy*params['omega_p']*params['cn_sq']*
                           (4*params['omega_p']/(params['omega_t_prefac']*magnitude[low]))**(2*params['nstar']-1))
                if average: prefactor*=params['avfac']*np.sqrt(magnitude[low])
                bound=8*np.spacing(expfactor[low])*prefactor
                assert np.all(abs(observed[low]-expected[low])<=bound)
                assert np.all(actual[(abs(field)<rate.thr)]==0)
                assert np.all(np.isfinite(actual))
                assert np.max(actual)>1e10
    print(material,'coefficients',max_coeff,'resolved rates',max_rate,'underflow samples',underflow_count)


def test_adk_independent_decimal_hydrogen_limit():
    # Ip=one half atomic energy gives n*=1 and C_n^2=4 exactly, so the
    # reference requires no floating-point gamma function or production setup.
    rate=IonRateADK(ion.AU_ENERGY/2,threshold=False)
    with localcontext() as ctx:
        ctx.prec=100
        d=Decimal.from_float
        ip=d(ion.AU_ENERGY)/2;wp=ip/d(ion.HBAR)
        wt=d(ion.ELECTRON)/(2*d(ion.M_E)*ip).sqrt()
        for field in [1e10,3e10,1e11]:
            x=4*wp/(wt*d(field))
            exact=2*wp*4*x*(-x/3).exp()
            error=abs(d(rate(field))/exact-1)
            print('100-digit ADK',field,float(error))
            assert error<Decimal('1e-13')


def test_adk_symmetry_occupancy_cycle_average_and_ownership():
    fields=np.array([[0,1e10,-1e10],[2e10,4e10,-4e10]])
    before=fields.copy();rate=IonRateADK('Ar');result=rate(fields)
    np.testing.assert_array_equal(fields,before)
    assert result.flags.owndata and result.shape==fields.shape
    assert isinstance(rate(1e10),float)
    np.testing.assert_array_equal(result,rate(-fields))
    np.testing.assert_allclose(result,2*IonRateADK('Ar',occupancy=1)(fields),rtol=1e-15)
    average=IonRateADK('Ar',cycle_average=True)
    np.testing.assert_allclose(average(fields),result*average.avfac*np.sqrt(abs(fields)),rtol=1e-15)
    assert abs(average(4e10)/rate(4e10)-1)>.1
    assert IonRateADK('Ar',threshold=False)(0.)==0
    for invalid in [np.nan,np.inf,-np.inf,[1,np.nan],[1+1j]]:
        with pytest.raises(ValueError): rate(invalid)
    assert rate(np.empty((0,2))).shape==(0,2)
    for _ in range(4): assert IonRateADK('Ar')(4e10)==rate(4e10)


@pytest.mark.parametrize('args,kwargs',[
    (('Air',),{}),(('unknown',),{}),((0.,),{}),((-1.,),{}),((np.inf,),{}),
    (('Ar',),{'occupancy':0}),(('Ar',),{'occupancy':1.5}),
    (('Ar',),{'threshold':1}),(('Ar',),{'cycle_average':'false'}),
    ((1e-300,),{}),((1e300,),{}),
])
def test_adk_invalid_setup(args,kwargs):
    with pytest.raises((ValueError,TypeError)): IonRateADK(*args,**kwargs)
    with pytest.raises(ValueError): ionisation_potential('Ar','invalid')
