"""Independent PPT physics, stable special functions, tables and local caching."""
import os
from pathlib import Path
import tomllib

import numpy as np
import pytest

from amalthea_native import IonRatePPT,IonRatePPTAccel
from amalthea_native.ppt import _phi,barrier_suppression,AU_POLARISABILITY
from amalthea_native.materials import ionisation_potential


@pytest.fixture
def oracle():
    path=os.environ.get('AMALTHEA_PPT_ORACLE')
    if path is None:pytest.skip('set AMALTHEA_PPT_ORACLE for independent PPT acceptance')
    return Path(path)


@pytest.mark.parametrize('name',['He','HeJ','HeB','Ne','Ar','ArB','Kr','Xe','N2','O2',
                                 'base1030','no-stark','no-dipole','average','integral','no-msum',
                                 'single','cnl','tight','occupancy'])
def test_independent_ppt_rates(oracle,name):
    params=tomllib.loads((oracle/name/'parameters.toml').read_text())
    kw={key:params[key] for key in ['stark_shift','dipole_corr','sum_tol','cycle_average',
                                   'sum_integral','msum','occupancy']}
    if 'Cnl' in params:kw['Cnl']=params['Cnl']
    if name=='occupancy':kw['occupancy']=lambda m:2 if m==0 else 1
    model=IonRatePPT(params['material'],params['lambda0'],**kw)
    for key in ['ionpot','Z','l','omega0_au','alpha_ion_au']:
        assert abs(getattr(model,key)-params[key])<=1e-13*abs(params[key])
    assert abs(model.kw['delta_alpha']-params['delta_alpha'])<=1e-13*abs(params['delta_alpha'])
    field,reference=np.loadtxt(oracle/name/'rates.txt').T
    actual=model(field);selected=field!=0
    errors=abs(actual[selected]/reference[selected]-1)
    print(name,'rate',np.max(errors))
    assert np.max(errors)<1e-13
    assert params['zero_evaluated'] is False and np.isnan(reference[field==0]).all()
    assert actual[field==0]==0
    assert np.max(actual)>1e10
    np.testing.assert_array_equal(actual,model(-field))
    assert actual.flags.owndata


def test_ppt_numeric_setup_and_option_effects(oracle):
    field,expected=np.loadtxt(oracle/'numeric-l2.txt').T
    model=IonRatePPT(ionisation_potential('Ar'),800e-9,1.,2,**{'Δα':2e-41,'α_ion':1e-41})
    error=np.max(abs(model(field)/expected-1));print('numeric l2',error)
    assert error<1e-13
    for name in ['no-stark','no-dipole','average','integral','no-msum','single','cnl','occupancy']:
        field,reference=np.loadtxt(oracle/name/'rates.txt').T
        selected=field>0
        base_field,base_rate=np.loadtxt(oracle/'base1030/rates.txt').T
        np.testing.assert_array_equal(field,base_field)
        base=base_rate[selected]
        effect=np.linalg.norm(base-reference[selected])/np.linalg.norm(base)
        print('PPT option',name,'effect',effect)
        assert effect>1e-5


@pytest.mark.parametrize('m',[0,1,2])
def test_ppt_phi_julia_and_independent_quadrature(oracle,m):
    x,original,refined=np.loadtxt(oracle/f'phi-{m}.txt').T
    actual=_phi(m,x);selected=x>0
    oracle_error=np.max(abs(original[selected]/refined[selected]-1))
    error=np.max(abs(actual[selected]/refined[selected]-1))
    print('phi',m,'refined error',error,'original Julia discrepancy',oracle_error)
    assert error<1e-13
    # The default large-x Julia quadrature has its own tolerance; preserve
    # its measured discrepancy separately from the refined 1e-13 gate.
    assert np.max(abs(actual[(x>0)&(x<=26)]/original[(x>0)&(x<=26)]-1))<1e-13
    from mpmath import mp
    previous=mp.dps
    ctx=mp.clone();ctx.dps=100
    for value in [25.99,26.01,40.,100.]:
        bx=ctx.mpf(value);b2=bx*bx
        # v=x*(x-y) resolves the narrow boundary layer at large x; this
        # independently integrates the defining scaled integral.
        bounds=sorted(set([ctx.mpf(0),ctx.mpf(1),ctx.mpf(4),ctx.mpf(16),ctx.mpf(64),b2]))
        exact=ctx.quad(lambda v:(2*v-v*v/b2)**m*ctx.exp(-2*v+v*v/b2),bounds)/bx
        assert abs(ctx.mpf(_phi(m,value))/exact-1)<ctx.mpf('1e-13')
    assert mp.dps==previous


def test_ppt_same_sample_table_oracle(oracle):
    field,rate=np.loadtxt(oracle/'table.txt').T
    prepared=IonRatePPTAccel('Ar',800e-9,N=1024,cache=False,stark_shift=False,dipole_corr=False)
    node_error=np.max(abs(prepared.field_nodes/field-1))
    rate_error=np.max(abs(prepared.rate_nodes/rate-1))
    print('independently generated PPT table',node_error,rate_error)
    assert node_error<1e-13 and rate_error<1e-13
    table=IonRatePPTAccel.from_samples(field,rate)
    queries,reference=np.loadtxt(oracle/'table-queries.txt').T
    actual=table(queries);selected=reference>0
    error=np.max(abs(actual[selected]/reference[selected]-1));print('PPT same-sample spline',error)
    assert error<1e-13
    np.testing.assert_array_equal(actual[~selected],reference[~selected])
    assert table(-table.Emax*10)==table(table.Emax)
    assert table(np.nextafter(table.Emin,0))==0
    before=table.field_nodes;before[:]=0
    assert table.Emin>0 and np.all(table.field_nodes>0)


def test_ppt_table_refinement_and_default_size(tmp_path):
    # Use direct evaluation as an independent interpolation control, retaining
    # changes from the multiphoton thresholds instead of testing node identity.
    model=IonRatePPT('Ar',800e-9,stark_shift=False,dipole_corr=False)
    queries=np.array([7.123e9,1.23456e10,2.1345e10,3.3456e10])
    expected=model(queries);errors=[]
    for n in [1024,4096,65536]:
        table=IonRatePPTAccel('Ar',800e-9,**({} if n==65536 else {'N':n}),
                              cache=False,stark_shift=False,dipole_corr=False)
        assert len(table.field_nodes)==n
        errors.append(np.max(abs(table(queries)/expected-1)))
    print('PPT table refinement',errors)
    assert errors[1]<errors[0] and errors[2]<errors[1]
    assert errors[-1]<1e-6


def test_ppt_cache_parameters_corruption_and_callbacks(tmp_path,monkeypatch):
    args=('Ar',800e-9);kw=dict(N=128,cachedir=tmp_path)
    first=IonRatePPTAccel(*args,**kw);reference=first([0,1e10,4e10])
    assert not first.cache_hit
    def fail(*args):raise AssertionError('recomputed a valid cached table')
    with monkeypatch.context() as patch:
        patch.setattr(IonRatePPT,'__call__',fail)
        cached=IonRatePPTAccel(*args,**kw)
        np.testing.assert_array_equal(cached([0,1e10,4e10]),reference)
        assert cached.cache_hit
    other=IonRatePPTAccel(*args,**kw,cycle_average=True)
    assert other.cache_path!=first.cache_path and not other.cache_hit
    first.cache_path.write_bytes(b'corrupt')
    repaired=IonRatePPTAccel(*args,**kw)
    assert not repaired.cache_hit
    np.testing.assert_array_equal(repaired([0,1e10,4e10]),reference)
    # Corruption can leave a valid NPZ with finite arrays: verify the digest, too.
    with np.load(repaired.cache_path,allow_pickle=False) as data:
        contents={k:np.array(data[k],copy=True) for k in data.files}
    contents['rate'][10]*=2
    np.savez_compressed(repaired.cache_path,**contents)
    repaired=IonRatePPTAccel(*args,**kw)
    assert not repaired.cache_hit
    np.testing.assert_array_equal(repaired([0,1e10,4e10]),reference)
    calls=[]
    def occupancy(m):calls.append(m);return 2 if m==0 else 1
    model=IonRatePPT('Ar',800e-9,occupancy=occupancy)
    result=model([1e10,2e10]);assert calls==[-1,0,1]*2
    assert np.all(result>0)
    with pytest.raises(ValueError,match='cache=False'):IonRatePPTAccel(*args,**kw,occupancy=occupancy)
    assert np.all(IonRatePPTAccel(*args,N=32,cache=False,occupancy=occupancy)([1e10,2e10])>0)
    boom=RuntimeError('occupancy failed')
    def raise_error(m):raise boom
    with pytest.raises(RuntimeError) as caught:IonRatePPT(*args,occupancy=raise_error)(1e10)
    assert caught.value is boom
    for invalid in [lambda m:np.nan,lambda m:[1,2],lambda m:-1]:
        with pytest.raises(ValueError):IonRatePPT(*args,occupancy=invalid)(1e10)


@pytest.mark.parametrize('kwargs',[
    {'sum_tol':0},{'sum_tol':1},{'max_terms':0},{'msum':1},{'cycle_average':'no'},
    {'delta_alpha':np.nan},{'alpha_ion':np.inf},{'occupancy':-1},{'Cnl':0},
    {'delta_alpha':0,'Δα':1},{'unknown':1},
])
def test_ppt_invalid_options(kwargs):
    with pytest.raises((ValueError,TypeError)):IonRatePPT('Ar',800e-9,**kwargs)


def test_ppt_invalid_fields_tables_and_nonconvergence():
    model=IonRatePPT('Ar',800e-9)
    for field in [np.nan,np.inf,[1,np.nan],[1j]]:
        with pytest.raises(ValueError):model(field)
    with pytest.raises(ValueError):IonRatePPT('H2',800e-9)
    with pytest.raises(ValueError):IonRatePPT(1e-18,800e-9,1,-1)
    with pytest.raises(ArithmeticError):IonRatePPT('Ar',800e-9,max_terms=1)(4e10)
    with pytest.raises(ValueError):IonRatePPT('Ar',800e-9,delta_alpha=-1e-30)(1e10)
    for field,rate in [([1,2,3],[1,2,3]),([1,1,2,3],[1,2,3,4]),([1,2,3,4],[0,0,0,0]),
                       ([1,2,3,4],[1,2,np.nan,4]),([1,2,3,4],[1,2,-3,4])]:
        with pytest.raises(ValueError):IonRatePPTAccel.from_samples(field,rate)
    assert model(np.empty((0,2))).shape==(0,2)
    field=np.array([[1e10,2e10],[3e10,4e10]]);before=field.copy()
    actual=model(field);np.testing.assert_array_equal(field,before)
    assert actual.flags.owndata and actual.shape==field.shape


def test_ppt_atomic_cache_concurrent_construction(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier=Barrier(2)
    original=IonRatePPT.__call__
    def synchronized(self,field):
        barrier.wait(timeout=30)
        return original(self,field)
    with monkeypatch.context() as patch:
        patch.setattr(IonRatePPT,'__call__',synchronized)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(IonRatePPTAccel,'Ar',800e-9,N=64,cachedir=tmp_path) for _ in range(2)]
            results=[job.result(timeout=30) for job in jobs]
    assert all(not r.cache_hit for r in results)
    loaded=IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    assert loaded.cache_hit and len(list(tmp_path.iterdir()))==1
    for result in results:np.testing.assert_array_equal(result.rate_nodes,loaded.rate_nodes)


@pytest.mark.parametrize('winerror',[5,32,33])
def test_ppt_cache_recovers_from_windows_publication_contention(tmp_path,monkeypatch,winerror):
    from amalthea_native import ppt
    replace=os.replace;attempts=[];delays=[]
    denied=PermissionError('temporary Windows cache contention');denied.winerror=winerror
    def contested(source,destination):
        attempts.append((source,destination))
        if len(attempts)<=3:raise denied
        replace(source,destination)
    with monkeypatch.context() as patch:
        patch.setattr(ppt.os,'replace',contested)
        patch.setattr(ppt.time,'sleep',delays.append)
        table=IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    assert not table.cache_hit and len(attempts)==4 and len(delays)==3
    assert all(delay>0 for delay in delays)
    assert list(tmp_path.iterdir())==[table.cache_path]
    def no_recompute(*args):raise AssertionError('valid published cache was not reused')
    monkeypatch.setattr(IonRatePPT,'__call__',no_recompute)
    loaded=IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    assert loaded.cache_hit
    np.testing.assert_array_equal(loaded.rate_nodes,table.rate_nodes)


@pytest.mark.parametrize('winerror',[5,32,33])
def test_ppt_cache_persistent_windows_error_preserves_destination(tmp_path,monkeypatch,winerror):
    from amalthea_native import ppt
    table=IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    previous=b'old cache requiring repair'
    table.cache_path.write_bytes(previous)
    denied=PermissionError('persistent cache access error');denied.winerror=winerror
    attempts=[];delays=[]
    def denied_replace(source,destination):
        attempts.append(source)
        raise denied
    with monkeypatch.context() as patch:
        patch.setattr(ppt.os,'replace',denied_replace)
        patch.setattr(ppt.time,'sleep',delays.append)
        with pytest.raises(PermissionError) as caught:
            IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    assert caught.value is denied
    assert 1<len(attempts)<=8 and len(delays)==len(attempts)-1
    assert 0<sum(delays)<=1.28
    assert table.cache_path.read_bytes()==previous
    assert list(tmp_path.iterdir())==[table.cache_path]


@pytest.mark.parametrize('kind',['posix-permission','other-windows-error','disk-full'])
def test_ppt_cache_unrelated_publication_errors_are_not_retried(tmp_path,monkeypatch,kind):
    import errno
    from amalthea_native import ppt
    error=(OSError(errno.ENOSPC,'disk full') if kind=='disk-full'
           else PermissionError(errno.EACCES,'access denied'))
    if kind=='other-windows-error':error.winerror=87
    attempts=[]
    def fail_replace(source,destination):
        attempts.append(source)
        raise error
    def unexpected_sleep(delay):raise AssertionError('unrelated error was retried')
    with monkeypatch.context() as patch:
        patch.setattr(ppt.os,'replace',fail_replace)
        patch.setattr(ppt.time,'sleep',unexpected_sleep)
        with pytest.raises(OSError) as caught:
            IonRatePPTAccel('Ar',800e-9,N=64,cachedir=tmp_path)
    assert caught.value is error and len(attempts)==1
    assert list(tmp_path.iterdir())==[]


def test_ppt_nonuniform_samples_match_julia(oracle):
    field,rate=np.loadtxt(oracle/'nonuniform-table.txt').T
    queries,expected=np.loadtxt(oracle/'nonuniform-queries.txt').T
    table=IonRatePPTAccel.from_samples(field,rate)
    actual=table(queries);selected=expected>0
    error=np.max(abs(actual[selected]/expected[selected]-1))
    print('PPT nonuniform normalized-knot spline',error)
    assert error<1e-13 and np.all(actual[~selected]==0)
    assert len(table.field_nodes)==5
