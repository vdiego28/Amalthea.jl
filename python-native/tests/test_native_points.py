"""Portable native point arrays, ownership, dispatch, and modal trajectories."""
import gc
import os
import tomllib
from pathlib import Path
import weakref

import numpy as np
import pytest

from amalthea_native import prop_capillary,solve_precon,_native
from amalthea_native.modal import _ModalCapillary
from amalthea_native._point_native import _NativePoints
from test_modal import BASE,LENGTH,STEP,KINDS,setup,readfield,relative,fixed_nodes


@pytest.mark.parametrize('envelope',[False,True])
@pytest.mark.parametrize('kind',['scalar','vector','raman','mixture'])
def test_point_arrays_and_repeated_calls(envelope,kind):
    gas=('N2','H2') if kind=='mixture' else 'N2' if kind=='raman' else 'Ar'
    pressure=(2.,1.) if kind=='mixture' else 2.
    kw=BASE|dict(envelope=envelope,modes=2,polarisation='circular' if kind=='vector' else 'linear',
                raman=kind in ('raman','mixture'),backend='python')
    model=_ModalCapillary(125e-6,LENGTH,gas,pressure,**kw)
    points=np.array([[.13*125e-6,.2],[.51*125e-6,.73],[.87*125e-6,2.1]])
    spectrum=model.space.at(0.).synthesize(model.initial,points);before=spectrum.copy()
    native=_NativePoints(model)
    for scale in (1.,0.,.7):
        input=spectrum*scale
        expected=model.point_response(input,points)
        actual=native(input)
        assert relative(actual,expected)<1e-13
        retained=actual.copy()
        native(spectrum*.3)
        assert np.array_equal(actual,retained)
    assert np.array_equal(spectrum,before)
    assert native(np.zeros((0,model.n,model.npol),complex)).shape==(0,model.n,model.npol)
    ref=weakref.ref(native);del native;gc.collect();assert ref() is None


NATIVE_CASES=[f'{kind}-{grid}' for grid in ('env','real') for kind in KINDS
              if kind not in ('gradient','custom','average-TE','average-TM','average-HE21','radial-thg')]


@pytest.mark.parametrize('name',NATIVE_CASES)
def test_native_points_independent_julia(name):
    value=os.environ.get('AMALTHEA_MODAL_CAPILLARY_ORACLE')
    if value is None:pytest.skip('set AMALTHEA_MODAL_CAPILLARY_ORACLE')
    root=Path(value)/name;args,kw,cfg=setup(root)
    model=_ModalCapillary(*args,**kw)
    assert model.native_points is not None
    initial=readfield(root/'initial.txt',cfg['field_shape'])
    geometry=model.space.at(0.)
    points=np.array([[.13*125e-6,.2],[.51*125e-6,.73],[.87*125e-6,2.1]])
    actual=model.point_response(geometry.synthesize(initial,points),points)
    errors=[relative(actual[i],readfield(root/f'point-spectrum-1-{i+1}.txt',(model.n,model.npol))) for i in range(3)]
    assert max(errors)<1e-13
    reference=_ModalCapillary(*args,**kw,backend='python')
    assert relative(model.rhs(0.,initial),reference.rhs(0.,initial))<1e-13
    shape=cfg['field_shape'];positions=np.loadtxt(root/'linear-positions.txt')
    operators=readfield(root/'linear-samples.txt',(np.prod(shape),len(positions)))
    transferred={z:operators[:,i].reshape(shape,order='F') for i,z in enumerate(positions)}
    interval=solve_precon(fixed_nodes(model,np.loadtxt(root/'rule.txt',ndmin=2)),lambda z:transferred[z],initial,STEP,
                         dt=STEP,min_dt=STEP,max_dt=STEP,rtol=1e-9,atol=1e-12,saveN=7,step_filter=model.window,
                         locextrap=kw['locextrap'])
    dense_error=relative(interval.field,readfield(root/'interval.txt',(*shape,7)))
    assert dense_error<1e-13
    controls=tomllib.loads((root/'adaptive-controls.toml').read_text())
    result=prop_capillary(*args,**kw,init_dz=controls['dt'],max_dz=controls['max_dt'],
                          rtol=controls['rtol'],atol=controls['atol'])
    error=relative(result.field,readfield(root/'adaptive.txt',(*cfg['field_shape'],7)))
    assert error<1e-6 and result.metadata['point_evaluator']=='rust'
    print('native points',name,'nodes',max(errors),'dense',dense_error,'trajectory',error,flush=True)


def test_auto_bypasses_python_temporal_response(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Python temporal response called')
    monkeypatch.setattr(_ModalCapillary,'polarization',forbidden)
    result=prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=2)
    assert result.metadata['backend']=='python' and result.metadata['point_evaluator']=='rust'
    assert result.metadata['point_fft']=='rustfft/realfft'
    with pytest.raises(AssertionError,match='Python temporal'):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**BASE,modes=2,backend='python')


@pytest.mark.parametrize('options',[dict(responses=lambda f,c:f*0),dict(plasma='ADK'),dict(thg=False)])
def test_unsupported_point_cases_keep_python(options):
    model=_ModalCapillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(modes=2)|options))
    assert model.native_points is None
    with pytest.raises(NotImplementedError):
        prop_capillary(125e-6,LENGTH,'Ar',2.,**(BASE|dict(modes=2)|options),backend='native')


def point_options():
    return dict(n_time=16,components=1,is_real=False,time_window=[1.]*32,prefactor=[-1j]*16,kerr=.1,dt=.2)


@pytest.mark.parametrize('overrides',[dict(n_time=17),dict(n_time=2**63),dict(components=3),dict(components=0),
    dict(time_window=[1.]*12),dict(time_window=[np.nan]*32),dict(prefactor=[1j]*15),dict(kerr=np.inf),
    dict(dt=0),dict(raman=[1.]*16),dict(components=2,raman=[1.]*32)])
def test_invalid_point_configuration(overrides):
    with pytest.raises(ValueError):_native.ModalPoints(**(point_options()|overrides))


def test_invalid_point_batches_and_result_overflow():
    native=_native.ModalPoints(**point_options())
    for fields,count in [([],1),([0j]*15,1),([0j]*16,0),([complex(np.nan,0)]*16,1),([],2**63)]:
        with pytest.raises(ValueError):native.evaluate(np.asarray(fields,dtype=complex),count)
    with pytest.raises(ValueError,match='nonfinite'):
        native.evaluate(np.full(16,1e200,dtype=complex),1)
    assert np.array_equal(native.evaluate(np.zeros(16,complex),1),np.zeros(16,complex))


def test_numpy_boundary_strides_dtype_readonly_and_lifetime():
    native=_native.ModalPoints(**point_options())
    field=np.linspace(.1,1.,16).astype(complex);field.setflags(write=False)
    result=native.evaluate(field,1);before=result.copy()
    with pytest.raises(ValueError,match='contiguous'):native.evaluate(np.zeros(32,complex)[::2],1)
    with pytest.raises(TypeError):native.evaluate(np.zeros(16),1)
    with pytest.raises(TypeError):native.evaluate(np.zeros((4,4),complex),1)
    native.evaluate(np.zeros(16,complex),1)
    del native;gc.collect()
    assert np.array_equal(result,before) and not field.flags.writeable
