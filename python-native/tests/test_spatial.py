"""Independent modal geometry oracle, analytic integrals and callback contracts."""
import json
import math
import os
from pathlib import Path
import threading
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from amalthea_native import Mode, MarcatiliMode
from amalthea_native import quadrature
from amalthea_native.spatial import _ModeSpace, _POWER_FACTOR, _normalization

A = 125e-6
CASES = ('radial', 'radial-full', 'polarized', 'mixed', 'x-only', 'cartesian', 'cartesian-reduced')


def radius(z):
    return A * (1 + .1*np.sin(3*z) + .05*z*z)


class CartesianMode(Mode):
    def __init__(self, index=1, tapered=False):
        self.index, self.tapered = index, tapered

    def dimlimits(self, *, z=0.):
        a = radius(z)
        return 'cartesian', (-a, -a/2), (a, a/2)

    def field(self, xy, *, z=0.):
        x, y = np.asarray(xy[0])/radius(z), np.asarray(xy[1])/(radius(z)/2)
        out = np.array([1+x, .2*(1-y*y)] if self.index == 1 else [.3*(1-x*x), y])
        return out*(1-x*x) if self.tapered else out

    def neff(self, omega, *, z=0.):
        return 1.0001 + 1e-7j + (self.index-1)*2e-6 + z*1e-5


class EquivalentMode(Mode):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def neff(self, omega, *, z=0.):
        return self.wrapped.neff(omega, z=z)

    def dimlimits(self, *, z=0.):
        return self.wrapped.dimlimits(z=z)

    def field(self, xy, *, z=0.):
        return self.wrapped.field(xy, z=z)


def relative(a, b):
    return np.linalg.norm(np.asarray(a)-b)/np.linalg.norm(b)


def readcomplex(path, shape):
    raw = np.loadtxt(path)
    return (raw[:, 0] + 1j*raw[:, 1]).reshape(shape, order='F')


@pytest.fixture
def oracle():
    root = os.environ.get('AMALTHEA_SPATIAL_ORACLE')
    if root is None:
        pytest.skip('set AMALTHEA_SPATIAL_ORACLE to the independent Julia fixture')
    root = Path(root)
    assert all((root/name/'projection-2-refined.txt').exists() for name in CASES)
    return root


@pytest.mark.parametrize('name', CASES)
def test_independent_spatial_oracle(oracle, name):
    root = oracle/name
    spec = tomllib.loads((root/'parameters.toml').read_text())
    modes = ([MarcatiliMode(radius, 'Ar', 2., **item) for item in spec['modes']]
             if spec['modes'] else [CartesianMode(1, spec['tapered']), CartesianMode(2, spec['tapered'])])
    space = _ModeSpace(modes, components=spec['components'], full=spec['full'])
    field = readcomplex(root/'field.txt', (5, len(modes)))
    shape = field.shape
    records = []
    for zi, z in enumerate((0., .137), 1):
        geometry = space.at(z)
        points = np.loadtxt(root/f'points-{zi}.txt')
        expected = readcomplex(root/f'synthesis-{zi}.txt', (5, len(space.indices), len(points)))
        synthesis = geometry.synthesize(field, points)
        assert relative(synthesis, expected.transpose(2, 0, 1)) < 1e-13
        nodes = (synthesis*np.sum(np.abs(synthesis)**2, axis=2, keepdims=True)) @ geometry.matrix(points).transpose(0,2,1)
        expected_nodes = readcomplex(root/f'nodes-{zi}.txt', (*field.shape, len(points))).transpose(2,0,1)
        node_error = relative(nodes, expected_nodes)
        assert node_error < 1e-13
        assert relative(geometry.scales**2, np.loadtxt(root/f'normalization-{zi}.txt')) < 1e-13
        wanted = readcomplex(root/f'projection-{zi}-refined.txt', shape)
        julia_fine = readcomplex(root/f'projection-{zi}-fine.txt', shape)
        response = lambda physical, points: physical*np.sum(np.abs(physical)**2, axis=2, keepdims=True)
        coarse = geometry.project(field, response, rtol=1e-4)
        fine = geometry.project(field, response, rtol=1e-10)
        refined = geometry.project(field, response, rtol=1e-12)
        errors = [relative(item.value, wanted) for item in (coarse, fine, refined)]
        assert errors[1] < 1e-9
        assert errors[2] < 1e-11
        assert relative(julia_fine, wanted) < 1e-9
        for item in (coarse, fine, refined):
            assert item.error_norm <= item.tolerance
        records.append(dict(z=z, errors=errors, node_error=node_error, evaluations=refined.evaluations,
                            julia_refinement=relative(julia_fine, wanted)))
    print('spatial', name, json.dumps(records))


@pytest.mark.parametrize('kind,n,m', [('HE',1,1), ('HE',2,2), ('TE',0,1), ('TM',0,2)])
def test_generic_normalization_matches_analytic_marcatili(kind, n, m):
    mode = MarcatiliMode(radius, 'Ar', 2., kind=kind, n=n, m=m, phi=.17)
    custom = EquivalentMode(mode)
    for z in (0., .237):
        assert abs(custom.N(z=z)/mode.N(z=z)-1) < 1e-13
        omega = np.array([1.3e15, 2.1e15, 3.7e15])
        assert relative(custom.beta(omega,z=z), mode.beta(omega,z=z)) < 1e-15
        assert relative(custom.alpha(omega,z=z), mode.alpha(omega,z=z)) < 1e-15
        assert custom.dispersion(1,2.1e15,z=z) == mode.dispersion(1,2.1e15,z=z)


@pytest.mark.parametrize('index', [1, 2])
def test_cartesian_normalization_analytic(index):
    mode = CartesianMode(index)
    integral = 16/3+.04*32/15 if index == 1 else .09*32/15+4/3
    for z in (0., .71):
        actual = _POWER_FACTOR*(radius(z)**2/2)*integral
        assert abs(mode.N(z=z)/actual-1) < 1e-14


@pytest.mark.parametrize('full', [False, True])
def test_orthogonality_and_nonzero_mode_transfer(full):
    modes = [MarcatiliMode(A, 'Ar', 2., m=m) for m in (1,2,3)]
    geometry = _ModeSpace(modes, components='y', full=full).at(.0)
    identity = np.eye(3, dtype=complex)
    overlap = geometry.project(identity, lambda f,p:f, rtol=1e-12).value*_POWER_FACTOR
    assert np.linalg.norm(overlap-identity) < 1e-13
    cubic = geometry.project(identity[:1],lambda f,p:f*np.abs(f)**2,rtol=1e-12).value
    transfer = abs(cubic[0,1]/cubic[0,0])
    assert transfer > .1
    print('spatial transfer',full,transfer)


def test_full_and_radial_agree_and_polarization_changes_coupling():
    modes = [MarcatiliMode(A, 'Ar', 2., m=1),MarcatiliMode(A, 'Ar', 2., m=2)]
    field = np.array([[1.+.3j,.5-.2j], [.2-.4j,.6+.1j]])
    response = lambda f,p:f*np.sum(abs(f)**2,axis=2,keepdims=True)
    radial = _ModeSpace(modes, components='xy', full=False).at(0.).project(field,response,rtol=1e-12)
    full = _ModeSpace(modes, components='xy', full=True).at(0.).project(field,response,rtol=1e-12)
    assert relative(radial.value,full.value)<1e-13
    modes[1] = MarcatiliMode(A,'Ar',2.,m=2,phi=math.pi/2)
    polarized = _ModeSpace(modes,components='xy',full=False).at(0.).project(field,response,rtol=1e-12)
    assert relative(polarized.value,radial.value)>.1


@pytest.mark.parametrize('rule', ['gk21', 'genz-malik'])
def test_independent_quadrature_rule_and_analytic_complex_array(rule):
    coeff = (np.arange(12).reshape(3,4)+1)*(1+.3j)
    exact = coeff*(math.e-1)*(1-math.cos(1.))
    value = quadrature.integrate(lambda p:np.exp(p[:,0,None,None])*np.sin(p[:,1,None,None])*coeff,
                                [0.,0.],[1.,1.],rtol=1e-10,rule=rule)
    assert relative(value.value,exact)<1e-10
    assert value.error_norm<=value.tolerance


def test_global_norm_not_component_tolerance(monkeypatch):
    # Each component is below atol=.1; their joint L2 error exceeds it.
    fake = SimpleNamespace(estimate=np.zeros((4,4,2)),error=np.full((4,4,2),.05),status='converged')
    monkeypatch.setattr(quadrature,'cubature',lambda *args,**kwargs:fake)
    with pytest.raises(RuntimeError,match='global L2'):
        quadrature.integrate(lambda p:p,[0.],[1.],rtol=0.,atol=.1)


def test_scaled_norm_avoids_overflow_and_underflow():
    for scale in (1e-280,1e280):
        assert abs(quadrature._norm(np.array([3,4])*scale)/scale-5.) < 2e-15


def test_global_acceptance_after_bounded_subdivision(monkeypatch):
    calls=[]
    def partial(*args,**kwargs):
        calls.append(kwargs['max_subdivisions'])
        error=2. if len(calls)==1 else .4
        return SimpleNamespace(estimate=np.zeros((2,2)),error=np.full((2,2),error),status='not_converged')
    monkeypatch.setattr(quadrature,'cubature',partial)
    result=quadrature.integrate(lambda p:p,[0.],[1.],rtol=0.,atol=1.)
    assert calls==[0,4] and result.error_norm==.8


@pytest.mark.parametrize('error,accepted', [(1e306,False),(1e292,True),(1e308,False)])
def test_extreme_finite_global_error_budget(monkeypatch,error,accepted):
    fake=SimpleNamespace(estimate=np.full((4,4,2),1e308),error=np.full((4,4,2),error))
    monkeypatch.setattr(quadrature,'cubature',lambda *a,**kw:fake)
    if accepted:
        result=quadrature.integrate(lambda p:p,[0.],[1.],rtol=1e-3)
        assert math.isfinite(result.tolerance) and result.error_norm<result.tolerance
        assert abs(result.tolerance/1e305-math.sqrt(32))<2e-15
    else:
        with pytest.raises(RuntimeError,match='global'):
            quadrature.integrate(lambda p:p,[0.],[1.],rtol=1e-3)


def test_serial_callbacks_ownership_and_fresh_position():
    positions, retained, threads = [], [], []
    class Recorded(CartesianMode):
        def N(self, *, z=0.):
            positions.append(z)
            return super().N(z=z)
    mode=Recorded();space=_ModeSpace([mode],components='xy',full=True)
    field=np.array([[1.+.2j]])
    before=field.copy()
    def response(physical,points):
        threads.append(threading.get_ident());retained.append((physical.copy(),physical,points))
        return physical
    first=space.at(.137).project(field,response,rtol=1e-10)
    second=space.at(.271).project(field,response,rtol=1e-10)
    assert positions==[.137,.271]
    assert len(set(threads))==1
    assert np.array_equal(field,before)
    assert relative(first.value,second.value)<1e-13  # normalized power integral
    for expected,actual,points in retained:assert np.array_equal(actual,expected)
    first.value[:]=0
    assert np.any(second.value)


def test_callback_original_exception_and_invalid_response():
    geometry=_ModeSpace([CartesianMode()]).at(0.)
    failure=LookupError('custom spatial error')
    def broken(*args):raise failure
    with pytest.raises(LookupError) as caught:geometry.project(np.ones((2,1)),broken)
    assert caught.value is failure
    for callback,match in [(lambda f,p: f[:,:,:1], 'shape'),(lambda f,p: f*np.nan,'nonfinite')]:
        with pytest.raises(ValueError,match=match):geometry.project(np.ones((2,1)),callback)


@pytest.mark.parametrize('fault', ['shape','complex','nan','normalization','domain','neff'])
def test_custom_mode_rejects_invalid_outputs(fault):
    class Bad(CartesianMode):
        def field(self,xy,*,z=0.):
            value=super().field(xy,z=z)
            return {'shape':lambda:value[0], 'complex':lambda:value+1j, 'nan':lambda:value*np.nan}.get(fault,lambda:value)()
        def N(self,*,z=0.):
            return -1. if fault=='normalization' else super().N(z=z)
        def dimlimits(self,*,z=0.):
            return ('polar',(0.,0.),(-1.,1.)) if fault=='domain' else super().dimlimits(z=z)
        def neff(self,omega,*,z=0.):return np.ones((2,2)) if fault=='neff' else super().neff(omega,z=z)
    with pytest.raises(ValueError):
        if fault=='neff':Bad().beta(np.array([1e15,2e15]))
        else:_ModeSpace([Bad()]).at(0.)


def test_domain_boundaries_and_domain_mismatch():
    mode=MarcatiliMode(A,'Ar',2.)
    geometry=_ModeSpace([mode]).at(0.)
    assert np.all(geometry.matrix([[A,0.],[2*A,0.]])==0)
    assert np.any(geometry.matrix([[0.,0.]]))
    with pytest.raises(ValueError,match='negative'):geometry.matrix([[-1.,0.]])
    with pytest.raises(ValueError,match='same integration domain'):
        _ModeSpace([mode,MarcatiliMode(A*2,'Ar',2.)]).at(0.)
    geometry=_ModeSpace([CartesianMode()]).at(0.)
    assert np.all(geometry.matrix([[A,0.],[0.,A/2],[-A,0.]])==0)


@pytest.mark.parametrize('options', [dict(rtol=-1),dict(atol=-1),dict(rtol=0,atol=0),
                                    dict(rtol=float('nan')),dict(maxevals=0),dict(batch_size=0),dict(rule='wrong')])
def test_invalid_quadrature_controls(options):
    with pytest.raises(ValueError):quadrature.integrate(lambda p:p,[0.],[1.],**options)


def test_quadrature_budget_and_invalid_arrays():
    with pytest.raises(RuntimeError,match='maxevals'):
        quadrature.integrate(lambda p:np.sin(100*p),[0.],[1.],maxevals=10)
    for fun in (lambda p:np.ones(2),lambda p:np.full_like(p,np.nan)):
        with pytest.raises(ValueError):quadrature.integrate(fun,[0.],[1.])


def test_repeated_geometry_destruction():
    for i in range(12):
        geometry=_ModeSpace([CartesianMode()]).at(i/100.)
        value=geometry.project(np.ones((3,1)),lambda f,p:f,rtol=1e-10)
        assert np.allclose(value.value,1/_POWER_FACTOR,rtol=1e-13)


@pytest.mark.parametrize('method', ['field', 'N', 'dimlimits'])
def test_mode_exception_identity(method):
    failure=LookupError('mode evaluation failure')
    def broken(*args,**kwargs):raise failure
    mode=CartesianMode()
    setattr(mode,method,broken)
    with pytest.raises(LookupError) as caught:_ModeSpace([mode]).at(.123)
    assert caught.value is failure


def test_polar_annular_normalization():
    class Annular(Mode):
        def dimlimits(self,*,z=0.):return 'polar',(.2,0.),(.7,2*math.pi)
        def field(self,xy,*,z=0.):return np.array([np.zeros_like(xy[0]),np.ones_like(xy[0])])
    mode=Annular()
    exact=_POWER_FACTOR*math.pi*(.7**2-.2**2)
    assert abs(mode.N()/exact-1)<1e-14
    value=_ModeSpace([mode],components='y',full=False).at(0.).project(np.ones((1,1)),lambda f,p:f,rtol=1e-12)
    assert abs(value.value[0,0]*_POWER_FACTOR-1)<1e-14
