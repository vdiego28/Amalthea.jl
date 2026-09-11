"""Independent readers verify result files without changing physical arrays."""
import json
import subprocess
import sys

import numpy as np
import pytest

from amalthea_native import prop_capillary, prop_gnlse


def response(field, context):
    return 2e-40 * field * np.abs(field)**2


@pytest.fixture(scope='module', params=['gnlse', 'carrier', 'modal'])
def result(request):
    common = dict(lambda0=800e-9, lambda_lims=(600e-9, 1200e-9),
                  trange=100e-15, tau_fwhm=20e-15, saveN=3)
    if request.param == 'gnlse':
        return prop_gnlse(.01, .001, [0, 0, -20e-27], power=2000., **common)
    extra = dict(envelope=True, modes=2, responses=response) if request.param == 'modal' else {}
    return prop_capillary(125e-6, 1e-5, 'Ar', 2., energy=100e-6,
                          plasma=False, raman=False, init_dz=5e-6, max_dz=5e-6,
                          **common, **extra)


def test_exact_roundtrip_and_independent_reconstruction(result, tmp_path, monkeypatch):
    h5py = pytest.importorskip('h5py')
    nested = dict(label='λ · 氩', values=np.array([1., 2.]),
                  enabled=np.bool_(True), count=np.int64(7))
    monkeypatch.setitem(result.metadata, 'user', nested)
    paths = [tmp_path/'result.npz', tmp_path/'result.h5']
    result.save_npz(paths[0])
    result.save_hdf5(paths[1])
    arrays = dict(Eomega=result.field, z=result.z)
    arrays.update({key: getattr(result.grid, key) for key in
                   ('t', 'omega', 'to', 'omega_over', 'twin', 'towin', 'omega_win', 'sidx')})
    with np.load(paths[0], allow_pickle=False) as archive, h5py.File(paths[1], 'r') as hdf:
        assert set(archive.files) == set(hdf)
        assert archive['format_version'] == hdf['format_version'][()] == 1
        for name, expected in arrays.items():
            np.testing.assert_array_equal(archive[name], expected)
            np.testing.assert_array_equal(hdf[name][()], expected)
            assert hdf[name].dtype == archive[name].dtype == expected.dtype
            assert hdf[name].compression == 'gzip'
        for name in ('grid', 'parameters', 'metadata'):
            assert archive[name].dtype.kind == 'U'
            assert json.loads(str(archive[name])) == json.loads(hdf[name].asstr()[()])
        description = json.loads(hdf['grid'].asstr()[()])
        assert description['is_real'] == result.grid.is_real
        assert description['reference_lambda'] == 800e-9
        assert description['field_axes'] == (
            ['frequency', 'mode', 'saved_position'] if result.field.ndim == 3
            else ['frequency', 'saved_position'])
        metadata = json.loads(hdf['metadata'].asstr()[()])
        assert metadata['user'] == dict(label='λ · 氩', values=[1., 2.], enabled=True, count=7)
        field = hdf['Eomega'][()]
        if description['is_real']:
            reconstructed = np.fft.irfft(field, n=hdf['t'].size, axis=0)
        else:
            reconstructed = np.fft.ifft(field, axis=0)
        np.testing.assert_array_equal(reconstructed, result.temporal_field())
        field[...] = 0
        assert np.any(result.field)
        assert np.any(hdf['Eomega'][()])
    if result.field.ndim == 3:
        assert result.parameters['responses']
        assert result.metadata['custom_response_calls'] > 0
    # Repeated writes close every file and preserve the existing overwrite API.
    result.save_npz(paths[0])
    result.save_hdf5(paths[1])
    with h5py.File(paths[1], 'r') as hdf:
        np.testing.assert_array_equal(hdf['Eomega'][()], result.field)


@pytest.mark.parametrize('method,extension', [('save_npz', 'npz'), ('save_hdf5', 'h5')])
@pytest.mark.parametrize('invalid', [lambda: None, 1+2j, np.nan, np.array([np.inf])])
def test_invalid_metadata_preserves_previous_file(tmp_path, monkeypatch, method, extension, invalid):
    if extension == 'h5':
        pytest.importorskip('h5py')
    from amalthea_native import EnvGrid, PropagationResult
    grid = EnvGrid(1., 800e-9, (600e-9, 1200e-9), 100e-15)
    result = PropagationResult(np.zeros((grid.omega.size, 2), complex), np.array([0., 1.]), grid, {}, {})
    monkeypatch.setitem(result.metadata, 'nested', {'invalid': invalid})
    destination = tmp_path/f'existing.{extension}'
    destination.write_bytes(b'previous result')
    with pytest.raises((TypeError, ValueError)):
        getattr(result, method)(destination)
    assert destination.read_bytes() == b'previous result'


def test_optional_dependency_is_lazy_and_base_simulation_works(tmp_path):
    code = r'''
import importlib.abc, pathlib, sys
class NoHdf5(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'h5py' or fullname.startswith('h5py.'):
            raise ModuleNotFoundError('optional h5py unavailable', name='h5py')
sys.meta_path.insert(0, NoHdf5())
import amalthea_native as an
assert 'h5py' not in sys.modules
result = an.prop_gnlse(.01, .001, [0, 0, -20e-27], lambda0=800e-9,
    lambda_lims=(600e-9, 1200e-9), trange=100e-15, tau_fwhm=20e-15,
    power=2000., saveN=3)
root = pathlib.Path(sys.argv[1])
result.save_npz(root/'base.npz')
assert 'h5py' not in sys.modules
try:
    result.save_hdf5(root/'missing.h5')
except ModuleNotFoundError as error:
    assert "pip install 'amalthea-native[hdf5]'" in str(error)
else:
    raise AssertionError('missing extra did not reject HDF5 output')
assert not (root/'missing.h5').exists()
print('base import, simulation and NPZ pass without h5py')
'''
    completed = subprocess.run([sys.executable, '-c', code, str(tmp_path)],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
