"""Save, read and process both APIs; install amalthea-native[hdf5] first."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from amalthea_native import prop_capillary, prop_gnlse
from amalthea_native.pulses import energy_t


common = dict(lambda0=800e-9, lambda_lims=(600e-9, 1200e-9), trange=100e-15,
              tau_fwhm=20e-15, saveN=5)
gnlse = prop_gnlse(.01, .003, [0, 0, -20e-27], power=2000., **common)
capillary = prop_capillary(125e-6, 1e-5, 'Ar', 2., modes=2, energy=100e-6,
                          plasma=False, raman=False, init_dz=5e-6, max_dz=5e-6,
                          **common)
with TemporaryDirectory(prefix='amalthea-results-') as directory:
    root = Path(directory)
    for name, result in [('gnlse', gnlse), ('capillary', capillary)]:
        result.save_npz(root/f'{name}.npz')
        result.save_hdf5(root/f'{name}.h5')
        with np.load(root/f'{name}.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['Eomega'], result.field)
        with h5py.File(root/f'{name}.h5', 'r') as saved:
            grid = json.loads(saved['grid'].asstr()[()])
            metadata = json.loads(saved['metadata'].asstr()[()])
            # Select only the final saved position without loading all fields.
            spectrum = saved['Eomega'][..., -1]
            np.testing.assert_array_equal(spectrum, result.field[..., -1])
            field = (np.fft.irfft(spectrum, n=saved['t'].size, axis=0)
                     if grid['is_real'] else np.fft.ifft(spectrum, axis=0))
            reference = result.temporal_field()[..., -1]
            assert field.shape == reference.shape and np.all(np.isfinite(field))
            # Different FFT batch shapes may round differently; file values stay exact.
            fft_error = np.linalg.norm(field-reference)/max(np.linalg.norm(reference), 1e-300)
            assert fft_error < 1e-13
            print(f'{name}: sliced/full FFT relative error={fft_error:.6g}')
        columns = field[:, None] if field.ndim == 1 else field
        energies = [energy_t(result.grid, column) for column in columns.T]
        assert all(np.isfinite(energy) and energy >= 0 for energy in energies)
        print(f'{name}: axes={grid["field_axes"]}, final energies (J)={energies}')
        print(f'  backend={metadata["backend"]}; stepper={metadata["stepper"]}; '
              f'accepted={metadata["accepted_steps"]}; rejected={metadata["rejected_steps"]}')
