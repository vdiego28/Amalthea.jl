"""Exact pressure gradients and callable tapers with molecular Raman."""
from tempfile import TemporaryDirectory
from pathlib import Path
import math

import numpy as np
from amalthea_native import prop_capillary

length = .0002
options = dict(lambda0=800e-9, lambda_lims=(200e-9,1700e-9), trange=300e-15,
               tau_fwhm=20e-15, energy=500e-6, plasma=False, saveN=7)

def radius(z):
    return 125e-6*(1+.1*math.sin(math.pi*z/length)+.05*(z/length)**2)

def pressure(z):
    return 2*(1+.4*math.sin(math.pi*z/length)+.1*(z/length)**2)

with TemporaryDirectory() as directory:
    for envelope in (True, False):
        baseline = prop_capillary(125e-6, length, 'N2', 2., **options, envelope=envelope)
        for name, fill in [('gradient', ([0., .00007, length], [2., 4., 1.])),
                           ('callable', pressure)]:
            result = prop_capillary(radius, length, 'N2', fill, **options, envelope=envelope)
            effect = np.linalg.norm(result.field-baseline.field)/np.linalg.norm(baseline.field)
            assert effect > 1e-5 and result.metadata['backend'] == 'python'
            output = Path(directory)/f'{envelope}-{name}.npz'
            result.save_npz(output)
            with np.load(output, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['Eomega'], result.field)
            print(f'N2 {name} + taper, envelope={envelope}: {result.field.shape}; effect={effect:.6e}')
