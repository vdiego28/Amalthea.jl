"""Molecular rotation/vibration on both grids, with optional carrier plasma."""
from tempfile import TemporaryDirectory

import numpy as np
from amalthea_native import prop_capillary

options = dict(lambda0=800e-9, lambda_lims=(200e-9, 1700e-9), trange=300e-15,
               tau_fwhm=20e-15, energy=500e-6, saveN=7)

with TemporaryDirectory() as directory:
    for envelope, plasma in ((True, False), (False, False), (False, 'PPT')):
        extra = {'PPT_options': {'cachedir': directory}} if plasma else {}
        control = prop_capillary(125e-6, .0002, 'N2', 2., **options, **extra,
                                 envelope=envelope, plasma=plasma, raman=False)
        reference = None
        for backend in ('native', 'python'):
            result = prop_capillary(125e-6, .0002, 'N2', 2., **options, **extra,
                                    envelope=envelope, plasma=plasma, backend=backend)
            effect = np.linalg.norm(result.field-control.field)/np.linalg.norm(control.field)
            assert effect > 1e-5 and result.metadata['backend'] == backend
            if reference is not None:
                assert np.linalg.norm(result.field-reference)/np.linalg.norm(reference) < 1e-13
            reference = result.field
            print(f'N2 Raman ({backend}, envelope={envelope}, plasma={plasma}): '
                  f'{result.field.shape}; effect={effect:.6e}')
