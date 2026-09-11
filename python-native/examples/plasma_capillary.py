"""Carrier-resolved ADK/PPT plasma with local setup and Rust stepping."""
from tempfile import TemporaryDirectory
import numpy as np
from amalthea_native import prop_capillary

options=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
             tau_fwhm=20e-15,energy=500e-6,raman=False,saveN=7)
control=prop_capillary(125e-6,.0002,'Ar',2.,**options,plasma=False)
with TemporaryDirectory() as directory:
    for plasma in ('ADK','PPT'):
        extra={'PPT_options':{'cachedir':directory}} if plasma=='PPT' else {}
        reference=None
        for backend in ('native','python'):
            result=prop_capillary(125e-6,.0002,'Ar',2.,**options,plasma=plasma,backend=backend,**extra)
            effect=np.linalg.norm(result.field-control.field)/np.linalg.norm(control.field)
            assert effect>1e-3 and result.metadata['backend']==backend
            if reference is not None:
                assert np.linalg.norm(result.field-reference)/np.linalg.norm(reference)<1e-13
            reference=result.field
            print(f'{plasma} plasma ({backend}): {result.field.shape}; effect={effect:.6e}; accepted={result.metadata["accepted_steps"]}')
