"""Carrier-resolved capillary propagation with and without third harmonics."""
import numpy as np
from amalthea_native import prop_capillary

options=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=300e-15,
             tau_fwhm=20e-15,energy=10e-6,plasma=False,saveN=21)
full=prop_capillary(125e-6,.02,'Ar',2.,**options)
no_thg=prop_capillary(125e-6,.02,'Ar',2.,**options,thg=False)
effect=np.linalg.norm(full.field-no_thg.field)/np.linalg.norm(full.field)
assert effect>1e-5
assert full.metadata['backend']=='native' and no_thg.metadata['backend']=='python'
assert full.temporal_field().shape==(full.grid.t.size,21)
assert not np.iscomplexobj(full.temporal_field())
print(f'Carrier capillary: {full.field.shape}, THG effect={effect:.6e}, '
      f'accepted steps={full.metadata["accepted_steps"]}')
