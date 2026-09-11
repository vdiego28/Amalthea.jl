"""Internal capillary envelope example; runs from an installed CPU wheel."""
import numpy as np
from amalthea_native import prop_capillary

parameters=dict(lambda0=800e-9,lambda_lims=(400e-9,1700e-9),trange=300e-15,
                tau_fwhm=20e-15,energy=10e-6,envelope=True,saveN=21)
result=prop_capillary(125e-6,.02,'Ar',2.,**parameters)
linear=prop_capillary(125e-6,.02,'Ar',2.,**parameters,kerr=False)
effect=np.linalg.norm(result.field-linear.field)/np.linalg.norm(linear.field)
assert effect>1e-5
assert result.metadata['backend']=='native'
assert result.metadata['stepper']=='rust-resident'
assert result.field.shape==(len(result.grid.omega),21)
assert np.all(np.isfinite(result.temporal_field()))
print(f'Capillary envelope: {result.field.shape}, Kerr effect={effect:.6e}, '
      f'accepted steps={result.metadata["accepted_steps"]}')
