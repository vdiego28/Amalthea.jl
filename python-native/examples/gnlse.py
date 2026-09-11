"""Julia-free pulses with Kerr, shock, dispersion, and both glass Raman models."""
import numpy as np

from amalthea_native import prop_gnlse

settings = dict(lambda0=800e-9, lambda_lims=(550e-9, 1700e-9), trange=400e-15,
                tau_fwhm=20e-15, power=2000., saveN=11)
linear = prop_gnlse(0., .03, [0, 0, -20e-27], **settings)
for model in ('sdo', 'SiO2'):
    result = prop_gnlse(.01, .03, [0, 0, -20e-27], **settings, ramanmodel=model)
    effect = np.linalg.norm(result.field - linear.field) / np.linalg.norm(linear.field)
    assert effect > .01
    assert np.all(np.isfinite(result.field))
    assert result.metadata['backend'] == 'native'
    assert result.metadata['stepper'] == 'rust-resident'
    assert result.parameters['ramanmodel'] == model
    print(f'{model}: field shape {result.field.shape}; nonlinear effect {effect:.3e}')
    print(f"Backend: {result.metadata['backend']}; stepper: {result.metadata['stepper']}")
