"""Multiple colors, a supplied spectrum, and a custom input propagator."""
import numpy as np
from amalthea_native import GaussPulse, SechPulse, DataPulse, prop_gnlse


def delay(field, grid):
    field *= np.exp(-1j*(grid.omega-grid.omega0)*30e-15)


omega=np.linspace(1.8e15,2.9e15,101)
spectrum=np.exp(-((omega-2.4e15)/1.5e14)**2).astype(complex)
inputs=[GaussPulse(lambda0=780e-9,tau_fwhm=24e-15,energy=30e-12),
        SechPulse(lambda0=880e-9,tau_w=15e-15,power=800.,propagator=delay),
        DataPulse(omega,spectrum,energy=10e-12)]
settings=dict(lambda0=800e-9,lambda_lims=(500e-9,1800e-9),trange=600e-15,
              pulses=inputs,saveN=11,ramanmodel='SiO2')
result=prop_gnlse(.01,.02,[0.,0.,-20e-27],**settings)
linear=prop_gnlse(0.,.02,[0.,0.,-20e-27],**settings)
effect=np.linalg.norm(result.field-linear.field)/np.linalg.norm(linear.field)
assert effect>1e-3 and np.all(np.isfinite(result.field))
assert result.metadata['backend']=='native'
print(f'Multi-input GNLSE: {result.field.shape}; nonlinear effect {effect:.3e}')
