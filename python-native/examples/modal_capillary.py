"""Two-mode transfer and a polarized carrier pulse, with owned modal outputs."""
import numpy as np
from amalthea_native import GaussPulse,SechPulse,prop_capillary

options=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=100e-15,
             plasma=False,raman=False,saveN=7,init_dz=2.5e-6,max_dz=2.5e-6)
pulses=[GaussPulse(lambda0=800e-9,tau_fwhm=20e-15,energy=400e-6,mode='HE11'),
        SechPulse(lambda0=790e-9,tau_fwhm=17e-15,energy=100e-6,mode='HE12',phi=[.2,12e-15])]
result=prop_capillary(125e-6,1e-5,'Ar',2.,modes=2,pulses=pulses,envelope=True,**options)
assert result.field.shape[1:]==(2,7)
assert np.linalg.norm(result.field[:,:,-1]-result.field[:,:,0])/np.linalg.norm(result.field[:,:,0])>1e-5
print('Two-mode envelope:',result.field.shape,result.metadata['backend_reason'])
polarized=prop_capillary(125e-6,1e-5,'Ar',2.,modes='HE11',envelope=False,
                         polarisation='circular',tau_fwhm=20e-15,energy=100e-6,**options)
assert polarized.temporal_field().shape[1:]==(2,7)
assert np.all(np.isfinite(polarized.field))
print('Circular carrier:',polarized.field.shape,polarized.metadata['backend_reason'])
