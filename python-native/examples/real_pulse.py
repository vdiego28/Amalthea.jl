"""Prepare a few-cycle carrier-resolved pulse from an installed CPU wheel."""
import numpy as np
from amalthea_native import RealGrid, GaussPulse
from amalthea_native.pulses import energy_t

grid=RealGrid(.1,800e-9,(150e-9,1800e-9),100e-15)
pulse=GaussPulse(lambda0=800e-9,tau_fwhm=2.5e-15,energy=30e-9,phi=[1.1,3e-15,1e-30])
spectrum=pulse.spectrum(grid)
field=np.fft.irfft(spectrum,n=grid.t.size)
energy=energy_t(grid,field)
assert abs(energy/30e-9-1)<1e-13
assert spectrum.size==field.size//2+1
assert np.all(np.isfinite(field))
print(f'Carrier pulse: {field.size} time samples, {spectrum.size} frequency samples, energy={energy:.6e} J')
