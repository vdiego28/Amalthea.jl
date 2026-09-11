"""Complete custom-mode propagation with generic normalization and dispersion."""
import numpy as np
from amalthea_native import Mode, MarcatiliMode, prop_capillary


class CustomCapillary(Mode):
    def __init__(self, radius):
        self.reference = MarcatiliMode(radius, 'Ar', 2., n=2, m=1, phi=.17)

    def neff(self, omega, *, z=0.):
        return self.reference.neff(omega, z=z)

    def field(self, coordinates, *, z=0.):
        return self.reference.field(coordinates, z=z)

    def dimlimits(self, *, z=0.):
        return self.reference.dimlimits(z=z)


mode = CustomCapillary(lambda z: 125e-6*(1+.1*z))
for z in (0., .137):
    normalization_error = abs(mode.N(z=z)/mode.reference.N(z=z)-1)
    assert normalization_error < 1e-13
    beta1 = mode.dispersion(1, 2.35e15, z=z)
    assert beta1 == mode.reference.dispersion(1, 2.35e15, z=z)
    points = (np.array([0., 40e-6, 80e-6]), np.array([0., .4, 1.2]))
    physical_field = mode.Exy(points, z=z)
    assert physical_field.shape == (2, 3) and np.all(np.isfinite(physical_field))
    print(f'Custom HE21 at z={z:g}: normalization error={normalization_error:.3e}; beta1={beta1:.9e} s/m')

options=dict(lambda0=800e-9,lambda_lims=(200e-9,1700e-9),trange=80e-15,
             tau_fwhm=20e-15,energy=1e-6,envelope=True,plasma=False,raman=False,saveN=5)
result=prop_capillary(125e-6,1e-4,'Ar',2.,modes=[mode],**options)
reference=prop_capillary(lambda z:125e-6*(1+.1*z),1e-4,'Ar',2.,
                         modes=[dict(kind='HE',n=2,m=1,phi=.17)],**options)
error=np.linalg.norm(result.field-reference.field)/np.linalg.norm(reference.field)
assert error<1e-6 and result.field.shape[1:]==(1,5)
print(f'Custom propagation relative error={error:.3e}; {result.metadata["backend_reason"]}')
