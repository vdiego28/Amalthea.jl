"""Complete-array Kerr and delayed response callbacks, in physical SI units."""
import numpy as np
from amalthea_native import prop_capillary
from amalthea_native.materials import EPS0, gamma3


def kerr(field, context):
    """Envelope vector Kerr, including coupling between x and y components."""
    coefficient = .75 * EPS0 * sum(gamma3(gas)*rho
                                   for gas, rho in zip(context.gases, context.densities))
    if field.shape[1] == 1:
        return coefficient * field * abs(field)**2
    x, y = field.T
    return coefficient * np.column_stack((
        (abs(x)**2 + 2/3*abs(y)**2)*x + np.conj(x)*y*y/3,
        (abs(y)**2 + 2/3*abs(x)**2)*y + np.conj(y)*x*x/3))


def delayed(field, context):
    """Illustrative causal response using the complete time array at each node."""
    dt = context.t[1] - context.t[0]
    lag = np.arange(len(field))*dt
    kernel = np.exp(-lag/12e-15)/12e-15
    intensity = np.sum(abs(field)**2, axis=1)
    history = np.convolve(intensity, kernel)[:len(field)]*dt
    coefficient = .1 * EPS0 * sum(gamma3(gas)*rho
                                  for gas, rho in zip(context.gases, context.densities))
    return coefficient*field*history[:, None]


options = dict(lambda0=800e-9, lambda_lims=(200e-9, 1700e-9), trange=100e-15,
               tau_fwhm=20e-15, energy=100e-6, envelope=True, modes=1,
               polarisation='circular', plasma=False, raman=False, saveN=7,
               init_dz=2.5e-6, max_dz=2.5e-6)
args = (125e-6, 1e-5, 'Ar', 2.)
reference = prop_capillary(*args, **options)
custom = prop_capillary(*args, **options, kerr=False, responses=kerr)
error = np.linalg.norm(custom.field-reference.field)/np.linalg.norm(reference.field)
assert error < 1e-13
combined = prop_capillary(*args, **options, kerr=False, responses=[kerr, delayed])
effect = np.linalg.norm(combined.field-custom.field)/np.linalg.norm(custom.field)
assert effect > 1e-5
assert combined.metadata['backend'] == 'python'
print(f'Custom Kerr error={error:.3e}; delayed effect={effect:.3e}; '
      f'response calls={combined.metadata["custom_response_calls"]}')
