"""Complete Julia-free nonlinear solve with an independent analytic check."""
import numpy as np

from amalthea_native import solve_precon

initial = np.array([1.0 + 0j, .5 + .1j])
linear = np.array([.13 + .2j, -.1 + .4j])
result = solve_precon(lambda z, field: field**2, linear, initial, .4,
                     dt=.4, rtol=1e-10, saveN=41)
z = result.z
reference = initial[:, None]*np.exp(linear[:, None]*z) / (
    1-initial[:, None]*np.expm1(linear[:, None]*z)/linear[:, None]
)
relative_error = np.linalg.norm(result.field-reference)/np.linalg.norm(reference)
assert relative_error < 1e-6
assert result.metadata["rejected_steps"] > 0
print(f"Relative error: {relative_error:.3e}")
print(f"Accepted: {result.metadata['accepted_steps']}; rejected: {result.metadata['rejected_steps']}")
